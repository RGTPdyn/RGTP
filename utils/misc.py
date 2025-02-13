import re
import subprocess
from pathlib import Path
from random import Random

import requests
import torch

from algorithms.modules import SimpleSTGTGate, TokenDeltaGate, TokenGate


class MeanValue:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def compute(self):
        return 0.0 if (self.count == 0) else self.sum / self.count

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value):
        self.sum += value
        self.count += 1


class TopKAccuracy:
    def __init__(self, k):
        self.k = k
        self.correct = 0
        self.total = 0

    def compute(self):
        return self.correct / self.total

    def reset(self):
        self.correct = 0
        self.total = 0

    def update(self, pred, true):
        _, top_k = pred.topk(self.k, dim=-1)
        self.correct += true.eq(top_k).sum().item()
        self.total += true.numel()


def decode_video(
    input_path,
    output_path,
    name_format="%d",
    image_format="png",
    ffmpeg_input_args=None,
    ffmpeg_output_args=None,
):
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)
    if ffmpeg_input_args is None:
        ffmpeg_input_args = []
    if ffmpeg_output_args is None:
        ffmpeg_output_args = []
    return subprocess.call(
        ["ffmpeg", "-loglevel", "error"]
        + ffmpeg_input_args
        + ["-i", input_path]
        + ffmpeg_output_args
        + [output_path / f"{name_format}.{image_format}"]
    )


def dict_to_device(x, device):
    return {key: value.to(device) for key, value in x.items()}


# https://gist.github.com/wasi0013/ab73f314f8070951b92f6670f68b2d80
def download_file(url, output_path, chunk_size=4096, verbose=True):
    if verbose:
        print(f"Downloading {url}...", flush=True)
    with requests.get(url, stream=True) as source:
        with open(output_path, "wb") as output_file:
            for chunk in source.iter_content(chunk_size=chunk_size):
                if chunk:
                    output_file.write(chunk)


def get_device_description(device):
    if device == "cuda":
        return torch.cuda.get_device_name()
    else:
        return f"CPU with {torch.get_num_threads()} threads"


def get_pytorch_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_patterns(pattern_file):
    patterns = []
    last_regex = None
    with open(pattern_file, "r") as text:
        for line in text:
            line = line.strip()
            if line == "":
                continue
            elif last_regex is None:
                last_regex = re.compile(line)
            else:
                patterns.append((last_regex, line))
                last_regex = None
    return patterns


def remap_weights(in_weights, patterns, verbose=False):
    n_remapped = 0
    out_weights = {}
    for in_key, weight in in_weights.items():
        out_key = in_key
        discard = False
        for regex, replacement in patterns:
            out_key, n_matches = regex.subn(replacement, out_key)
            if n_matches > 0:
                if replacement == "DISCARD":
                    discard = True
                    out_key = "DISCARD"
                n_remapped += 1
                if verbose:
                    print(f"{in_key}  ==>  {out_key}")
                break
        if not discard:
            out_weights[out_key] = weight
    return out_weights, n_remapped


def seeded_shuffle(sequence, seed):
    rng = Random()
    rng.seed(seed)
    rng.shuffle(sequence)


def set_policies(model, policy_class, **policy_kwargs):
    for gate_class in [SimpleSTGTGate, TokenDeltaGate, TokenGate]:
        for gate in model.modules_of_type(gate_class):
            gate.policy = policy_class(**policy_kwargs)


def squeeze_dict(x, dim=None):
    return {key: value.squeeze(dim=dim) for key, value in x.items()}


def tee_print(s, file, flush=True):
    print(s, flush=flush)
    print(s, file=file, flush=flush)


def generate_maskvd_heatmap(output_mask_size, images_input_size, num_tokens_to_keep, bboxes,
                            bboxes_scores, score_thr, gilbert_lut, add_noise):
    boxes_mask = torch.zeros(output_mask_size[0], output_mask_size[1], dtype=torch.bool, device=bboxes.device)
    valid_boxes = torch.greater(bboxes_scores, score_thr)
    if torch.any(valid_boxes):
        boxes = bboxes[valid_boxes, :]
        ds_factor = output_mask_size[0] / images_input_size[0]
        ds_boxes = boxes * ds_factor
        ds_boxes[:, :2] = torch.floor(ds_boxes[:, :2])
        ds_boxes[:, 2:] = torch.ceil(ds_boxes[:, 2:])
        ds_boxes = torch.clip(ds_boxes, 0.0, output_mask_size[0] - 1).int()
        for box in ds_boxes:
            boxes_mask[box[1]:box[3], box[0]:box[2]] = True
        num_active_in_boxes_mask = torch.sum(boxes_mask)
        if num_active_in_boxes_mask == num_tokens_to_keep:
            sample_mask = boxes_mask.flatten()
        elif num_active_in_boxes_mask < num_tokens_to_keep:
            inds_outside_boxes_mask = torch.where(boxes_mask.flatten() == 0)[0]
            num_to_sample_outside = num_tokens_to_keep - num_active_in_boxes_mask
            sample_idx = torch.round(torch.linspace(0, inds_outside_boxes_mask.shape[0] - 1, num_to_sample_outside)).to(
                torch.int64)
            if gilbert_lut is not None:
                # sample using glibert
                gilbert_lut_t = torch.tensor(gilbert_lut, device=inds_outside_boxes_mask.device)
                inds_outside_boxes_curve_inds = gilbert_lut_t[inds_outside_boxes_mask]
                inds_outside_boxes_sort_inds = torch.argsort(inds_outside_boxes_curve_inds)
                sample_outside_idx = inds_outside_boxes_sort_inds[sample_idx]
                inds_to_sample_outside = inds_outside_boxes_mask[sample_outside_idx]
            else:
                # stride sampling
                inds_to_sample_outside = inds_outside_boxes_mask[sample_idx]
            sample_mask = boxes_mask.flatten()
            sample_mask[inds_to_sample_outside] = True
        else:
            inds_inside_boxes_mask = torch.where(boxes_mask.flatten() == 1)[0]
            sample_idx = torch.round(torch.linspace(0, inds_inside_boxes_mask.shape[0] - 1, num_tokens_to_keep)).to(
                torch.int64)
            if gilbert_lut is not None:
                # sample using glibert
                gilbert_lut_t = torch.tensor(gilbert_lut, device=inds_inside_boxes_mask.device)
                inds_inside_boxes_curve_inds = gilbert_lut_t[inds_inside_boxes_mask]
                inds_inside_boxes_sort_inds = torch.argsort(inds_inside_boxes_curve_inds)
                sample_inside_idx = inds_inside_boxes_sort_inds[sample_idx]
                inds_to_sample_inside = inds_inside_boxes_mask[sample_inside_idx]
            else:
                # stride sampling
                inds_to_sample_inside = inds_inside_boxes_mask[sample_idx]
            sample_mask = torch.zeros_like(boxes_mask.flatten())
            sample_mask[inds_to_sample_inside] = True
    else:
        sample_idx = torch.round(torch.linspace(0, output_mask_size[0] * output_mask_size[1] - 1, num_tokens_to_keep)).to(
            torch.int64)
        sample_mask = torch.zeros_like(boxes_mask.flatten())
        sample_mask[sample_idx] = True
    heatmap = torch.zeros(1, sample_mask.flatten().shape[0], dtype=torch.float32, device=bboxes.device)
    heatmap[0, sample_mask] = 1 / torch.sum(sample_mask)
    if add_noise:
        max_bg_val = 1 / (10 * torch.sum(sample_mask))
        noise_vec = torch.rand(heatmap.shape[0], heatmap.shape[1], device=heatmap.device) * max_bg_val
        heatmap = heatmap + noise_vec
    return heatmap
