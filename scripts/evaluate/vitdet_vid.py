#!/usr/bin/env python3

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

from datasets.vid import VIDResize, VID, ROOT_DATA_PATH
from models.vitdet import ViTDet
from utils.config import initialize_run
from utils.evaluate import run_evaluations
from utils.misc import dict_to_device, squeeze_dict
import sys


def evaluate_vitdet_metrics(device, model, data, config):
    model.counting()
    model.clear_counts()
    n_frames = 0
    outputs = []
    labels = []
    n_items = config.get("n_items", len(data))

    for _, vid_item in tqdm(zip(range(n_items), data), total=n_items, ncols=0):
        vid_item = DataLoader(vid_item, batch_size=1)
        n_frames += len(vid_item)
        model.reset()
        for t, (frame, annotations) in enumerate(vid_item):
            if model.backbone.drop_using_rollout or model.backbone.force_compute_rollout:
                with torch.inference_mode():
                    frame_output, heatmaps, tome_clusters_maps, active_tokens_ind = model(frame.to(device))
                    outputs.extend(frame_output)
            else:
                with torch.inference_mode():
                    outputs.extend(model(frame.to(device)))
            labels.append(squeeze_dict(dict_to_device(annotations, device), dim=0))

    # MeanAveragePrecision is extremely slow. It seems fastest to call
    # update() and compute() just once, after all predictions are done.
    mean_ap = MeanAveragePrecision()
    mean_ap.update(outputs, labels)
    metrics = mean_ap.compute()

    counts = model.total_counts() / n_frames
    model.clear_counts()
    return {"metrics": metrics, "counts": counts}


def main(argv):
    config = initialize_run(config_location=Path("configs", "evaluate", "vitdet_vid"))
    long_edge = max(config["model"]["input_shape"][-2:])
    frames_stride = config.get("frames_stride", None)
    data = VID(
        Path(ROOT_DATA_PATH, "vid"),
        split=config["split"],
        tar_path=Path(ROOT_DATA_PATH, "vid", "vid_data.tar"),
        combined_transform=VIDResize(
            short_edge_length=640 * long_edge // 1024, max_size=long_edge
        ),
        sampling_period=frames_stride,
    )
    run_evaluations(config, ViTDet, data, evaluate_vitdet_metrics)


if __name__ == "__main__":
    main(sys.argv)
