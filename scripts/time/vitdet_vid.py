#!/usr/bin/env python3

import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.vid import VIDResize, VID, ROOT_DATA_PATH
from models.vitdet import ViTDet
from utils.config import initialize_run
from utils.evaluate import run_evaluations
from utils.misc import MeanValue
from utils.rollout_utils import create_rollout_anchors_mask
import sys
import numpy as np


def evaluate_vitdet_runtime(device, model, data, config):
    model.no_counting()
    backbone = MeanValue()
    backbone_non_first = MeanValue()
    other = MeanValue()
    other_non_first = MeanValue()
    rollout_overhead = MeanValue()
    rollout_overhead_non_first = MeanValue()
    n_items = config.get("n_items", len(data))
    if model.backbone.drop_using_rollout:
        blocks_attention = None
        blocks_samples_idx = None
        active_tokens_ind = None
        tome_clusters_maps = None
        heatmaps = None
        t_min1_heatmaps = None
        for _, vid_item in tqdm(zip(range(n_items), data), total=n_items, ncols=0):
            vid_item = DataLoader(vid_item, batch_size=1)
            model.reset()
            t_min1_metric = None
            for t, (frame, annotations) in enumerate(vid_item):
                with torch.inference_mode():
                    frame = frame.to(device)
                    torch.cuda.synchronize()
                    t_0 = time.time()
                    images, x = model.pre_backbone(frame)
                    torch.cuda.synchronize()
                    t_1 = time.time()
                    x, heatmaps, tome_clusters_maps, active_tokens_ind, blocks_attention, blocks_samples_idx, metric_t = \
                        model.backbone(x, t == 0, t_min1_heatmaps, t, t_min1_metric)
                    torch.cuda.synchronize()
                    t_2 = time.time()
                    results = model.post_backbone(images, x)
                    torch.cuda.synchronize()
                    t_3 = time.time()
                    if model.backbone.rollout_by_bbox:
                        anchors_mask = create_rollout_anchors_mask(results[0]["boxes"], results[0]["scores"],
                                                                   images[0].shape[-2:], model.backbone_input_size,
                                                                   model.backbone.rollout_bbox_score_thr, x.device)
                        heatmaps = model.backbone.generate_rollout_heatmap(blocks_attention, blocks_samples_idx,
                                                                           x.shape[0], x.shape[1], x.device,
                                                                           active_tokens_ind, tome_clusters_maps,
                                                                           t == 0, t_min1_heatmaps, anchors_mask)
                    t_min1_metric = None if metric_t is None else metric_t.clone().detach()
                    t_min1_heatmaps = heatmaps.clone().detach()
                    t_4 = time.time()
                    t_backbone = t_2 - t_1
                    t_other = (t_3 - t_2) + (t_1 - t_0)
                    t_rollout_overhead = t_4 - t_3
                    backbone.update(t_backbone)
                    other.update(t_other)
                    rollout_overhead.update(t_rollout_overhead)
                    if t > 0:
                        backbone_non_first.update(t_backbone)
                        other_non_first.update(t_other)
                        rollout_overhead_non_first.update(t_rollout_overhead)
    else:
        for _, vid_item in tqdm(zip(range(n_items), data), total=n_items, ncols=0):
            vid_item = DataLoader(vid_item, batch_size=1)
            model.reset()
            for t, (frame, annotations) in enumerate(vid_item):
                with torch.inference_mode():
                    frame = frame.to(device)
                    torch.cuda.synchronize()
                    t_0 = time.time()
                    images, x = model.pre_backbone(frame)
                    torch.cuda.synchronize()
                    t_1 = time.time()
                    x = model.backbone(x)
                    torch.cuda.synchronize()
                    t_2 = time.time()
                    model.post_backbone(images, x)
                    torch.cuda.synchronize()
                    t_3 = time.time()
                    t_backbone = t_2 - t_1
                    t_other = (t_3 - t_2) + (t_1 - t_0)
                    backbone.update(t_backbone)
                    other.update(t_other)
                    if t > 0:
                        backbone_non_first.update(t_backbone)
                        other_non_first.update(t_other)
    times = {
        "backbone": backbone.compute(),
        "backbone_non_first": backbone_non_first.compute(),
        "other": other.compute(),
        "other_non_first": other_non_first.compute(),
        "rollout_overhead": rollout_overhead.compute(),
        "rollout_overhead_non_first": rollout_overhead_non_first.compute(),
        "total": backbone.compute() + other.compute() + rollout_overhead.compute(),
        "total_non_first": backbone_non_first.compute() + other_non_first.compute() +
                           rollout_overhead_non_first.compute(),
        "ms_to_report": np.around(1000 * (backbone_non_first.compute() + rollout_overhead_non_first.compute()), 1),
    }
    return {"times": times}


def main(argv):
    config = initialize_run(config_location=Path("configs", "time", "vitdet_vid"))
    input_size = config.get("input_size", 1024)
    data = VID(
        Path(ROOT_DATA_PATH, "vid"),
        split=config["split"],
        tar_path=Path(ROOT_DATA_PATH, "vid", "data.tar"),
        combined_transform=VIDResize(
            short_edge_length=640 * input_size // 1024, max_size=input_size
        ),
    )
    run_evaluations(config, ViTDet, data, evaluate_vitdet_runtime)


if __name__ == "__main__":
    main(sys.argv)
