# Rollout-Guided Token Pruning for Efficient Video Understanding

![ICIP_fig1_v3-1](https://github.com/RGTPdyn/RGTP/blob/main/ICIP_fig1_v3.png)

An official PyTorch code for our paper **"Rollout-Guided Token Pruning for Efficient Video Understanding"** (ICIP'25):

<p align="justify"> 
Vision Transformers have been proven powerful in various vision applications. Yet, their adaptations for video understanding tasks incur large computational costs, limiting their practical deployment on resource-constrained devices. Token pruning can effectively alleviate the processing overhead of underlying attention blocks, but often neglects the iterative processing nature of video models applied frame-by-frame. We propose to prune tokens according to the estimated contribution of their corresponding tokens in previous frames to previous predictions. We leverage attention rollout and token tracking to propagate token importance of previous outputs to current input tokens. Our method is interpretable, requires no training and has negligible memory overhead. We show the efficacy of our method for both video object detection and action recognition using different transformer architectures, achieving up to 65\% reduction in FLOPS on ImageNet VID and 60\% on EPIC-Kitchens with no accuracy degradation.
</p>

It builds on the [Eventful Transformers](https://github.com/WISION-Lab/eventful-transformer) codebase—our thanks to the original authors.

## Table of Contents

- [Overview](#overview)
- [Code Structure](#code-structure)
- [Installation](#installation)
- [Environment Setup](#environment-setup)
- [Weights](#weights)
- [Data Preparation](#data-preparation)
- [Results](#results)
- [License](#license)

## Overview

This repository implements efficient video understanding via Rollout-Guided Token Pruning. It includes scripts for evaluation, training, spatial caching, and weight conversion.

## Code Structure

- **Scripts:** All scripts must be executed from the repository’s root.
- **Configuration Files:** Located in the `configs` folder. Their structure mirrors that of the `scripts` directory.

> **Example:** To evaluate the ViTDet VID model using the `base_672` configuration:
> ```bash
> ./scripts/evaluate/vitdet_vid.py ./configs/evaluate/vitdet_vid/base_672.yml
> ```

## Installation

Dependencies are managed with Conda. The environment is defined in `environment.yml`.

```bash
conda env create -f environment.yml
conda activate rgtp
```

## Environment Setup

Ensure the current directory is on your Python path.

- **Bash:**
  ```bash
  export PYTHONPATH="$PYTHONPATH:."
  ```
- **Fish:**
  ```fish
  set -ax PYTHONPATH .
  ```

## Weights

### ViViT Action Recognition Model

- **Sources:** [TAdaConv MODEL_ZOO](https://github.com/alibaba-mmai-research/TAdaConv/blob/main/MODEL_ZOO.md)
- **Weights:** Use the "ViViT Fact. Enc." weights.
- **Conversion:** Remap weights using:
  ```bash
  ./scripts/convert/vivit.py <old_weights> <new_weights> ./configs/convert/vivit_b.txt
  ```
  Replace `<old_weights>` and `<new_weights>` with the paths to the downloaded and converted weights, respectively.

### ViTDet Object Detection Model

- **Sources:**
  - [COCO weights](https://github.com/facebookresearch/detectron2/tree/main/projects/ViTDet) ("Cascade Mask R-CNN, ViTDet, ViT-B")
  - [ImageNet VID weights](https://drive.google.com/drive/folders/1tNtIOYlCIlzb2d_fCsIbmjgIETd-xzW-) (`frcnn_vitdet_final.pth`)
- **Conversion:** Remap weights using:
  ```bash
  ./scripts/convert/vitdet.py <old_weights> <new_weights> ./configs/convert/vitdet_b.txt
  ```

### Fine-Tuning ViViT

Some evaluation scripts require a fine-tuned temporal sub-model. You can either download pre-fine-tuned weights [here](https://drive.google.com/drive/folders/1V7mYC5Lc4vdv26vnhs-ZOZgykmJfhHvX) or fine-tune manually:

1. **Cache the spatial sub-model forward pass:**
   ```bash
   ./scripts/spatial/vivit_kinetics400.py ./configs/spatial/vivit_kinetics400/rollout_reset_1in2_frq_8_keep_0215.yml
   ```
2. **Train:**
   ```bash
   ./scripts/train/vivit_kinetics400.py tokendrop_rollout_only spatial_cache_suffix=_rollout_reset_1in2_frq_8_keep_0215
   ```
   This produces `weights/vivit_b_kinetics400_rollout_reset_1in2_frq_8_keep_0215.pth`.

## Data Preparation

- **Kinetics-400:** Downloads and prepares automatically on first use.
- **ImageNet VID:**  
  - **Download:** Get `vid_data.tar` from [here](https://drive.google.com/drive/folders/1tNtIOYlCIlzb2d_fCsIbmjgIETd-xzW-).
  - **Placement:** Place it at `./data/vid/data.tar`—the dataset unpacks on first use.
- **EPIC-Kitchens:**
  - **Videos:** Download from [here](https://drive.google.com/drive/folders/1OKJpgSKR1QnWa2tMMafknLF-CpEaxDbY) and place them in `./data/epic_kitchens/videos`.
  - **Labels:** Download `EPIC_100_train.csv` and `EPIC_100_validation.csv` from [EPIC-Kitchens 100 annotations](https://github.com/epic-kitchens/epic-kitchens-100-annotations) and place them in `./data/epic_kitchens`.

## Results

The tables below summarize our main results and the configurations to reproduce them.

### Video Object Detection on ImageNet VID (Size 672)

| **mAP50 (%)** | **GFLOPs** | **Configuration** |
|---------------|------------|-------------------|
| 82.28         | 174.5      | [baseline](https://github.com/RGTPdyn/RGTP/blob/main/configs/evaluate/vitdet_vid/base_672.yml) |
| 82.37         | 93.4       | [res672_keep768](https://github.com/RGTPdyn/RGTP/blob/main/configs/evaluate/vitdet_vid/rollout_tokenwise_frq8_res672_keep768.yml) |
| 82.39         | 72.6       | [res672_keep512](https://github.com/RGTPdyn/RGTP/blob/main/configs/evaluate/vitdet_vid/rollout_tokenwise_frq8_res672_keep512.yml) |
| 82.38         | 62.1       | [res672_keep384](https://github.com/RGTPdyn/RGTP/blob/main/configs/evaluate/vitdet_vid/rollout_tokenwise_frq8_res672_keep384.yml) |
| 82.06         | 46.8       | [res672_keep196](https://github.com/RGTPdyn/RGTP/blob/main/configs/evaluate/vitdet_vid/rollout_tokenwise_frq8_res672_keep196.yml) |
| 80.53         | 36.0       | [res672_keep64](https://github.com/RGTPdyn/RGTP/blob/main/configs/evaluate/vitdet_vid/rollout_tokenwise_frq8_res672_keep64.yml) |

### Video Object Detection on ImageNet VID (Size 1024)

| **mAP50 (%)** | **GFLOPs** | **Configuration** |
|---------------|------------|-------------------|
| 82.93         | 467.4      | [baseline](https://github.com/RGTPdyn/RGTP/blob/main/configs/evaluate/vitdet_vid/base_1024.yml) |
| 83.08         | 288.9      | [res1024_keep2200](https://github.com/RGTPdyn/RGTP/blob/main/configs/evaluate/vitdet_vid/rollout_tokenwise_frq8_res1024_keep2200.yml) |
| 83.04         | 213.7      | [res1024_keep1400](https://github.com/RGTPdyn/RGTP/blob/main/configs/evaluate/vitdet_vid/rollout_tokenwise_frq8_res1024_keep1400.yml) |
| 83.04         | 152.4      | [res1024_keep750](https://github.com/RGTPdyn/RGTP/blob/main/configs/evaluate/vitdet_vid/rollout_tokenwise_frq8_res1024_keep750.yml) |
| 82.68         | 119.1      | [res1024_keep400](https://github.com/RGTPdyn/RGTP/blob/main/configs/evaluate/vitdet_vid/rollout_tokenwise_frq8_res1024_keep400.yml) |
| 80.03         | 88.8       | [res1024_keep75](https://github.com/RGTPdyn/RGTP/blob/main/configs/evaluate/vitdet_vid/rollout_tokenwise_frq8_res1024_keep75.yml) |

### Action Recognition on EPIC-Kitchens

| **Accuracy (%)** | **TFLOPs** | **Configuration** |
|------------------|------------|-------------------|
| 67.14            | 7.12       | [baseline](https://github.com/RGTPdyn/RGTP/blob/main/configs/evaluate/vivit_epic_kitchens/base.yml) |
| 68.13            | 5.51       | [fg_300_bg_025](https://github.com/RGTPdyn/RGTP/blob/main/configs/evaluate/vivit_epic_kitchens/rollout_reset_frq_4_w_offset_fg_300_bg_025_wo_ft.yml) |
| 67.70            | 4.04       | [fg_180_bg_016](https://github.com/RGTPdyn/RGTP/blob/main/configs/evaluate/vivit_epic_kitchens/rollout_reset_frq_4_w_offset_fg_180_bg_016.yml) |
| 67.40            | 2.97       | [fg_120_bg_005](https://github.com/RGTPdyn/RGTP/blob/main/configs/evaluate/vivit_epic_kitchens/rollout_reset_frq_4_w_offset_fg_120_bg_005_wo_ft.yml) |
| 66.35            | 2.25       | [fg_160s05_bg_009](https://github.com/RGTPdyn/RGTP/blob/main/configs/evaluate/vivit_epic_kitchens/rollout_reset_frq_4_w_offset_fg_160s05_bg_009.yml) |
| 64.27            | 1.71       | [fg_96s05_bg_003](https://github.com/RGTPdyn/RGTP/blob/main/configs/evaluate/vivit_epic_kitchens/rollout_reset_frq_4_w_offset_fg_96s05_bg_003_wo_ft.yml) |
| 63.79            | 1.34       | [fg_28s05_bg_00](https://github.com/RGTPdyn/RGTP/blob/main/configs/evaluate/vivit_epic_kitchens/rollout_reset_frq_4_w_offset_fg_28s05_bg_00.yml) |

### Action Recognition on Kinetics-400

| **Accuracy (%)** | **TFLOPs** | **Configuration** |
|------------------|------------|-------------------|
| 79.06            | 3.36       | [baseline](https://github.com/RGTPdyn/RGTP/blob/main/configs/evaluate/vivit_kinetics400/base.yml) |
| 78.00            | 1.79       | [keep_049](https://github.com/RGTPdyn/RGTP/blob/main/configs/evaluate/vivit_kinetics400/rollout_reset_1in2_frq_8_keep_049.yml) |
| 75.93            | 1.01       | [keep_0215](https://github.com/RGTPdyn/RGTP/blob/main/configs/evaluate/vivit_kinetics400/rollout_reset_1in2_frq_8_keep_0215.yml) |
| 75.50            | 0.61       | [keep_0066](https://github.com/RGTPdyn/RGTP/blob/main/configs/evaluate/vivit_kinetics400/rollout_reset_1in2_frq_8_keep_0066.yml) |

## License
This project is licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)** license.
[![CC BY-NC-SA 4.0](https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

