
## Overview

PyTorch code for our paper "Rollout-Guided Token Pruning for Efficient Video Understanding". 

## Code Structure

Scripts should be run from the repo's base directory.

Many scripts expect a `.yml` configuration file as a command-line argument. These configuration files are in `configs`. The structure of the `configs` folder is set to mirror the structure of the `scripts` folder. For example, to run the `base_672` evaluation for the ViTDet VID model:
```
./scripts/evaluate/vitdet_vid.py ./configs/evaluate/vitdet_vid/base_672.yml
```
This repository is based on the code of [Eventful Transformers](https://github.com/WISION-Lab/eventful-transformer). We thank the authors for their excellent code and documentation.

## Installation

Dependencies are managed using Conda. The environment is defined in `environment.yml`.

To create the environment, run:
```
conda env create -f environment.yml
```
Then activate the environment with:
```
conda activate rgtp
```
## Other Setup

Scripts assume that the current working directory is on the Python path. In the Bash shell, run
```
export PYTHONPATH="$PYTHONPATH:."
```
Or in the Fish shell:
```
set -ax PYTHONPATH .
```

## Weights

Weights for the ViViT action recognition model (on Kinetics-400 and EPIC-Kitchens) are available [here](https://github.com/alibaba-mmai-research/TAdaConv/blob/main/MODEL_ZOO.md). We use the "ViViT Fact. Enc." weights.

Weights for the ViTDet object detection model (on COCO) are available [here](https://github.com/facebookresearch/detectron2/tree/main/projects/ViTDet). We use the "Cascade Mask R-CNN, ViTDet, ViT-B" weights. Weights on ImageNet VID are available [here](https://drive.google.com/drive/folders/1tNtIOYlCIlzb2d_fCsIbmjgIETd-xzW-) (`frcnn_vitdet_final.pth`).

The weight names need to be remapped to work with this codebase. To remap the ViViT weights, run:
```
./scripts/convert/vivit.py <old_weights> <new_weights> ./configs/convert/vivit_b.txt
```
with `<old_weights>` and `<new_weigtht>` replaced by the path of the downloaded weights and the path where the converted weights should be saved, respectively.

To remap the ViTDet weights, run:
```
./scripts/convert/vitdet.py <old_weights> <new_weights> ./configs/convert/vitdet_b.txt
```

Some ViViT evaluation scripts assume a fine-tuned temporal sub-model. Fine-tuned weights can be downloaded [here](https://drive.google.com/drive/folders/rgtb_fint_tuned_vivit).

Alternatively, you can run the fine-tuning yourself. To do this, run a `spatial` configuration (to cache the forward pass of the spatial sub-model), followed by a `train` configuration. For example:
```
./scripts/spatial/vivit_kinetics400.py ./configs/spatial/vivit_kinetics400/rollout_reset_1in2_frq_8_keep_0215.yml
```
then
```
./scripts/train/vivit_kinetics400.py tokendrop_rollout_only spatial_cache_suffix=_rollout_reset_1in2_frq_8_keep_0215
```
This will produce `weights/vivit_b_kinetics400_rollout_reset_1in2_frq_8_keep_0215.pth`.

## Data

The `datasets` folder defines PyTorch `Dataset` classes for Kinetics-400, VID, and EPIC-Kitchens.

The Kinetics-400 class will automatically download and prepare the dataset on first use.

VID requires a manual download. Download `vid_data.tar` from [here](https://drive.google.com/drive/folders/1tNtIOYlCIlzb2d_fCsIbmjgIETd-xzW-) and place it at `./data/vid/data.tar`. The VID class will take care of unpacking and preparing the data on first use.

EPIC-Kitchens also requires a manual download. Download the videos from [here](https://drive.google.com/drive/folders/1OKJpgSKR1QnWa2tMMafknLF-CpEaxDbY) and place them in `./data/epic_kitchens/videos`. Download the labels `EPIC_100_train.csv` and `EPIC_100_validation.csv` from [here](https://github.com/epic-kitchens/epic-kitchens-100-annotations) and place them in `./data/epic_kitchens`. The EPICKitchens class will prepare the data on first use.

## Results

Below are the main results from the paper, including the configurations needed to reproduce them.

### Video OD on ImageNet VID (Size 672)

| **mAP50(%)** | **GFLOPs**   | **Configuration** |
| :----------- | :----------- | :---------------- |
| 82.28        | 174.5        | [baseline](https://github.com/RGTPdyn/RGTP/blob/main/configs/evaluate/vitdet_vid/base_672.yml)            |
| 82.37        | 93.4         | [res672_keep768](https://github.com/RGTPdyn/RGTP/blob/main/configs/evaluate/vitdet_vid/rollout_tokenwise_frq8_res672_keep768.yml)            |
| 82.39        | 72.6         | [res672_keep512](https://github.com/RGTPdyn/RGTP/blob/main/configs/evaluate/vitdet_vid/rollout_tokenwise_frq8_res672_keep512.yml)            |
| 82.38        | 62.1         | [res672_keep384](https://github.com/RGTPdyn/RGTP/blob/main/configs/evaluate/vitdet_vid/rollout_tokenwise_frq8_res672_keep384.yml)            |
| 82.06        | 46.8         | [res672_keep196](https://github.com/RGTPdyn/RGTP/blob/main/configs/evaluate/vitdet_vid/rollout_tokenwise_frq8_res672_keep196.yml)            |
| 80.53        | 36.0         | [res672_keep64](https://github.com/RGTPdyn/RGTP/blob/main/configs/evaluate/vitdet_vid/rollout_tokenwise_frq8_res672_keep64.yml)            |


### Video OD on ImageNet VID (Size 1024)

| **mAP50(%)** | **GFLOPs**   | **Configuration** |
| :----------- | :----------- | :---------------- |
| 82.93        | 467.4        | [baseline](https://github.com/RGTPdyn/RGTP/blob/main/configs/evaluate/vitdet_vid/base_1024.yml)            |
| 83.08        | 288.9        | [res1024_keep2200](https://github.com/RGTPdyn/RGTP/blob/main/configs/evaluate/vitdet_vid/rollout_tokenwise_frq8_res1024_keep2200.yml)            |
| 83.04        | 213.7        | [res1024_keep1400](https://github.com/RGTPdyn/RGTP/blob/main/configs/evaluate/vitdet_vid/rollout_tokenwise_frq8_res1024_keep1400.yml)            |
| 83.04        | 152.4        | [res1024_keep750](https://github.com/RGTPdyn/RGTP/blob/main/configs/evaluate/vitdet_vid/rollout_tokenwise_frq8_res1024_keep750.yml)            |
| 82.68        | 119.1        | [res1024_keep400](https://github.com/RGTPdyn/RGTP/blob/main/configs/evaluate/vitdet_vid/rollout_tokenwise_frq8_res1024_keep400.yml)            |
| 80.03        | 88.8         | [res1024_keep75](https://github.com/RGTPdyn/RGTP/blob/main/configs/evaluate/vitdet_vid/rollout_tokenwise_frq8_res1024_keep75.yml)            |


### Action Recognition on EPIC-Kitchens

| **Accuracy(%)** | **TFLOPs**  | **Configuration** |
| :----------- | :------------- | :---------------- |
| 67.14        | 7.12           | [baseline](https://github.com/RGTPdyn/RGTP/blob/main/configs/evaluate/vivit_epic_kitchens/base.yml)            |
| 68.13        | 5.51           | [fg_300_bg_025](https://github.com/RGTPdyn/RGTP/blob/main/configs/evaluate/vivit_epic_kitchens/rollout_reset_frq_4_w_offset_fg_300_bg_025_wo_ft.yml)            |
| 67.70        | 4.04           | [fg_180_bg_016](https://github.com/RGTPdyn/RGTP/blob/main/configs/evaluate/vivit_epic_kitchens/rollout_reset_frq_4_w_offset_fg_180_bg_016.yml)            |
| 67.40        | 2.97           | [fg_120_bg_005](https://github.com/RGTPdyn/RGTP/blob/main/configs/evaluate/vivit_epic_kitchens/rollout_reset_frq_4_w_offset_fg_120_bg_005_wo_ft.yml)            |
| 66.35        | 2.25           | [fg_160s05_bg_009](https://github.com/RGTPdyn/RGTP/blob/main/configs/evaluate/vivit_epic_kitchens/rollout_reset_frq_4_w_offset_fg_160s05_bg_009.yml)            |
| 64.27        | 1.71           | [fg_96s05_bg_003](https://github.com/RGTPdyn/RGTP/blob/main/configs/evaluate/vivit_epic_kitchens/rollout_reset_frq_4_w_offset_fg_96s05_bg_003_wo_ft.yml)            |
| 63.79        | 1.34           | [fg_28s05_bg_00](https://github.com/RGTPdyn/RGTP/blob/main/configs/evaluate/vivit_epic_kitchens/rollout_reset_frq_4_w_offset_fg_28s05_bg_00.yml)            |

### Action Recognition on Kinetics-400

| **Accuracy(%)** | **TFLOPs**  | **Configuration** |
| :----------- | :------------- | :---------------- |
| 79.06        | 3.36           | [baseline](https://github.com/RGTPdyn/RGTP/blob/main/configs/evaluate/vivit_kinetics400/base.yml)  
| 78.00        | 1.79           | [keep_049](https://github.com/RGTPdyn/RGTP/blob/main/configs/evaluate/vivit_kinetics400/rollout_reset_1in2_frq_8_keep_049.yml)            |
| 75.93        | 1.01           | [keep_0215](https://github.com/RGTPdyn/RGTP/blob/main/configs/evaluate/vivit_kinetics400/rollout_reset_1in2_frq_8_keep_0215.yml.yml)            |
| 75.50        | 0.61           | [keep_0066](https://github.com/RGTPdyn/RGTP/blob/main/configs/evaluate/vivit_kinetics400/rollout_reset_1in2_frq_8_keep_0066.yml)            |

