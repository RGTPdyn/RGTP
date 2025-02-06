# Rollout-Guided Token Pruning for Efficient Video Understanding
An official code of "Rollout-Guided Token Pruning for Efficient Video Understanding" paper:
<p align="justify"> 
  Vision Transformers have been proven powerful in various vision applications. Yet, their adaptations for video understanding tasks incur large computational costs, limiting their practical deployment on resource-constrained devices. Token pruning can effectively alleviate the processing overhead of underlying attention blocks, but often neglects the iterative processing nature of video models applied frame-by-frame. We propose to prune tokens according to the estimated contribution of their corresponding tokens in previous frames to previous predictions. We leverage attention rollout and token tracking to propagate token importance of previous outputs to current input tokens. Our method is interpretable, requires no training and has negligible memory overhead. We show the efficacy of our method for both video object detection and action recognition using different transformer architectures, achieving up to 65\% reduction in FLOPS on ImageNet VID and 60\% on EPIC-Kitchens with no accuracy degradation.
</p>
<img width="1266" alt="image" src="https://github.com/user-attachments/assets/9cecf0b4-9370-40e3-bc51-e66352a2718b" />

## Table of Contents
- [Installation](#installation)
- [Datsets](#datasets)
- [Usage](#usage)
- [Results](#results)



## Installation

Detail the steps required to set up the environment and install necessary dependencies. For example:

```bash
git clone https://github.com/your_username/your_project_name.git
cd your_project_name
pip install -r requirements.txt
```

## Datasets

We utilize the following datasets for video object detection and action recognition tasks:

1. **Kinetics-400**
2. **EPIC-KITCHENS**
3. **ImageNet VID**

### 1. Kinetics-400

The Kinetics-400 dataset comprises approximately 240,000 video clips across 400 human action classes, with each clip lasting around 10 seconds. ([Kinetics Dataset GitHub](https://github.com/cvdfoundation/kinetics-dataset))

Download Instructions:
  ```bash
  git clone https://github.com/cvdfoundation/kinetics-dataset.git
  cd kinetics-dataset
  python download.py
  ```

### 2. EPIC-KITCHENS

EPIC-KITCHENS is the largest dataset in first-person (egocentric) vision, capturing daily activities in kitchen environments. It provides a rich collection of video recordings annotated for tasks such as action recognition, object detection, and hand-object interaction. ([EPIC-KITCHENS Official Website](https://epic-kitchens.github.io))

Download Instructions:
  ```bash
  git clone https://github.com/epic-kitchens/epic-kitchens-download-scripts.git
  cd epic-kitchens-download-scripts
  python download.py
  ```

### 3. ImageNet VID

The ImageNet VID dataset is designed for video object detection tasks and contains over 1 million annotated video frames for training and over 100,000 frames for validation. The dataset is part of the larger ImageNet challenge and provides high-quality video sequences labeled with object bounding boxes. ([ImageNet Official Website](https://image-net.org/download))

Download Instructions:
1. Sign up for an ImageNet account at [ImageNet Download Page](http://image-net.org/download-images).
2. Navigate to the **Object Detection from Video (VID)** section and follow the instructions for downloading.

## Usage
### Training
python train.py --data_path /path/to/data --epochs 50

### Evaluation
python evaluate.py --model_path /path/to/model --data_path /path/to/data

## Results
**Qualitative results on ImageNet VID:**
![clip_205p2st4_keep400_start21_comp2](https://github.com/user-attachments/assets/22b5c353-be14-488e-8b93-db739f9ccce5)

**Qualitative results on Kinetics-400:**
![ar_qualitative_clip_270_c](https://github.com/user-attachments/assets/aed148f5-c1fc-419b-9e82-c8421a6e12f1)

**Video object detection on ILSVRC 2015 ImageNet VID:**
![IV_flops_memory](https://github.com/user-attachments/assets/44228457-3903-4168-90cb-05331c76e1d0)

**Video action recognition on EPIC-Kitchens:**
![EK_flops_memory](https://github.com/user-attachments/assets/9dbd65cf-8802-476b-b1fd-7b67d09ef8ea)
