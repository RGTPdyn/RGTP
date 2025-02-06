# RGTP
An official code of "Rollout-Guided Token Pruning":
<p align="justify"> 
  Vision Transformers have been proven powerful in various vision applications. Yet, their adaptations for video understanding tasks incur large computational costs, limiting their practical deployment on resource-constrained devices. Token pruning can effectively alleviate the processing overhead of underlying attention blocks, but often neglects the iterative processing nature of video models applied frame-by-frame. We propose to prune tokens according to the estimated contribution of their corresponding tokens in previous frames to previous predictions. We leverage attention rollout and token tracking to propagate token importance of previous outputs to current input tokens. Our method is interpretable, requires no training and has negligible memory overhead. We show the efficacy of our method for both video object detection and action recognition using different transformer architectures, achieving up to 65\% reduction in FLOPS on ImageNet VID and 60\% on EPIC-Kitchens with no accuracy degradation.
</p>
<img width="1266" alt="image" src="https://github.com/user-attachments/assets/9cecf0b4-9370-40e3-bc51-e66352a2718b" />

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)

## Introduction

Provide an overview of your research, including the problem addressed, methodology, and key findings. You may also include a link to your published paper and a representative figure or diagram.

## Installation

Detail the steps required to set up the environment and install necessary dependencies. For example:

```bash
git clone https://github.com/your_username/your_project_name.git
cd your_project_name
pip install -r requirements.txt
```

## Usage
### Training
python train.py --data_path /path/to/data --epochs 50

### Evaluation
python evaluate.py --model_path /path/to/model --data_path /path/to/data

## Results

