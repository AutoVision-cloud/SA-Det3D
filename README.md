# SA-Det3D: Self-Attention Based Context-Aware 3D Object Detection

By [Prarthana Bhattacharyya](https://scholar.google.com/citations?user=v6pGkNQAAAAJ&hl=en), [Chengjie Huang](https://scholar.google.com/citations?user=O6gvGZgAAAAJ&hl=en) and [Krzysztof Czarnecki](https://scholar.google.com/citations?hl=en&user=ZzCpumQAAAAJ).

We provide code support and configuration files to reproduce the results in the paper:
Self-Attention Based Context-Aware 3D Object Detection. 
<br/> Our code is based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet), which is a clean open-sourced project for benchmarking 3D object detection methods. 


## Overview

<div align="center">
  <img src="docs/overview.png" width="600px" height="280px" />
  <p>Fig.1. Self-Attention augmented global-context aware backbone networks. </p>
</div>

In this paper, we explore variations of 
self-attention for contextual modeling in 3D object 
detection by augmenting convolutional features with 
self-attention features. 
We first incorporate the pairwise self-attention 
mechanism into the current state-of-the-art BEV, voxel 
and point-based detectors and show consistent 
improvement over strong baseline models 
while simultaneously significantly reducing 
their parameter footprint and computational cost. 
We also propose a self-attention variant that 
samples a subset of the most representative features by 
learning deformations over randomly sampled locations. 
This not only allows us to scale explicit global contextual 
modeling to larger point-clouds, 
but also leads to more discriminative and informative feature 
descriptors.


## Results
For similar number of parameters and FLOPs, self-attention (SA) systematically 
improves 3D object detection across  state-of-the-art  3D  detectors (PointPillars, SECOND and 
Point-RCNN). AP on moderate Car class of KITTI val split (R40) vs. the 
number of parameters (Top) and 
GFLOPs (Bottom) for baseline models and proposed baseline extensions with 
Deformable and Full SA.
<div align="center">
  <img src="docs/demo_params_flops.png" width="300px" />
  <p>Fig.2. 3D Car AP with respect to params and FLOPs of baseline and proposed 
self-attention variants. </p>
</div>
<br/>

Performance illustrations on KITTI val split. Red bounding box 
represents ground truth; green represents detector outputs. 
From left to right: (a) RGB image of challenging scenes. 
(b) Result of the state-of-the-art methods: PointPillars, 
SECOND, Point-RCNN, PV-RCNN. (c) Result of our full self-attention (FSA) 
augmented baselines, which uses significantly fewer 
parameters and FLOPs. 
FSA attends to the entire point-cloud to produce global 
context-aware feature representations. 
Our method identifies missed detections and removes false positives.
<div align="center">
  <img src="docs/demo_qual.png" width="600px" />
  <p>Fig.3. Visualizing qualitative results between baseline and
our proposed self-attention module.</p>
</div>

## Usage
a. Clone the repo:
```
git clone --recursive https://github.com/AutoVision-cloud/SA-Det3D
```
b. Copy SA-Det3D src into OpenPCDet: 
```
sh ./init.sh
```

c. Install OpenPCDet and prepare KITTI data:

Please refer to [INSTALL.md](docs/INSTALL.md) for installation and dataset preparation.

d. Run experiments with a specific configuration file:

Please refer to [GETTING_STARTED.md](docs/GETTING_STARTED.md) to learn more about how to train and run inference on this detector.

## Acknowledgement
* [second.pytorch](https://github.com/traveller59/second.pytorch)
* [PointRCNN](https://github.com/sshaoshuai/PointRCNN)
* [PV-RCNN](https://github.com/open-mmlab/OpenPCDet)
* [MLCVNet](https://github.com/NUAAXQ/MLCVNet)