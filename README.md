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

<div align="center">
  <img src="docs/demo_qual.png" width="700px" />
  <p>Fig.2. Visualizing qualitative results between baseline and
our proposed self-attention module.</p>
</div>

<div align="center">
  <img src="docs/demo_params_flops.png" width="300px" />
  <p>Fig.3. mAP with respect to params and FLOPs of baseline and proposed 
self-attention variants. </p>
</div>