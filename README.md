# SA-Det3D: Self-Attention Based Context-Aware 3D Object Detection

By [Prarthana Bhattacharyya](https://scholar.google.com/citations?user=v6pGkNQAAAAJ&hl=en), [Chengjie Huang](https://scholar.google.com/citations?user=O6gvGZgAAAAJ&hl=en) and [Krzysztof Czarnecki](https://scholar.google.com/citations?hl=en&user=ZzCpumQAAAAJ).

We provide code support and configuration files to reproduce the results in the paper:
Self-Attention Based Context-Aware 3D Object Detection. 
<br/> Our code is based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet), which is a clean open-sourced project for benchmarking 3D object detection methods. 


## Overview

<div align="center">
  <img src="docs/overview.png" width="450px" height="150px" />
  <p>Self-Attention augmented global-context aware backbone networks. </p>
</div>

Most existing point-cloud based 3D object detectors use 
convolution-like operators to process information in a 
local neighbourhood with fixed-weight kernels and aggregate 
global context hierarchically. 
However, recent work on non-local neural networks and 
self-attention for 2D vision has shown that explicitly 
modeling global context and long-range interactions 
between positions can lead to more robust and competitive 
models. 
<br/> <br/> In this paper, we explore variations of 
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
<br/> Our method can be flexibly applied to most state-of-the-art detectors with increased accuracy and parameter and compute efficiency. We achieve new state-of-the-art detection performance on KITTI and nuScenes datasets.

