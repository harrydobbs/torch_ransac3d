<!-- PROJECT LOGO -->
<br />
<div align="center">
  <!-- <a href="https://github.com/github_username/repo_name">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a> -->

<h3 align="center">torch_ransac3d</h3>

  <p align="center">
    A high-performance implementation of 3D RANSAC (Random Sample Consensus) algorithm using PyTorch and CUDA.
    <br />
    <a href="https://harrydobbs.github.io/torch_ransac3d/"><strong>Explore the docs »</strong></a>    <br />
    <br />
    <br />
    <a href="https://github.com/harrydobbs/torch_ransac3d/">View Demo</a>
    ·
    <a href="https://github.com/harrydobbs/torch_ransac3d//issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/harrydobbs/torch_ransac3d//issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>

# Installation
Requirements: torch

Install with Pypi:

```pip install torch_ransac3d```

# Example Usage

```
import torch
from torch_ransac3d.line import line_fit

points = torch.rand(1000,3)
direction, intercept, inliers = line_fit(points)
```

Currently other supported geometries include planes and spheres.


# Credit:
This is based on the work done at https://github.com/leomariga/pyRANSAC-3D/
