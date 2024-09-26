
<!-- PROJECT LOGO -->
<br />
<div align="center">
<a href="https://github.com/harrydobbs/torch_ransac3d">
<img src="images/logo.png" alt="Logo">
</a>
<br><br>
<!-- <h3 align="center">torch_ransac3d</h3> -->
<p align="center">
 A high-performance implementation of 3D RANSAC algorithm using PyTorch and CUDA.
<br />
<a href="https://harrydobbs.github.io/torch_ransac3d/"><strong>Explore the docs »</strong></a> <br />
<br />
<br />
<a href="https://github.com/harrydobbs/torch_ransac3d/">View Demo</a>
 ·
<a href="https://github.com/harrydobbs/torch_ransac3d//issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
 ·
<a href="https://github.com/harrydobbs/torch_ransac3d//issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
</p>
</div>

## Installation

Requirements: torch

Install with <a href="https://pypi.org/project/torch-ransac3d"> PyPI </a>:
```
pip install torch-ransac3d
```

## Features

- High-performance RANSAC implementation using PyTorch and CUDA
- Supports fitting of multiple geometric primitives:
  - Lines
  - Planes
  - Spheres
- Batch processing capability for improved efficiency

## Example Usage

### Line Fitting

```python
import torch
from torch_ransac3d.line import line_fit

points = torch.rand(1000, 3)
direction, point, inliers = line_fit(
    pts=points,
    thresh=0.01,
    max_iterations=1000,
    iterations_per_batch=100,
    epsilon=1e-8,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)
```

### Plane Fitting

```python
from torch_ransac3d.plane import plane_fit

points = torch.rand(1000, 3)
equation, inliers = plane_fit(
    pts=points,
    thresh=0.05,
    max_iterations=1000,
    iterations_per_batch=100,
    epsilon=1e-8,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)
```

### Sphere Fitting

```python
from torch_ransac3d.sphere import sphere_fit

points = torch.rand(1000, 3)
center, radius, inliers = sphere_fit(
    pts=points,
    thresh=0.05,
    max_iterations=1000,
    iterations_per_batch=100,
    epsilon=1e-8,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)
```

## Parameters

- `pts`: Input point cloud (torch.Tensor of shape (N, 3))
- `thresh`: Distance threshold for considering a point as an inlier
- `max_iterations`: Maximum number of RANSAC iterations
- `iterations_per_batch`: Number of iterations to process in parallel
- `epsilon`: Small value to avoid division by zero
- `device`: Torch device to run computations on (CPU or CUDA)

## Batch Processing

All fitting functions support batch processing to improve performance. The `iterations_per_batch` parameter determines how many RANSAC iterations are processed in parallel, leading to significant speedups on GPU hardware.

## Credit

This project is based on the work done at https://github.com/leomariga/pyRANSAC-3D/

## Contact

**Maintainer:** Harry Dobbs  
**Email:** harrydobbs87@gmail.com