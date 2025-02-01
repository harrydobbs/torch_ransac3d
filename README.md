<!-- PROJECT LOGO -->
<br />
<div align="center">
<a href="https://github.com/harrydobbs/torch_ransac3d">
<img src="images/logo.png">
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

Requirements: torch, numpy

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
  - Circles
  - Cylinders
  - Cuboids
  - Points
- Batch processing capability for improved efficiency
- Support for both PyTorch tensors and NumPy arrays as input
- Clean dataclass return types for all fitting functions

## Example Usage

### Line Fitting

```python
import torch
import numpy as np
from torch_ransac3d.line import line_fit

# Using PyTorch tensor
points_torch = torch.rand(1000, 3)
result = line_fit(
    pts=points_torch,
    thresh=0.01,
    max_iterations=1000,
    iterations_per_batch=100,
    epsilon=1e-8,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)
print(f"Direction: {result.direction}")
print(f"Point: {result.point}")
print(f"Number of inliers: {len(result.inliers)}")

# Using NumPy array
points_numpy = np.random.rand(1000, 3)
result = line_fit(
    pts=points_numpy,
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

# Works with both PyTorch tensors and NumPy arrays
points = torch.rand(1000, 3)  # or np.random.rand(1000, 3)
result = plane_fit(
    pts=points,
    thresh=0.05,
    max_iterations=1000,
    iterations_per_batch=100,
    epsilon=1e-8,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)
print(f"Plane equation: {result.equation}")  # [a, b, c, d] for ax + by + cz + d = 0
print(f"Number of inliers: {len(result.inliers)}")
```

### Sphere Fitting

```python
from torch_ransac3d.sphere import sphere_fit

# Works with both PyTorch tensors and NumPy arrays
points = torch.rand(1000, 3)  # or np.random.rand(1000, 3)
result = sphere_fit(
    pts=points,
    thresh=0.05,
    max_iterations=1000,
    iterations_per_batch=100,
    epsilon=1e-8,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)
print(f"Center: {result.center}")
print(f"Radius: {result.radius}")
print(f"Number of inliers: {len(result.inliers)}")
```

## Parameters

- `pts`: Input point cloud (torch.Tensor or numpy.ndarray of shape (N, 3))
- `thresh`: Distance threshold for considering a point as an inlier
- `max_iterations`: Maximum number of RANSAC iterations
- `iterations_per_batch`: Number of iterations to process in parallel
- `epsilon`: Small value to avoid division by zero
- `device`: Torch device to run computations on (CPU or CUDA)

## Input Flexibility

All fitting functions support both PyTorch tensors and NumPy arrays as input. The library automatically converts NumPy arrays to PyTorch tensors internally, allowing for seamless integration with various data formats.

## Batch Processing

All fitting functions support batch processing to improve performance. The `iterations_per_batch` parameter determines how many RANSAC iterations are processed in parallel, leading to significant speedups on GPU hardware.

## Credit

This project is based on the work done at https://github.com/leomariga/pyRANSAC-3D/

## Citation
```
@software{Dobbs_torch_ransac3d,
  author       = {Dobbs, Harry},
  title        = {torch\_ransac3d: A high-performance implementation of 3D RANSAC algorithm using PyTorch and CUDA},
  year         = {2024},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  url          = {https://github.com/harrydobbs/torch_ransac3d},
}
```



## Contact

**Maintainer:** Harry Dobbs
**Email:** harrydobbs87@gmail.com
