from dataclasses import dataclass

import torch


@dataclass
class SphereFitResult:
    """Results from fitting a sphere to point cloud data."""

    center: torch.Tensor
    radius: float
    inliers: torch.Tensor


@dataclass
class PlaneFitResult:
    """Results from fitting a plane to point cloud data."""

    equation: torch.Tensor  # [a, b, c, d] for plane ax + by + cz + d = 0
    inliers: torch.Tensor


@dataclass
class CylinderFitResult:
    """Results from fitting a cylinder to point cloud data."""

    center: torch.Tensor  # Point on axis
    axis: torch.Tensor  # Direction vector
    radius: float
    inliers: torch.Tensor


@dataclass
class CircleFitResult:
    """Results from fitting a circle to point cloud data."""

    center: torch.Tensor
    normal: torch.Tensor  # Normal to circle plane
    radius: float
    inliers: torch.Tensor


@dataclass
class CuboidFitResult:
    """Results from fitting a cuboid to point cloud data."""

    equations: torch.Tensor  # [3, 4] array of plane equations
    inliers: torch.Tensor


@dataclass
class PointFitResult:
    """Results from fitting a point cluster."""

    center: torch.Tensor
    inliers: torch.Tensor


@dataclass
class LineFitResult:
    """Results from fitting a line to point cloud data."""

    direction: torch.Tensor
    point: torch.Tensor
    inliers: torch.Tensor
