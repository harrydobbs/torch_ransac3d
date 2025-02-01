from typing import Tuple

import torch

from .dataclasses import PlaneFitResult
from .wrapper import numpy_to_torch


@numpy_to_torch
@torch.compile
@torch.no_grad()
def plane_fit(
    pts: torch.Tensor,
    thresh: float = 0.05,
    max_iterations: int = 1000,
    iterations_per_batch: int = 1,
    epsilon: float = 1e-8,
    device: torch.device = torch.device("cpu"),
) -> PlaneFitResult:
    """
    Find the best equation for a plane using a batched RANSAC approach.

    This function fits a plane to a 3D point cloud using a RANSAC-like method,
    processing multiple iterations in parallel for efficiency.

    :param pts: 3D point cloud.
    :type pts: torch.Tensor
    :param thresh: Threshold distance from the plane to consider a point as an inlier.
    :type thresh: float
    :param max_iterations: Maximum number of iterations for the RANSAC algorithm.
    :type max_iterations: int
    :param iterations_per_batch: Number of iterations to process in parallel.
    :type iterations_per_batch: int
    :param epsilon: Small value to avoid division by zero.
    :type epsilon: float
    :param device: Device to run the computations on.
    :type device: torch.device

    :return: A PlaneFitResult containing the plane equation and inlier indices
    :rtype: PlaneFitResult

    Example:
        >>> pts = torch.randn(1000, 3)
        >>> result = plane_fit(pts)
        >>> print(f"Plane equation: {result.equation}")
        >>> print(f"Number of inliers: {result.inliers.shape[0]}")
    """

    pts = pts.to(device).to(torch.float32)
    num_pts = pts.shape[0]

    best_inlier_indices = torch.tensor([], dtype=torch.long, device=device)
    best_inlier_count = 0
    best_eq = None

    for start_idx in range(0, max_iterations, iterations_per_batch):
        end_idx = min(start_idx + iterations_per_batch, max_iterations)
        current_batch_size = end_idx - start_idx

        # Sample 3 random points for each iteration in the batch
        rand_pt_idx = torch.randint(0, num_pts, (current_batch_size, 3), device=device)
        sampled_points = pts[rand_pt_idx]  # (batch_size, 3, 3)

        # Compute vectors vecA and vecB
        vec_A = sampled_points[:, 1, :] - sampled_points[:, 0, :]  # (batch_size, 3)
        vec_B = sampled_points[:, 2, :] - sampled_points[:, 0, :]  # (batch_size, 3)

        # Compute cross product vecC, which is the normal to the plane
        vec_C = torch.cross(vec_A, vec_B, dim=1)  # (batch_size, 3)

        # Normalize vecC to get the unit normal vector
        vec_C = vec_C / (
            torch.norm(vec_C, dim=1, keepdim=True) + epsilon
        )  # (batch_size, 3)

        # Compute the constant term k for each plane
        k = -torch.einsum("ij,ij->i", vec_C, sampled_points[:, 1, :])  # (batch_size,)

        # Plane equation coefficients (Ax + By + Cz + D)
        plane_eq = torch.cat([vec_C, k.unsqueeze(1)], dim=1)  # (batch_size, 4)

        # Compute distances of all points to each plane in the batch
        # Using the point-plane distance formula for each point and each plane in the batch
        dist_pts = (
            plane_eq[:, 0:1] * pts[:, 0].unsqueeze(0)
            + plane_eq[:, 1:2] * pts[:, 1].unsqueeze(0)
            + plane_eq[:, 2:3] * pts[:, 2].unsqueeze(0)
            + plane_eq[:, 3:4]
        ) / torch.sqrt(
            plane_eq[:, 0] ** 2 + plane_eq[:, 1] ** 2 + plane_eq[:, 2] ** 2
        ).unsqueeze(
            1
        )

        # Inlier mask: points where distance <= threshold
        inlier_mask = torch.abs(dist_pts) <= thresh  # (batch_size, num_pts)
        inlier_counts = inlier_mask.sum(dim=1)  # (batch_size,)

        # Find the best iteration in this batch
        best_in_batch_idx = torch.argmax(inlier_counts)  # Best plane in this batch
        best_inlier_count_in_batch = inlier_counts[best_in_batch_idx].item()

        if best_inlier_count_in_batch > best_inlier_count:
            best_inlier_count = best_inlier_count_in_batch
            best_inlier_indices = torch.where(inlier_mask[best_in_batch_idx])[0]
            best_eq = plane_eq[best_in_batch_idx]

    return PlaneFitResult(equation=best_eq, inliers=best_inlier_indices)
