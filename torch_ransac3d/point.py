from typing import Tuple

import torch

from .dataclasses import PointFitResult
from .wrapper import numpy_to_torch


@numpy_to_torch
@torch.compile
@torch.no_grad()
def point_fit(
    pts: torch.Tensor,
    thresh: float = 0.05,
    max_iterations: int = 1000,
    iterations_per_batch: int = 1,
    device: torch.device = torch.device("cpu"),
) -> PointFitResult:
    """
    Find the best center point using a batched RANSAC approach.

    This function finds the point with the most neighbors within a specified radius.

    :param pts: 3D point cloud.
    :type pts: torch.Tensor
    :param thresh: Threshold radius from the point which is considered inlier.
    :type thresh: float
    :param max_iterations: Maximum number of iterations for the RANSAC algorithm.
    :type max_iterations: int
    :param iterations_per_batch: Number of iterations to process in parallel.
    :type iterations_per_batch: int
    :param device: Device to run the computations on.
    :type device: torch.device

    :return: A PointFitResult containing the center point and inlier indices
    :rtype: PointFitResult

    Example:
        >>> pts = torch.randn(1000, 3)
        >>> center, inliers = point_fit(pts)
        >>> print(f"Best point: {center}")
        >>> print(f"Number of inliers: {inliers.shape[0]}")
    """
    pts = pts.to(device).to(torch.float32)
    num_pts = pts.shape[0]

    best_inliers = torch.tensor([], dtype=torch.long, device=device)
    best_center = torch.zeros(3, device=device)

    for start_idx in range(0, max_iterations, iterations_per_batch):
        end_idx = min(start_idx + iterations_per_batch, max_iterations)
        current_batch_size = end_idx - start_idx

        # Sample random points for each iteration in the batch
        rand_pt_idx = torch.randint(0, num_pts, (current_batch_size,), device=device)
        pt_samples = pts[rand_pt_idx]  # (batch_size, 3)

        # Compute distances from sampled points to all other points
        dist_pt = torch.cdist(pt_samples.unsqueeze(1), pts.unsqueeze(0)).squeeze(
            1
        )  # (batch_size, num_pts)

        # Select inliers
        inlier_mask = dist_pt <= thresh
        inlier_counts = inlier_mask.sum(dim=1)

        # Find the best iteration in this batch
        best_in_batch_idx = torch.argmax(inlier_counts)
        best_inlier_count_in_batch = inlier_counts[best_in_batch_idx].item()

        if best_inlier_count_in_batch > best_inliers.shape[0]:
            best_inliers = torch.where(inlier_mask[best_in_batch_idx])[0]
            best_center = pt_samples[best_in_batch_idx]

    return PointFitResult(center=best_center, inliers=best_inliers)
