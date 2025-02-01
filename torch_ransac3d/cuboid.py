from typing import Tuple

import torch

from .wrapper import numpy_to_torch


@numpy_to_torch
@torch.no_grad()
def cuboid_fit(
    pts: torch.Tensor,
    thresh: float = 0.05,
    max_iterations: int = 1000,
    iterations_per_batch: int = 1,
    epsilon: float = 1e-8,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find the best equations for 3 planes which define a complete cuboid using a batched RANSAC approach.

    This function fits a cuboid to a 3D point cloud using a RANSAC-like method,
    processing multiple iterations in parallel for efficiency.

    :param pts: 3D point cloud.
    :type pts: torch.Tensor
    :param thresh: Threshold distance from the plane which is considered inlier.
    :type thresh: float
    :param max_iterations: Maximum number of iterations for the RANSAC algorithm.
    :type max_iterations: int
    :param iterations_per_batch: Number of iterations to process in parallel.
    :type iterations_per_batch: int
    :param epsilon: Small value to avoid division by zero.
    :type epsilon: float
    :param device: Device to run the computations on.
    :type device: torch.device

    :return: A tuple containing:
        - best_eq (torch.Tensor): Array of 3 best planes' equations (shape: (3, 4))
        - best_inliers (torch.Tensor): Indices of points from the dataset considered as inliers
    :rtype: Tuple[torch.Tensor, torch.Tensor]

    Example:
        >>> pts = torch.randn(1000, 3)
        >>> equations, inliers = cuboid_fit(pts)
        >>> print(f"Cuboid equations:\n{equations}")
        >>> print(f"Number of inliers: {inliers.shape[0]}")
    """
    pts = pts.to(device)
    num_pts = pts.shape[0]

    best_inliers = torch.tensor([], dtype=torch.long, device=device)
    best_eq = torch.zeros((3, 4), device=device)

    for start_idx in range(0, max_iterations, iterations_per_batch):
        end_idx = min(start_idx + iterations_per_batch, max_iterations)
        current_batch_size = end_idx - start_idx

        # Sample 6 random points for each iteration in the batch
        rand_pt_idx = torch.randint(0, num_pts, (current_batch_size, 6), device=device)
        pt_samples = pts[rand_pt_idx]  # (batch_size, 6, 3)

        # Compute vectors for the first plane
        vec_A = pt_samples[:, 1] - pt_samples[:, 0]
        vec_B = pt_samples[:, 2] - pt_samples[:, 0]
        vec_C = torch.linalg.cross(vec_A, vec_B, dim=-1)
        vec_C = vec_C / (torch.norm(vec_C, dim=1, keepdim=True) + epsilon)

        # Compute k for the first plane
        k = -torch.sum(vec_C * pt_samples[:, 1], dim=1)
        plane_eq = torch.cat([vec_C, k.unsqueeze(1)], dim=1)  # (batch_size, 4)

        # Compute the second plane
        dist_p4_plane = (
            torch.sum(plane_eq[:, :3] * pt_samples[:, 3], dim=1) + plane_eq[:, 3]
        )
        dist_p4_plane = dist_p4_plane / (torch.norm(plane_eq[:, :3], dim=1) + epsilon)
        p4_proj_plane = pt_samples[:, 3] - dist_p4_plane.unsqueeze(1) * vec_C

        vec_D = p4_proj_plane - pt_samples[:, 3]
        vec_E = pt_samples[:, 4] - pt_samples[:, 3]
        vec_F = torch.linalg.cross(vec_D, vec_E, dim=-1)
        vec_F = vec_F / (torch.norm(vec_F, dim=1, keepdim=True) + epsilon)

        k = -torch.sum(vec_F * pt_samples[:, 4], dim=1)
        plane_eq = torch.cat(
            [plane_eq, torch.cat([vec_F, k.unsqueeze(1)], dim=1)], dim=1
        )  # (batch_size, 8)

        # Compute the third plane
        vec_G = torch.cross(vec_C, vec_F, dim=-1)
        k = -torch.sum(vec_G * pt_samples[:, 5], dim=1)
        plane_eq = torch.cat(
            [plane_eq, torch.cat([vec_G, k.unsqueeze(1)], dim=1)], dim=1
        )  # (batch_size, 12)

        # Reshape plane_eq to (batch_size, 3, 4)
        plane_eq = plane_eq.view(current_batch_size, 3, 4)

        # Compute distances of all points to each plane in the batch
        dist_pt = torch.abs(
            torch.einsum("bij,kj->bik", plane_eq[:, :, :3], pts)
            + plane_eq[:, :, 3].unsqueeze(2)
        )
        dist_pt = dist_pt / (
            torch.norm(plane_eq[:, :, :3], dim=2, keepdim=True) + epsilon
        )

        # Select inliers
        min_dist_pt = torch.min(dist_pt, dim=1)[0]
        inlier_mask = min_dist_pt <= thresh
        inlier_counts = inlier_mask.sum(dim=1)

        # Find the best iteration in this batch
        best_in_batch_idx = torch.argmax(inlier_counts)
        best_inlier_count_in_batch = inlier_counts[best_in_batch_idx].item()

        if best_inlier_count_in_batch > best_inliers.shape[0]:
            best_inliers = torch.where(inlier_mask[best_in_batch_idx])[0]
            best_eq = plane_eq[best_in_batch_idx]

    return best_eq, best_inliers
