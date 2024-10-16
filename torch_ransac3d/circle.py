import torch
from typing import Tuple
from .wrapper import numpy_to_torch
from .util import rodrigues_rot_torch


@numpy_to_torch
@torch.compile
@torch.no_grad()
def circle_fit(
    pts: torch.Tensor,
    thresh: float = 0.2,
    max_iterations: int = 1000,
    iterations_per_batch: int = 1,
    epsilon: float = 1e-8,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Find the parameters (center, axis, and radius) to define a circle using a batched RANSAC approach.

    This function fits a circle to a 3D point cloud using a RANSAC-like method,
    processing multiple iterations in parallel for efficiency.

    :param pts: 3D point cloud.
    :type pts: torch.Tensor
    :param thresh: Threshold distance from the circle which is considered inlier.
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
        - center (torch.Tensor): Center of the circle (shape: (3,))
        - axis (torch.Tensor): Vector describing circle's plane normal (shape: (3,))
        - radius (torch.Tensor): Radius of the circle (shape: (1,))
        - inliers (torch.Tensor): Indices of points from the dataset considered as inliers
    :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

    Example:
        >>> pts = torch.randn(1000, 3)
        >>> center, axis, radius, inliers = circle_fit(pts)
        >>> print(f"Circle center: {center}")
        >>> print(f"Circle axis: {axis}")
        >>> print(f"Circle radius: {radius}")
        >>> print(f"Number of inliers: {inliers.shape[0]}")
    """
    pts = pts.to(device)
    num_pts = pts.shape[0]

    best_inliers = torch.tensor([], dtype=torch.long, device=device)
    best_center = torch.zeros(3, device=device)
    best_axis = torch.zeros(3, device=device)
    best_radius = torch.tensor(0.0, device=device)

    for start_idx in range(0, max_iterations, iterations_per_batch):
        end_idx = min(start_idx + iterations_per_batch, max_iterations)
        current_batch_size = end_idx - start_idx

        # Sample 3 random points for each iteration in the batch
        rand_pt_idx = torch.randint(0, num_pts, (current_batch_size, 3), device=device)
        pt_samples = pts[rand_pt_idx]  # (batch_size, 3, 3)

        # Compute vectors for the plane
        vec_A = pt_samples[:, 1] - pt_samples[:, 0]
        vec_A_norm = vec_A / (torch.norm(vec_A, dim=1, keepdim=True) + epsilon)
        vec_B = pt_samples[:, 2] - pt_samples[:, 0]
        vec_B_norm = vec_B / (torch.norm(vec_B, dim=1, keepdim=True) + epsilon)

        # Compute normal vector to the plane
        vec_C = torch.cross(vec_A_norm, vec_B_norm)
        vec_C = vec_C / (torch.norm(vec_C, dim=1, keepdim=True) + epsilon)

        # Compute plane equation
        k = -torch.sum(vec_C * pt_samples[:, 1], dim=1)
        plane_eq = torch.cat([vec_C, k.unsqueeze(1)], dim=1)  # (batch_size, 4)

        # Rotate points to align with z-axis
        P_rot = rodrigues_rot_torch(
            pt_samples, vec_C, torch.tensor([0, 0, 1], device=device)
        )

        # Find center from 3 points
        ma = (P_rot[:, 1, 1] - P_rot[:, 0, 1]) / (
            P_rot[:, 1, 0] - P_rot[:, 0, 0] + epsilon
        )
        mb = (P_rot[:, 2, 1] - P_rot[:, 1, 1]) / (
            P_rot[:, 2, 0] - P_rot[:, 1, 0] + epsilon
        )

        p_center_x = (
            ma * mb * (P_rot[:, 0, 1] - P_rot[:, 2, 1])
            + mb * (P_rot[:, 0, 0] + P_rot[:, 1, 0])
            - ma * (P_rot[:, 1, 0] + P_rot[:, 2, 0])
        ) / (2 * (mb - ma + epsilon))
        p_center_y = (
            -1 / (ma + epsilon) * (p_center_x - (P_rot[:, 0, 0] + P_rot[:, 1, 0]) / 2)
            + (P_rot[:, 0, 1] + P_rot[:, 1, 1]) / 2
        )
        p_center = torch.stack(
            [p_center_x, p_center_y, torch.zeros_like(p_center_x)], dim=1
        )
        radius = torch.norm(p_center - P_rot[:, 0, :], dim=1)

        # Rotate center back to original orientation
        center = rodrigues_rot_torch(
            p_center.unsqueeze(1), torch.tensor([0, 0, 1], device=device), vec_C
        ).squeeze(1)

        # Compute distances from points to circle
        dist_pt_plane = torch.abs(
            torch.sum(pts * plane_eq[:, :3].unsqueeze(1), dim=2)
            + plane_eq[:, 3].unsqueeze(1)
        ) / (torch.norm(plane_eq[:, :3], dim=1, keepdim=True) + epsilon)
        dist_pt_inf_circle = torch.norm(
            torch.cross(
                vec_C.unsqueeze(1).expand(-1, num_pts, -1),
                (center.unsqueeze(1) - pts.unsqueeze(0)),
            ),
            dim=2,
        ) - radius.unsqueeze(1)
        dist_pt = torch.sqrt(dist_pt_inf_circle**2 + dist_pt_plane**2)

        # Select inliers
        inlier_mask = dist_pt <= thresh
        inlier_counts = inlier_mask.sum(dim=1)

        # Find the best iteration in this batch
        best_in_batch_idx = torch.argmax(inlier_counts)
        best_inlier_count_in_batch = inlier_counts[best_in_batch_idx].item()

        if best_inlier_count_in_batch > best_inliers.shape[0]:
            best_inliers = torch.where(inlier_mask[best_in_batch_idx])[0]
            best_center = center[best_in_batch_idx]
            best_axis = vec_C[best_in_batch_idx]
            best_radius = radius[best_in_batch_idx]

    return best_center, best_axis, best_radius, best_inliers
