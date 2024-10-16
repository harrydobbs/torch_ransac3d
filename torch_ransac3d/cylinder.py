import torch
import numpy as np

from typing import Tuple
from .wrapper import numpy_to_torch
from .util import rodrigues_rot_torch, estimate_normals
from scipy.optimize import least_squares


@numpy_to_torch
# @torch.compile
@torch.no_grad()
def cylinder_fit(
    pts: torch.Tensor,
    normals: torch.Tensor = None,
    thresh: float = 0.2,
    max_iterations: int = 10000,
    iterations_per_batch: int = 100,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Find the parameters (axis, center, and radius) defining a cylinder using an improved RANSAC approach.

    This function fits a cylinder to a 3D point cloud using a RANSAC-like method,
    processing multiple iterations in parallel for efficiency. It incorporates normal information
    and uses non-linear optimization for refinement.

    :param pts: 3D point cloud.
    :type pts: torch.Tensor
    :param normals: Normal vectors for each point (optional).
    :type normals: torch.Tensor
    :param thresh: Threshold distance from the cylinder surface which is considered inlier.
    :type thresh: float
    :param max_iterations: Maximum number of iterations for the RANSAC algorithm.
    :type max_iterations: int
    :param iterations_per_batch: Number of iterations to process in parallel.
    :type iterations_per_batch: int
    :param device: Device to run the computations on.
    :type device: torch.device

    :return: A tuple containing:
        - center (torch.Tensor): Center point on the cylinder axis (shape: (3,))
        - axis (torch.Tensor): Vector describing cylinder's axis (shape: (3,))
        - radius (torch.Tensor): Radius of the cylinder (shape: (1,))
        - inliers (torch.Tensor): Indices of points from the dataset considered as inliers
    :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

    Example:
        >>> pts = torch.randn(1000, 3)
        >>> normals = torch.randn(1000, 3)
        >>> normals = normals / torch.norm(normals, dim=1, keepdim=True)
        >>> center, axis, radius, inliers = cylinder_fit(pts, normals)
        >>> print(f"Cylinder center: {center}")
        >>> print(f"Cylinder axis: {axis}")
        >>> print(f"Cylinder radius: {radius}")
        >>> print(f"Number of inliers: {inliers.shape[0]}")
    """
    pts = pts.to(device)
    num_pts = pts.shape[0]

    if normals is None:
        # Estimate normals if not provided
        normals = estimate_normals(pts)
    else:
        normals = normals.to(device)

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
        normal_samples = normals[rand_pt_idx]  # (batch_size, 3, 3)

        # Compute initial axis estimate using cross product of normals
        axis_est = torch.cross(
            normal_samples[:, 1] - normal_samples[:, 0],
            normal_samples[:, 2] - normal_samples[:, 0],
        )
        axis_est = axis_est / torch.norm(axis_est, dim=1, keepdim=True)

        # Rotate points to align estimated axis with z-axis
        P_rot = rodrigues_rot_torch(
            pt_samples, axis_est, torch.tensor([0, 0, 1], device=device)
        )

        # Estimate center and radius
        center_est, radius_est = estimate_circle_2d(P_rot[:, :, :2])

        # Rotate center back to original orientation
        center_est = rodrigues_rot_torch(
            center_est.unsqueeze(1), torch.tensor([0, 0, 1], device=device), axis_est
        ).squeeze(1)

        # Compute distances from points to cylinder surface
        dist_pt = point_to_cylinder_distance(pts, center_est, axis_est, radius_est)

        # Select inliers
        inlier_mask = dist_pt <= thresh
        inlier_counts = inlier_mask.sum(dim=1)

        # Find the best iteration in this batch
        best_in_batch_idx = torch.argmax(inlier_counts)
        best_inlier_count_in_batch = inlier_counts[best_in_batch_idx].item()

        if best_inlier_count_in_batch > best_inliers.shape[0]:
            best_inliers = torch.where(inlier_mask[best_in_batch_idx])[0]
            best_center = center_est[best_in_batch_idx]
            best_axis = axis_est[best_in_batch_idx]
            best_radius = radius_est[best_in_batch_idx]

    # Refine the result using non-linear optimization
    best_params = refine_cylinder_fit(
        pts[best_inliers], normals[best_inliers], best_center, best_axis, best_radius
    )

    best_center = torch.tensor(best_params[:3], device=device)
    best_axis = torch.tensor(best_params[3:6], device=device)
    best_radius = torch.tensor(best_params[6], device=device)

    return best_center, best_axis, best_radius, best_inliers


def estimate_circle_2d(points_2d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Estimate circle parameters from 2D points."""
    mean = torch.mean(points_2d, dim=1)
    centered = points_2d - mean.unsqueeze(1)
    _, _, v = torch.svd(centered)
    radius = torch.mean(torch.norm(centered, dim=2), dim=1)
    return torch.cat([mean, torch.zeros_like(mean[:, :1])], dim=1), radius


def point_to_cylinder_distance(
    pts: torch.Tensor, center: torch.Tensor, axis: torch.Tensor, radius: torch.Tensor
) -> torch.Tensor:
    """Compute the distance from points to cylinder surface."""
    v = pts - center.unsqueeze(1)
    proj = torch.sum(v * axis.unsqueeze(1), dim=2, keepdim=True) * axis.unsqueeze(1)
    dist = torch.norm(v - proj, dim=2) - radius.unsqueeze(1)
    return torch.abs(dist)


def cylinder_error(params, points, normals):
    """Error function for cylinder fitting optimization."""
    center = params[:3]
    axis = params[3:6]
    radius = params[6]

    v = points - center
    proj = np.dot(v, axis)[:, np.newaxis] * axis
    dist = np.linalg.norm(v - proj, axis=1) - radius
    normal_alignment = np.abs(np.dot(normals, axis))

    return np.concatenate([dist, normal_alignment])


def refine_cylinder_fit(points, normals, initial_center, initial_axis, initial_radius):
    """Refine cylinder parameters using non-linear optimization."""
    initial_params = np.concatenate(
        [
            initial_center.cpu().numpy(),
            initial_axis.cpu().numpy(),
            [initial_radius.cpu().numpy()],
        ]
    )

    result = least_squares(
        cylinder_error,
        initial_params,
        args=(points.cpu().numpy(), normals.cpu().numpy()),
    )

    return result.x
