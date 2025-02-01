import torch
from typing import Optional


def rodrigues_rot_torch(
    P: torch.Tensor, n0: torch.Tensor, n1: torch.Tensor
) -> torch.Tensor:
    """
    Rotate a set of points between two normal vectors using Rodrigues' formula.

    :param P: Set of points (shape: (..., N, 3))
    :type P: torch.Tensor
    :param n0: Origin vector (shape: (..., 3))
    :type n0: torch.Tensor
    :param n1: Destiny vector (shape: (..., 3))
    :type n1: torch.Tensor
    :returns: Set of rotated points P (shape: (..., N, 3))
    :rtype: torch.Tensor
    """
    # Ensure all inputs are floating point
    P = P.to(torch.float32)
    n0 = n0.to(torch.float32)
    n1 = n1.to(torch.float32)

    # Reshape inputs if necessary
    if P.dim() == 2:  # If P is [N, 3]
        P = P.unsqueeze(0)  # Make it [1, N, 3]

    # Reshape vectors to match batch dimension
    if n0.dim() == 1:
        n0 = n0.unsqueeze(0)
    if n1.dim() == 1:
        n1 = n1.unsqueeze(0)

    # Handle batch dimension
    if n0.dim() == 2 and P.dim() == 3:
        if n0.size(0) == 1:
            n0 = n0.expand(P.size(0), -1)
        if n1.size(0) == 1:
            n1 = n1.expand(P.size(0), -1)

    # Normalize vectors
    n0 = n0 / (torch.linalg.norm(n0, dim=-1, keepdim=True) + 1e-8)
    n1 = n1 / (torch.linalg.norm(n1, dim=-1, keepdim=True) + 1e-8)

    # Get vector of rotation k and angle theta
    k = torch.linalg.cross(n0, n1, dim=-1)
    k_norm = torch.linalg.norm(k, dim=-1, keepdim=True)

    # Handle cases where k is zero (parallel vectors)
    mask = k_norm > 1e-8
    k_normalized = torch.where(mask, k / (k_norm + 1e-8), k)

    theta = torch.acos(torch.clamp(torch.sum(n0 * n1, dim=-1), -1, 1))

    # Prepare for broadcasting
    cos_theta = theta.cos().view(-1, 1, 1)
    sin_theta = theta.sin().view(-1, 1, 1)

    # Ensure k_normalized has the same shape as P
    k_normalized = k_normalized.unsqueeze(1)  # [batch, 1, 3]
    k_normalized = k_normalized.expand(-1, P.size(1), -1)  # [batch, N, 3]

    # Compute rotated points
    k_dot_p = torch.sum(k_normalized * P, dim=-1, keepdim=True)
    P_rot = (
        P * cos_theta
        + torch.linalg.cross(k_normalized, P, dim=-1) * sin_theta
        + k_normalized * k_dot_p * (1 - cos_theta)
    )

    # Use original points where k is zero
    mask = mask.unsqueeze(-1).unsqueeze(-1)
    P_rot = torch.where(mask, P_rot, P)

    return P_rot.squeeze(0)  # Remove the extra dimension we added


def knn_pytorch(x, k):
    """
    K-nearest neighbors search using PyTorch operations.

    Args:
    x (torch.Tensor): Input tensor of shape (N, D) where N is the number of points and D is the dimension.
    k (int): Number of nearest neighbors to find.

    Returns:
    torch.Tensor: Indices of k-nearest neighbors for each point, shape (N, k).
    """
    inner = -2 * torch.matmul(x, x.t())
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.t()

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (N, k)
    return idx


def estimate_normals(
    pts: torch.Tensor, 
    k: Optional[int] = None
) -> torch.Tensor:
    """Estimate normals using PCA on local neighborhoods.
    
    Args:
        pts: Input point cloud
        k: Number of neighbors, defaults to min(20, pts.shape[0]-1)
    """
    if k is None:
        k = min(20, pts.shape[0] - 1)
    # Find k-nearest neighbors
    knn_indices = knn_pytorch(pts, k)

    # Get the neighbors for each point
    neighbors = pts[knn_indices]  # shape: (num_points, k, 3)

    # Center the neighborhood
    centered = neighbors - pts.unsqueeze(1)  # shape: (num_points, k, 3)

    # Compute covariance matrix for each neighborhood
    cov = torch.bmm(centered.transpose(1, 2), centered)  # shape: (num_points, 3, 3)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)

    # The eigenvector corresponding to the smallest eigenvalue is the normal
    normals = eigenvectors[:, :, 0]

    # Ensure normals point towards the viewer (assuming viewer is at origin)
    flip_mask = torch.sum(pts * normals, dim=1) > 0
    normals[flip_mask] = -normals[flip_mask]

    return normals
