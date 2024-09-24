import torch


@torch.compile
@torch.no_grad()
def sphere_fit(
    pts,
    thresh=0.05,
    max_iterations=1000,
    iterations_per_batch=1,
    epsilon=1e-8,
    device=torch.device("cpu"),
):
    """
    Find the best parameters (center and radius) for a sphere in batched RANSAC approach.

    :param pts: 3D point cloud as a torch.Tensor (N, 3).
    :param thresh: Threshold distance from the sphere hull which is considered inlier.
    :param max_iterations: Maximum number of iterations for RANSAC.
    :param iterations_per_batch: Number of iterations to run in parallel.
    :param device: Device for running the algorithm (default is cuda).
    :param epsilon: Small value to avoid division by zero.
    :returns:
    - `best_center`: Center of the sphere as a torch.Tensor (3,).
    - `best_radius`: Radius of the sphere.
    - `best_inlier_indices`: Indices of points from the dataset considered inliers.
    """

    pts = pts.to(device)
    num_pts = pts.shape[0]

    best_inlier_indices = torch.tensor([], dtype=torch.long, device=device)
    best_inlier_count = 0
    best_center = None
    best_radius = None

    for start_idx in range(0, max_iterations, iterations_per_batch):
        end_idx = min(start_idx + iterations_per_batch, max_iterations)
        current_batch_size = end_idx - start_idx

        # Sample 4 random points for each iteration in the batch
        rand_pt_idx = torch.randint(0, num_pts, (current_batch_size, 4), device=device)
        sampled_points = pts[rand_pt_idx]  # (batch_size, 4, 3)

        # Step 1: Compute 4x4 determinants for each batch (M11, M12, M13, M14, M15)

        # Prepare the 4x4 matrix d_matrix
        d_matrix = torch.ones((current_batch_size, 4, 4), device=device)

        # Fill in x, y, z coordinates for M11
        d_matrix[:, :, :3] = sampled_points

        # M11 determinant
        M11 = torch.linalg.det(d_matrix) + epsilon

        # Replace the x-coordinate with (x² + y² + z²) for M12
        d_matrix[:, :, 0] = torch.sum(sampled_points**2, dim=2)
        M12 = torch.linalg.det(d_matrix)

        # Replace the y-coordinate for M13
        d_matrix[:, :, 1] = sampled_points[:, :, 0]  # Replace y by x
        M13 = torch.linalg.det(d_matrix)

        # Replace the z-coordinate for M14
        d_matrix[:, :, 2] = sampled_points[:, :, 1]  # Replace z by y
        M14 = torch.linalg.det(d_matrix)

        # Replace the 1 column (the fourth) for M15
        d_matrix[:, :, 3] = sampled_points[:, :, 2]  # Replace 1 by z
        M15 = torch.linalg.det(d_matrix)

        # Step 2: Compute sphere center and radius for each batch
        center = torch.stack(
            [0.5 * (M12 / M11), -0.5 * (M13 / M11), 0.5 * (M14 / M11)], dim=1
        )  # (batch_size, 3)
        radius = torch.sqrt(torch.sum(center**2, dim=1) - (M15 / M11))  # (batch_size,)

        # Step 3: Compute distances of all points to each sphere in the batch
        dist_pts = torch.cdist(center.unsqueeze(1), pts.unsqueeze(0), p=2).squeeze(
            1
        )  # (batch_size, num_pts)

        # Inlier mask: points where abs(distance - radius) <= threshold
        inlier_mask = (
            torch.abs(dist_pts - radius.unsqueeze(1)) <= thresh
        )  # (batch_size, num_pts)
        inlier_counts = inlier_mask.sum(dim=1)  # (batch_size,)

        # Find the best iteration in this batch
        best_in_batch_idx = torch.argmax(inlier_counts)  # Best sphere in this batch
        best_inlier_count_in_batch = inlier_counts[best_in_batch_idx].item()

        if best_inlier_count_in_batch > best_inlier_count:
            best_inlier_count = best_inlier_count_in_batch
            best_inlier_indices = torch.where(inlier_mask[best_in_batch_idx])[0]
            best_center = center[best_in_batch_idx]
            best_radius = radius[best_in_batch_idx]

    return best_center, best_radius, best_inlier_indices
