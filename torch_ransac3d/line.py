import torch


@torch.compile
@torch.no_grad()
def line_fit(
    pts,
    thresh=0.01,
    max_iterations=1000,
    iterations_per_batch=1,
    epsilon=1e-8,
    device=torch.device("cpu"),
):
    """
    Fit a line using a RANSAC-like method in a batched approach.

    :param pts: 3D point cloud as a `torch.Tensor (N, 3)`.
    :param distance_threshold: Threshold distance to consider a point as an inlier.
    :param max_iterations: Maximum number of iterations for the RANSAC loop.
    :param iterations_per_batch: Number of iterations processed in each batch.
    :param epsilon: Small value to avoid division by zero.
    :param device: Device to run the computations (CPU/CUDA).
    :returns:
    - `best_line_direction`: Best line direction vector found.
    - `best_line_point`: A point on the best line found.
    - `best_inlier_indices`: Indices of points considered inliers.
    """

    # Move the point cloud to the specified device (CUDA or CPU)
    # pts = pts.to(device)
    num_pts = pts.shape[0]

    # Initialize variables to store the best result
    best_inlier_indices = torch.tensor([], dtype=torch.long, device=device)
    best_inlier_count = 0

    # Iterate over batches of iterations
    for start_idx in range(0, max_iterations, iterations_per_batch):
        end_idx = min(start_idx + iterations_per_batch, max_iterations)
        current_batch_size = end_idx - start_idx

        # Randomly sample 2 points for each iteration in the batch
        rand_pt_idx = torch.randint(
            0,
            num_pts,
            (current_batch_size, 2),
            device=device,
        )
        sampled_points = pts[rand_pt_idx]

        # Calculate direction vectors (line vectors) between pairs of points
        line_vectors = sampled_points[:, 1, :] - sampled_points[:, 0, :]

        # Normalize line vectors to unit length
        normalized_line_vectors = line_vectors / (
            torch.norm(line_vectors, dim=1, keepdim=True) + epsilon
        )

        # Expand the point cloud to match the batch size for pairwise distance calculations
        all_points_expanded = pts.unsqueeze(0).expand(current_batch_size, -1, -1)

        # Calculate vectors from the sampled start point to all other points
        point_to_line_vectors = (
            sampled_points[:, 0, :].unsqueeze(1) - all_points_expanded
        )

        # Calculate the perpendicular distances from all points to the line
        distances = torch.norm(
            torch.cross(
                normalized_line_vectors.unsqueeze(1),
                point_to_line_vectors,
                dim=2,
            ),
            dim=2,
        )

        # Create a mask for points that are within the distance threshold (inliers)
        inlier_mask = torch.abs(distances) <= thresh
        inlier_counts = inlier_mask.sum(dim=1)

        # Find the batch iteration with the most inliers
        best_iteration_in_batch = torch.argmax(inlier_counts)
        best_inlier_count_in_batch = inlier_counts[best_iteration_in_batch]

        # Update the best result if this iteration has more inliers than previous ones
        if best_inlier_count_in_batch > best_inlier_count:
            best_inlier_count = best_inlier_count_in_batch
            best_inlier_indices = torch.where(inlier_mask[best_iteration_in_batch])[0]
            best_line_direction = normalized_line_vectors[best_iteration_in_batch]
            best_line_point = sampled_points[best_iteration_in_batch, 0, :]

    # Return the best line direction, point, and inlier indices
    return best_line_direction, best_line_point, best_inlier_indices
