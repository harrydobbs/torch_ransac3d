import torch
import numpy as np
from torch_ransac3d.line import (
    line_fit,
)
import open3d as o3d


def generate_line_points(
    num_points,
    line_length,
    line_direction,
    line_origin,
    noise_std,
    outlier_fraction,
):
    # Generate points along a line with some noise and outliers
    t = np.linspace(0, line_length, num_points)

    points = line_origin + np.outer(t, line_direction)
    points += np.random.normal(0, noise_std, points.shape)

    # Add outliers
    num_outliers = int(num_points * outlier_fraction)
    outlier_indices = np.random.choice(num_points, num_outliers, replace=False)
    points[outlier_indices] += np.random.uniform(-5, 5, (num_outliers, 3))

    return torch.tensor(points, dtype=torch.float32)


if __name__ == "__main__":

    line_direction = np.array([1.0, 0.0, 0.0])
    line_origin = np.array([0.0, 0.0, 0.0])
    line_length = 10.0

    points = generate_line_points(
        1000,
        line_length,
        line_direction,
        line_origin,
        0.001,
        0.0,
    )

    ransac_direction, ransac_point, inliers = line_fit(points, max_iterations=1000)

    t = np.linalg.norm(points[0] - ransac_point) / line_length

    line_end = ransac_point + ransac_direction * ((1 - t) * line_length)

    line_start = ransac_point - ransac_direction * (t * line_length)

    vertices = np.array([[line_start], [line_end]]).reshape(-1, 3)
    edges = np.array([[0], [1]]).reshape(-1, 2)

    line_set = o3d.geometry.LineSet(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector2iVector(edges),
    ).paint_uniform_color([0, 1, 0])

    pcd = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(points)
    ).paint_uniform_color([1, 0, 0])

    o3d.visualization.draw_geometries([pcd, line_set])
