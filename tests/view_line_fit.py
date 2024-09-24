import torch
import numpy as np
from torch_ransac3d.line import (
    line_fit,
)
import open3d as o3d


def generate_line_points(num_points, noise_std, outlier_fraction):
    # Generate points along a line with some noise and outliers
    t = np.linspace(0, 10, num_points)
    line_direction = np.array([1.0, 2.0, 3.0])
    line_direction /= np.linalg.norm(line_direction)
    line_origin = np.array([0.0, 1.0, 2.0])

    points = line_origin + np.outer(t, line_direction)
    points += np.random.normal(0, noise_std, points.shape)

    # Add outliers
    num_outliers = int(num_points * outlier_fraction)
    outlier_indices = np.random.choice(num_points, num_outliers, replace=False)
    points[outlier_indices] += np.random.uniform(-5, 5, (num_outliers, 3))

    return torch.tensor(points, dtype=torch.float32), line_direction, line_origin


if __name__ == "__main__":

    points, line_direction, line_origin = generate_line_points(1000, 0, 0)

    pcd = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(points)
    ).paint_uniform_color([1, 0, 0])

    o3d.visualization.draw_geometries([pcd])
