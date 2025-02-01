import numpy as np
import pytest
import torch

from torch_ransac3d.cuboid import cuboid_fit


def generate_cuboid_points(
    center,
    dimensions,
    rotation_matrix,
    num_points=1000,
    noise_std=0.0,
    outlier_fraction=0.0,
):
    # Generate points on each face of the cuboid
    points_per_face = num_points // 6
    num_points = points_per_face * 6  # Ensure total points is divisible by 6
    points = []

    # Define the six faces of the cuboid
    faces = [
        ([1, 0, 0], dimensions[0] / 2),  # +x face
        ([-1, 0, 0], dimensions[0] / 2),  # -x face
        ([0, 1, 0], dimensions[1] / 2),  # +y face
        ([0, -1, 0], dimensions[1] / 2),  # -y face
        ([0, 0, 1], dimensions[2] / 2),  # +z face
        ([0, 0, -1], dimensions[2] / 2),  # -z face
    ]

    for normal, d in faces:
        # Generate random points on the face
        if normal[0] != 0:  # x-face
            y = np.random.uniform(
                -dimensions[1] / 2, dimensions[1] / 2, points_per_face
            )
            z = np.random.uniform(
                -dimensions[2] / 2, dimensions[2] / 2, points_per_face
            )
            x = np.full_like(y, d)
            face_points = np.column_stack((x, y, z))
        elif normal[1] != 0:  # y-face
            x = np.random.uniform(
                -dimensions[0] / 2, dimensions[0] / 2, points_per_face
            )
            z = np.random.uniform(
                -dimensions[2] / 2, dimensions[2] / 2, points_per_face
            )
            y = np.full_like(x, d)
            face_points = np.column_stack((x, y, z))
        else:  # z-face
            x = np.random.uniform(
                -dimensions[0] / 2, dimensions[0] / 2, points_per_face
            )
            y = np.random.uniform(
                -dimensions[1] / 2, dimensions[1] / 2, points_per_face
            )
            z = np.full_like(x, d)
            face_points = np.column_stack((x, y, z))

        points.append(face_points)

    points = np.vstack(points)

    # Apply rotation
    points = np.dot(points, rotation_matrix.T)

    # Translate to center
    points += center

    # Add noise
    if noise_std > 0:
        points += np.random.normal(0, noise_std, points.shape)

    # Add outliers
    if outlier_fraction > 0:
        num_outliers = int(num_points * outlier_fraction)
        outlier_indices = np.random.choice(num_points, num_outliers, replace=False)
        max_dim = max(dimensions)
        points[outlier_indices] += np.random.uniform(
            -max_dim, max_dim, (num_outliers, 3)
        )

    return torch.tensor(points, dtype=torch.float32)


def test_cuboid_fit_perfect_cuboid(device):
    center = np.array([1.0, 2.0, 3.0])
    dimensions = np.array([2.0, 3.0, 4.0])
    rotation_matrix = np.eye(3)  # No rotation
    points = generate_cuboid_points(center, dimensions, rotation_matrix)
    points = points.to(device)

    equations, inliers = cuboid_fit(points, thresh=0.01, device=device)

    assert equations.shape == (3, 4)  # Three plane equations
    assert len(inliers) > 900  # Most points should be inliers


def test_cuboid_fit_noisy_data(device):
    center = np.array([0.0, 0.0, 0.0])
    dimensions = np.array([1.0, 1.0, 1.0])
    # Create a rotation matrix for 45 degrees around z-axis
    theta = np.pi / 4
    rotation_matrix = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    points = generate_cuboid_points(center, dimensions, rotation_matrix, noise_std=0.05)
    points = points.to(device)

    equations, inliers = cuboid_fit(points, thresh=0.1, device=device)

    assert equations.shape == (3, 4)
    assert len(inliers) > 700


def test_cuboid_fit_with_outliers(device):
    center = np.array([-1.0, 1.0, 1.0])
    dimensions = np.array([2.0, 1.5, 1.0])
    rotation_matrix = np.eye(3)
    points = generate_cuboid_points(
        center, dimensions, rotation_matrix, noise_std=0.02, outlier_fraction=0.3
    )
    points = points.to(device)

    equations, inliers = cuboid_fit(points, thresh=0.1, device=device)

    assert equations.shape == (3, 4)
    assert len(inliers) > 600


def test_cuboid_fit_edge_cases(device):
    # Test with minimal number of points
    points = torch.tensor(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1]],
        dtype=torch.float32,
    ).to(device)

    equations, inliers = cuboid_fit(points, thresh=0.1, device=device)
    assert equations.shape == (3, 4)
    assert len(inliers) >= 6

    # Test with all points the same
    points = torch.ones((100, 3), dtype=torch.float32).to(device)
    equations, inliers = cuboid_fit(points, thresh=0.1, device=device)
    assert equations.shape == (3, 4)
    assert len(inliers) == 100


def test_cuboid_fit_performance(device):
    center = np.array([0.0, 0.0, 0.0])
    dimensions = np.array([2.0, 2.0, 2.0])
    rotation_matrix = np.eye(3)
    points = generate_cuboid_points(
        center, dimensions, rotation_matrix, num_points=10000, noise_std=0.01
    )
    points = points.to(device)

    import time

    start_time = time.time()
    equations, inliers = cuboid_fit(points, thresh=0.1, device=device)
    end_time = time.time()

    assert end_time - start_time < 1.0


if __name__ == "__main__":
    pytest.main([__file__])
