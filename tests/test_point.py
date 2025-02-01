import numpy as np
import pytest
import torch

from torch_ransac3d.point import point_fit


def generate_cluster_points(
    center, num_points=1000, noise_std=0.0, outlier_fraction=0.0
):
    center = np.array(center, dtype=np.float32)
    points = np.random.normal(center, noise_std, (num_points, 3)).astype(np.float32)

    # Add outliers
    if outlier_fraction > 0:
        num_outliers = int(num_points * outlier_fraction)
        outlier_indices = np.random.choice(num_points, num_outliers, replace=False)
        points[outlier_indices] = np.random.uniform(-5, 5, (num_outliers, 3)).astype(
            np.float32
        )

    return torch.tensor(points, dtype=torch.float32)


def test_point_fit_perfect_cluster(device):
    true_center = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    points = generate_cluster_points(true_center, noise_std=0.1)
    points = points.to(device)

    result = point_fit(points, thresh=0.2, device=device)

    assert torch.allclose(
        result.center, torch.tensor(true_center, device=device), atol=0.2
    )
    assert len(result.inliers) > 700  # Adjust threshold for more realistic expectation


def test_point_fit_noisy_data(device):
    true_center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    points = generate_cluster_points(true_center, noise_std=0.3)
    points = points.to(device)

    result = point_fit(points, thresh=0.5, device=device)

    assert torch.allclose(
        result.center, torch.tensor(true_center, device=device), atol=0.5
    )
    assert len(result.inliers) > 500  # Lower threshold for noisy data


def test_point_fit_with_outliers(device):
    true_center = np.array([-1.0, 1.0, 1.0], dtype=np.float32)
    points = generate_cluster_points(true_center, noise_std=0.2, outlier_fraction=0.3)
    points = points.to(device)

    result = point_fit(points, thresh=0.4, device=device)

    assert torch.allclose(
        result.center, torch.tensor(true_center, device=device), atol=0.5
    )
    assert len(result.inliers) > 500  # Adjust threshold for high noise and outliers


def test_point_fit_edge_cases(device):
    # Test with minimal number of points
    points = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32).to(device)
    result = point_fit(points, thresh=0.1, device=device)
    assert len(result.inliers) == 1

    # Test with all points the same
    points = torch.ones((100, 3), dtype=torch.float32).to(device)
    result = point_fit(points, thresh=0.1, device=device)
    assert len(result.inliers) == 100


def test_point_fit_performance(device):
    points = generate_cluster_points(
        np.array([0.0, 0.0, 0.0]), num_points=10000, noise_std=0.1
    )
    points = points.to(device)

    import time

    start_time = time.time()
    result = point_fit(points, thresh=0.2, device=device)
    end_time = time.time()

    assert end_time - start_time < 1.0


if __name__ == "__main__":
    pytest.main([__file__])
