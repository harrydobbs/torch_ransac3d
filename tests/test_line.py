import numpy as np
import pytest
import torch

from torch_ransac3d.line import line_fit


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def test_line_fit_perfect_line(device):
    points, true_direction, true_origin = generate_line_points(1000, 0, 0)
    points = points.to(device)

    result = line_fit(points, thresh=1e-5, device=device)

    assert torch.allclose(
        result.direction.abs(),
        torch.tensor(true_direction).abs().to(device).to(result.direction.dtype),
        atol=0.1,
    )


def test_line_fit_noisy_data(device):
    points, true_direction, true_origin = generate_line_points(1000, 0.001, 0)
    points = points.to(device)

    result = line_fit(
        points, thresh=0.5, device=device, max_iterations=10000
    )

    assert torch.allclose(
        result.direction.abs(),
        torch.tensor(true_direction).abs().to(device).to(result.direction.dtype),
        atol=0.1,
    )


def test_line_fit_with_outliers(device):
    points, true_direction, true_origin = generate_line_points(1000, 0.1, 0.2)
    points = points.to(device)

    result = line_fit(points, thresh=0.2, device=device)

    assert torch.allclose(
        result.direction.abs(),
        torch.tensor(true_direction, device=device).abs().to(result.direction.dtype),
        atol=1.0,
    )


def test_line_fit_edge_cases(device):
    # Test with minimal number of points
    points = torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.float32).to(device)
    result = line_fit(points, thresh=1e-5, device=device)
    assert len(result.inliers) == 2

    # Test with all points the same
    points = torch.ones((100, 3), dtype=torch.float32).to(device)
    result = line_fit(points, thresh=1e-5, device=device)
    assert len(result.inliers) == 100


def test_line_fit_performance(device):
    points, _, _ = generate_line_points(10000, 0.1, 0.1)
    points = points.to(device)

    import time

    start_time = time.time()
    result = line_fit(points, thresh=0.2, device=device)
    end_time = time.time()

    assert (
        end_time - start_time < 1.0
    )  # The function should complete in less than 1 second


if __name__ == "__main__":
    pytest.main([__file__])
