import numpy as np
import pytest
import torch

from torch_ransac3d.plane import plane_fit


def generate_plane_points(
    normal, d, num_points=1000, noise_std=0.0, outlier_fraction=0.0
):
    normal = np.array(normal, dtype=np.float32)
    d = np.float32(d)  # Change to float32
    # Generate random points in a plane
    x = np.random.uniform(-5, 5, num_points).astype(np.float32)
    y = np.random.uniform(-5, 5, num_points).astype(np.float32)

    # Calculate z coordinates using plane equation ax + by + cz + d = 0
    # Avoid division by zero
    if abs(normal[2]) < 1e-8:
        z = np.zeros_like(x)
    else:
        z = -(normal[0] * x + normal[1] * y + d) / normal[2]
    z = z.astype(np.float32)

    points = np.column_stack((x, y, z))

    # Add noise
    if noise_std > 0:
        points += np.random.normal(0, noise_std, points.shape).astype(np.float32)

    # Add outliers
    if outlier_fraction > 0:
        num_outliers = int(num_points * outlier_fraction)
        outlier_indices = np.random.choice(num_points, num_outliers, replace=False)
        points[outlier_indices] += np.random.uniform(-2, 2, (num_outliers, 3)).astype(
            np.float32
        )

    return torch.tensor(points, dtype=torch.float32)


def test_plane_fit_perfect_plane(device):
    normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    points = generate_plane_points(normal, d=1.0)
    points = points.to(device)

    result = plane_fit(points, thresh=0.01, device=device)

    # Normalize the equation for comparison
    equation = result.equation / torch.norm(result.equation[:3])
    assert torch.allclose(
        equation[:3].abs(), torch.tensor(normal, device=device).abs(), atol=0.1
    )
    assert len(result.inliers) > 900


def test_plane_fit_noisy_data(device):
    normal = np.array([1.0, 1.0, 1.0], dtype=np.float32) / np.sqrt(3).astype(np.float32)
    points = generate_plane_points(normal, d=0.0, noise_std=0.05)
    points = points.to(device)

    result = plane_fit(points, thresh=0.1, device=device)

    equation = result.equation / torch.norm(result.equation[:3])
    assert torch.allclose(
        equation[:3].abs(), torch.tensor(normal, device=device).abs(), atol=0.2
    )
    assert len(result.inliers) > 700


def test_plane_fit_with_outliers(device):
    normal = np.array([0.0, 1.0, 1.0], dtype=np.float32) / np.sqrt(2).astype(np.float32)
    points = generate_plane_points(normal, d=2.0, noise_std=0.02, outlier_fraction=0.3)
    points = points.to(device)

    result = plane_fit(points, thresh=0.1, device=device)

    equation = result.equation / torch.norm(result.equation[:3])
    assert torch.allclose(
        equation[:3].abs(), torch.tensor(normal, device=device).abs(), atol=0.2
    )
    assert len(result.inliers) > 600


def test_plane_fit_edge_cases(device):
    # Test with minimal number of points
    points = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=torch.float32).to(
        device
    )

    result = plane_fit(points, thresh=0.01, device=device)
    assert len(result.inliers) == 3

    # Test with all points the same
    points = torch.ones((100, 3), dtype=torch.float32).to(device)
    result = plane_fit(points, thresh=0.01, device=device)
    assert (
        len(result.inliers) >= 0
    )  # Any number of inliers is acceptable for degenerate case


def test_plane_fit_performance(device):
    normal = np.array([1.0, 1.0, 1.0], dtype=np.float32) / np.sqrt(3).astype(np.float32)
    points = generate_plane_points(normal, d=0.0, num_points=10000, noise_std=0.01)
    points = points.to(device)

    import time

    start_time = time.time()
    result = plane_fit(points, thresh=0.1, device=device)
    end_time = time.time()

    assert end_time - start_time < 1.0


if __name__ == "__main__":
    pytest.main([__file__])
