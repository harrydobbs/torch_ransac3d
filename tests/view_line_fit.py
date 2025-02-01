import numpy as np
import open3d as o3d
import torch
from typing import List, Optional

from torch_ransac3d.line import line_fit


def generate_line_points(
    direction: np.ndarray,
    point: np.ndarray,
    num_points: int = 1000,
    noise_scale: float = 0,
    line_length: float = 2.0,
) -> np.ndarray:
    """Generate synthetic line points with optional noise."""
    # Normalize direction vector
    direction = direction / np.linalg.norm(direction)

    # Generate points along the line
    t = np.linspace(-line_length / 2, line_length / 2, num_points)
    points = point + np.outer(t, direction)

    # Add noise perpendicular to line direction
    if noise_scale > 0:
        # Create random vectors
        noise = np.random.normal(0, noise_scale, (num_points, 3))
        # Project noise to be perpendicular to line direction
        noise = noise - np.outer(np.dot(noise, direction), direction)
        points = points + noise

    return points.astype(np.float32)


def create_line_mesh(
    direction, 
    point, 
    points, 
    color: Optional[List[float]] = None
) -> o3d.geometry.TriangleMesh:
    """Create a cylinder mesh representing the line.
    
    Args:
        direction: Direction vector of the line
        point: Point on the line
        points: Point cloud
        color: RGB color values, defaults to [0, 1, 0]
    """
    if color is None:
        color = [0, 1, 0]

    # Ensure direction is normalized
    direction = direction / (np.linalg.norm(direction) + 1e-8)

    # Use points extent to determine line length
    proj = np.dot(points - point, direction)
    min_t, max_t = np.min(proj), np.max(proj)

    # Ensure cylinder has positive height
    if max_t <= min_t:
        max_t = min_t + 1.0

    # Create cylinder along the line
    radius = 0.02  # Small radius for visualization
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radius, height=max_t - min_t
    )

    # Rotate cylinder to align with line direction
    z_axis = np.array([0, 0, 1])
    # Handle case where direction is zero or parallel to z-axis
    if np.allclose(direction, 0) or np.allclose(direction, z_axis):
        rotation_matrix = np.eye(3)
    else:
        rotation_axis = np.cross(z_axis, direction)
        rotation_angle = np.arccos(np.dot(z_axis, direction))
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(
            rotation_axis * rotation_angle
        )
    # Get cylinder center for rotation
    cylinder_center = np.mean(np.asarray(cylinder.vertices), axis=0)
    cylinder.rotate(rotation_matrix, center=cylinder_center)

    # Move to correct position
    start_point = point + min_t * direction
    cylinder.translate(start_point + (max_t - min_t) / 2 * direction)

    cylinder.paint_uniform_color(color)
    cylinder.compute_vertex_normals()
    return cylinder


def visualize_line_fit(points, direction, point, inliers):
    """Visualize line fitting results."""
    # Create line mesh
    line = create_line_mesh(direction.numpy(), point.numpy(), points)

    # Create point cloud for all points (red)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([1, 0, 0])  # Red for all points

    # Create point cloud for inliers (blue)
    inlier_points = pcd.select_by_index(inliers.numpy())
    inlier_points.paint_uniform_color([0, 0, 1])  # Blue for inliers

    # Create coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.5, origin=point.numpy()
    )

    # Visualize
    o3d.visualization.draw_geometries([pcd, inlier_points, line, coord_frame])


def main():
    # Generate synthetic line data
    true_direction = np.array([1.0, 1.0, 1.0])
    true_direction = true_direction / np.linalg.norm(true_direction)
    true_point = np.array([0.0, 0.0, 0.0])
    points = generate_line_points(
        true_direction, true_point, noise_scale=0.1, line_length=5.0
    )

    # Fit line
    result = line_fit(points, thresh=0.2)

    # Print results
    print(f"True direction: {true_direction}")
    print(f"Fitted direction: {result.direction.numpy()}")
    print(f"True point: {true_point}")
    print(f"Fitted point: {result.point.numpy()}")
    print(f"Number of inliers: {len(result.inliers)}")

    # Visualize results
    visualize_line_fit(points, result.direction, result.point, result.inliers)


if __name__ == "__main__":
    main()
