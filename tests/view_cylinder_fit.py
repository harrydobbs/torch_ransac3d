import torch
import numpy as np
import open3d as o3d
from torch_ransac3d.cylinder import (
    cylinder_fit,
    estimate_normals,
)  # Assuming the previous code is in cylinder_fit.py


def generate_cylinder_points(n_points, center, axis, radius, height, noise_level=0.05):
    """Generate synthetic points on a cylinder surface with noise."""
    # Generate random heights along the cylinder axis
    h = np.random.uniform(0, height, n_points)

    # Generate random angles around the cylinder
    theta = np.random.uniform(0, 2 * np.pi, n_points)

    # Calculate points on the cylinder surface
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    # Rotate the cylinder to align with the specified axis
    rotation_axis = np.cross([0, 0, 1], axis)
    rotation_angle = np.arccos(np.dot([0, 0, 1], axis))
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(
        rotation_axis * rotation_angle
    )

    points = np.dot(rotation_matrix, np.vstack((x, y, h)))
    points = points.T + center

    # Add noise
    noise = np.random.normal(0, noise_level, points.shape)
    points += noise

    return torch.tensor(points, dtype=torch.float32)


def create_cylinder_mesh(center, axis, radius, height, resolution=50):
    """Create an Open3D cylinder mesh."""
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radius, height=height, resolution=resolution
    )

    # Rotate cylinder to align with axis
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(
        np.cross([0, 0, 1], axis) * np.arccos(np.dot([0, 0, 1], axis))
    )
    cylinder.rotate(rotation_matrix, center=True)

    # Move cylinder to center
    cylinder.translate(center)

    return cylinder


def visualize_cylinder_fit(points, center, axis, radius, inliers):
    # Create Open3D point cloud for all points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.numpy())

    # Color all points blue
    colors = np.zeros_like(points.numpy()) + [0, 0, 1]  # Blue

    # Color inliers red
    colors[inliers.numpy()] = [1, 0, 0]  # Red
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Create cylinder mesh
    height = np.max(points.numpy(), axis=0) - np.min(points.numpy(), axis=0)
    height = np.linalg.norm(height)
    cylinder_mesh = create_cylinder_mesh(
        center.numpy(), axis.numpy(), radius.item(), height
    )
    cylinder_mesh.paint_uniform_color([0, 1, 0])  # Green

    # Create coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1.0, origin=center.numpy()
    )

    # Visualize
    o3d.visualization.draw_geometries([pcd, cylinder_mesh, coord_frame])


def main():
    # Generate synthetic cylinder data
    n_points = 1000
    true_center = np.array([1.0, 2.0, 3.0])  # Explicitly use floating-point values
    true_axis = np.array([1.0, 1.0, 1.0])  # Explicitly use floating-point values
    true_axis /= np.linalg.norm(true_axis)
    true_radius = 2.0
    height = 10.0
    noise_level = 0.1

    points = generate_cylinder_points(
        n_points, true_center, true_axis, true_radius, height, noise_level
    )

    # Estimate normals
    normals = estimate_normals(points)

    # Fit cylinder
    center, axis, radius, inliers = cylinder_fit(points, normals)

    # Print results
    print(f"True center: {true_center}")
    print(f"Fitted center: {center.numpy()}")
    print(f"True axis: {true_axis}")
    print(f"Fitted axis: {axis.numpy()}")
    print(f"True radius: {true_radius}")
    print(f"Fitted radius: {radius.item()}")
    print(f"Number of inliers: {len(inliers)}")

    # Visualize results
    visualize_cylinder_fit(points, center, axis, radius, inliers)


if __name__ == "__main__":
    main()
