import numpy as np
import open3d as o3d
import torch

from torch_ransac3d.sphere import sphere_fit


def generate_sphere_points(
    centre: np.ndarray,
    radius: float,
    num_points: int = 1000,
    noise_scale: float = 0,
) -> np.ndarray:
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    phi = np.arccos(1 - 2 * np.random.uniform(0, 1, num_points))

    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)

    noise = np.random.normal(0, noise_scale, num_points)
    z += noise

    return np.column_stack((x, y, z)) + centre


def create_sphere_mesh(center, radius, resolution=20):
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(
        radius=radius, resolution=resolution
    )
    mesh_sphere.compute_vertex_normals()
    mesh_sphere.paint_uniform_color([0, 1.0, 0])
    mesh_sphere.translate(center)
    return mesh_sphere


def visualize_sphere_fit(points, center, radius, inliers):
    # Create sphere mesh
    sphere = create_sphere_mesh(center.numpy(), radius)

    # Create point cloud for all points (red)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([1, 0, 0])  # Red for all points

    # Create point cloud for inliers (blue)
    inlier_points = pcd.select_by_index(inliers.numpy())
    inlier_points.paint_uniform_color([0, 0, 1])  # Blue for inliers

    # Create coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1.0, origin=center.numpy()
    )

    # Visualize
    o3d.visualization.draw_geometries([pcd, inlier_points, sphere, coord_frame])


def main():
    # Generate synthetic sphere data
    n_points = 1000
    true_center = np.array([1.0, 2.0, 3.0])
    true_radius = 2.0
    points = generate_sphere_points(true_center, true_radius, noise_scale=0.1)

    # Fit sphere
    result = sphere_fit(points, thresh=0.2)

    # Print results
    print(f"True center: {true_center}")
    print(f"Fitted center: {result.center.numpy()}")
    print(f"True radius: {true_radius}")
    print(f"Fitted radius: {result.radius}")
    print(f"Number of inliers: {len(result.inliers)}")

    # Visualize results
    visualize_sphere_fit(points, result.center, result.radius, result.inliers)


if __name__ == "__main__":
    main()
