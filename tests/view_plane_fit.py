import numpy as np
import open3d as o3d
import torch

from torch_ransac3d.plane import plane_fit


def generate_plane_points(
    normal, d, num_points=1000, noise_std=0.0, outlier_fraction=0.0
):
    """Generate synthetic plane points with optional noise."""
    normal = np.array(normal, dtype=np.float32)
    d = np.float32(d)

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

    return points


def create_plane_mesh(equation, points, color=[0, 1, 0]):
    a, b, c, d = equation

    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])

    # Create a grid of points
    xx, yy = np.meshgrid(np.linspace(min_x, max_x, 20), np.linspace(min_y, max_y, 20))
    zz = (-a * xx - b * yy - d) / c

    # Create the mesh
    vertices = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    triangles = []
    for i in range(19):
        for j in range(19):
            triangles.append([i * 20 + j, i * 20 + j + 1, (i + 1) * 20 + j])
            triangles.append([i * 20 + j + 1, (i + 1) * 20 + j + 1, (i + 1) * 20 + j])

    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(vertices),
        triangles=o3d.utility.Vector3iVector(triangles),
    )
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh


def visualize_plane_fit(points, equation, inliers):
    # Create plane mesh
    plane_mesh = create_plane_mesh(equation.numpy(), points[inliers])

    # Create point cloud for all points (red)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([1, 0, 0])  # Red for all points

    # Create point cloud for inliers (blue)
    inlier_points = pcd.select_by_index(inliers.numpy())
    inlier_points.paint_uniform_color([0, 0, 1])  # Blue for inliers

    # Create coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1.0, origin=[0, 0, 0]
    )

    # Visualize
    o3d.visualization.draw_geometries([pcd, inlier_points, plane_mesh, coord_frame])


def main():
    """Run plane fitting demo."""
    # Generate synthetic plane data
    true_normal = np.array([1.0, 1.0, 1.0])
    true_normal = true_normal / np.linalg.norm(true_normal)
    true_d = 2.0
    points = generate_plane_points(true_normal, true_d, noise_std=0.1)

    # Fit plane
    result = plane_fit(points, thresh=0.1)

    # Print results
    print(f"True normal: {true_normal}")
    normal = result.equation[:3].numpy()
    normal = normal / np.linalg.norm(normal)
    print(f"Fitted normal: {normal}")
    print(f"True d: {true_d}")
    print(f"Fitted d: {result.equation[3].item()}")
    print(f"Number of inliers: {len(result.inliers)}")

    # Visualize results
    visualize_plane_fit(points, result.equation, result.inliers)


if __name__ == "__main__":
    main()
