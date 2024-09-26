import torch
import numpy as np
from torch_ransac3d.sphere import (
    sphere_fit,
)
import open3d as o3d


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


if __name__ == "__main__":

    centre = np.array([0, 0, 0])
    radius = 1.0
    points = generate_sphere_points(centre, radius, num_points=1000, noise_scale=0.2)

    ransac_centre, ransac_radius, inliers = sphere_fit(points)

    print(ransac_centre, ransac_radius)

    sphere = create_sphere_mesh(ransac_centre, ransac_radius)

    pcd = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(points)
    ).paint_uniform_color([1, 0, 0])

    o3d.visualization.draw_geometries([pcd, sphere])
