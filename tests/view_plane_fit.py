import torch
import numpy as np
from torch_ransac3d.plane import (
    plane_fit,
)
import open3d as o3d


def generate_plane_points(n_points=1000, noise_scale=0.2):

    x = np.random.uniform(-1, 1, n_points)
    y = np.random.uniform(-1, 1, n_points)
    z = x + y

    noise = np.random.normal(0, noise_scale, n_points)
    z += noise

    return np.column_stack((x, y, z))


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


if __name__ == "__main__":

    points = generate_plane_points(10000, 0.05)

    equation, inliers = plane_fit(points)

    plane_mesh = create_plane_mesh(equation, points[inliers])

    pcd = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(points)
    ).paint_uniform_color([1, 0, 0])

    plane_points = pcd.select_by_index(inliers).paint_uniform_color([0, 1, 0])

    o3d.visualization.draw_geometries([pcd, plane_points, plane_mesh])
