import open3d as o3d
import numpy as np

def _downsample_pointcloud(pointcloud, voxel_size:float=0.05):

    downpcd = pointcloud.voxel_down_sample(voxel_size=voxel_size)

    return downpcd

def cloud_to_mesh(pointcloud):
    '''Converts the cloud data to a mesh to make it easier to work with
    Args: 
        pointcloud: a list of touples containing the x, y and z axis data for the object scanned
    Returns:
        mesh: the mesh object returned'''
    downsampled_cloud = _downsample_pointcloud(pointcloud)
    downsampled_cloud.estimate_normals()
    distances = downsampled_cloud.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 1.5 * avg_dist

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        downsampled_cloud,
        o3d.utility.DoubleVector([radius, radius * 2])
    )
    o3d.io.write_triangle_mesh("output_mesh.ply", mesh)
    o3d.visualization.draw_geometries([mesh])
    return mesh

def denoise_data(mesh_in, iterations):
    '''Cleans the noise from mesh data and returns the noiseless mesh data using a mesh filter
    Args:
        mesh_in: a list of touples containing x, y and z positions
        iterations: the number of times the smoothing filter is applied
    Returns
        noiseless_mesh: a list of touples that have been denoised'''

    mesh_out = mesh_in.filter_smooth_simple(number_of_iterations=iterations)
    mesh_out.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh_out])

    return mesh_out