import open3d as o3d
import numpy as np
import ast
import re

def message_to_pointcloud(msg):
    '''converts the string received back into a list of floats to construct a pointcloud
    Args:
        msg: the message to convert
    Return:
        pointcloud: the pointcloud data reconstructed from the message'''

    if isinstance(msg, bytes):
        msg = msg.decode("utf-8")

    if msg is None:
        return np.empty((0, 3), dtype=float)

    text = str(msg).strip()
    if not text:
        return np.empty((0, 3), dtype=float)

    try:
        if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
            text = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        pass

    if isinstance(text, (list, tuple)):
        pointcloud = np.asarray(text, dtype=float)
    else:
        text = str(text).strip()
        text = text.replace("np.array(", "").replace("array(", "")
        if text.endswith(")"):
            text = text[:-1]
        if text.startswith("[") and text.endswith("]"):
            rows = []
            for row in re.findall(r"\[[^\]]*\]", text):
                row_text = row.strip("[]")
                if not row_text.strip():
                    continue
                numbers = re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", row_text)
                if numbers:
                    rows.append([float(num) for num in numbers])
            if rows:
                pointcloud = np.asarray(rows, dtype=float)
            else:
                pointcloud = np.asarray(ast.literal_eval(text), dtype=float)
        else:
            try:
                pointcloud = np.asarray(ast.literal_eval(text), dtype=float)
            except (SyntaxError, ValueError) as exc:
                raise ValueError(f"unable to parse pointcloud payload: {msg!r}") from exc

    if pointcloud.ndim == 1:
        pointcloud = pointcloud.reshape(1, -1)

    return pointcloud

def normalise_cloud(pointcloud):
    '''normalises the pointcloud to center the x and y axis on 0 and the z axis lowest value to 0'''

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