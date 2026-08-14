import open3d as o3d
import math
import numpy as np

def _display_data(pointcloud):
    '''a helper function that displays the pointcloud data so it can be interpreted and debugged easier
    Args: 
        pointcloud: the pointcloud data, a list of touples containing the x, y and z of each point
        '''
    
    # Open3D visualization expects an Open3D geometry object, not a raw numpy array.
    point_cloud_for_display = o3d.geometry.PointCloud()
    point_cloud_for_display.points = o3d.utility.Vector3dVector(np.asarray(pointcloud, dtype=np.float64))
    try:
        o3d.visualization.draw_geometries([point_cloud_for_display],
                                        zoom=0.3412,
                                        front=[0.4257, -0.2125, -0.8795],
                                        lookat=[2.6172, 2.0475, 1.532],
                                        up=[-0.0694, -0.9768, 0.2024])
    except Exception as draw_error:
        print(f"Visualization skipped: {draw_error}")

def get_subject_depth(pointcloud):
    '''returns the depth of the subject
    Args:
        pointcloud: the pointcloud data
    Returns:
        depth: the difference between the highest and lowest point.'''
    
    if len(pointcloud) == 0:
        raise ValueError("pointcloud cannot be empty")
    
    highest_z = -math.inf
    lowest_z = math.inf
    depth = 0

    for point in pointcloud:
        if point[2]>highest_z: highest_z = point[2]
        if point[2]<lowest_z: lowest_z = point[2] 

    print(f"highest point of {highest_z} lowest point of {lowest_z}")

    depth = highest_z-lowest_z

    if depth <= 0:
        raise ValueError(f"depth of {depth} is not valid, must be positive number above 0")

    return depth

def get_subject_radius(pointcloud):
    '''returns the radius of the subject, assuming the subject is spherical
    using the average difference between the largest X and Y coordenates
    Args:
        pointcloud: a list of point data
    Returns:
        radius: the radius of the subject'''
    if len(pointcloud) == 0:
            raise ValueError("pointcloud cannot be empty")
        
    highest_x = -math.inf
    lowest_x = math.inf
    highest_y = -math.inf
    lowest_y = math.inf
    radius = 0

    for point in pointcloud:
        if point[0]>highest_x: highest_x = point[0]
        if point[0]<lowest_x: lowest_x = point[0]
        if point[1]>highest_y: highest_y = point[1]
        if point[1]<lowest_y: lowest_y = point[1]

    radius = ((highest_x-lowest_x)+(highest_y-lowest_y))/2

    if radius <= 0:
        raise ValueError(f"radius of {radius} is not valid, must be positive number above 0")

    return radius

def _find_edges(pointcloud, segment_Angle=1):
    '''returns the outermost points from the center of the cloud, sampled by angular segment
    Args:
        pointcloud: a list of point data
        segment_Angle: angular width of each sampling segment in degrees
    Returns:
        edge_cloud: a pointcloud array of points corresponding to the edges of the subject'''
    if len(pointcloud) == 0:
        raise ValueError("pointcloud cannot be empty")
    if not 0 < segment_Angle <= 360:
        raise ValueError(f"segment angle of {segment_Angle} is not valid, must be between 0 and 360")

    point_array = np.asarray(pointcloud, dtype=float)
    if point_array.ndim != 2 or point_array.shape[1] < 2:
        raise ValueError("pointcloud must contain 2D or 3D point data")

    xy_points = point_array[:, :2]
    center_point = np.mean(xy_points, axis=0)
    relative_points = xy_points - center_point

    if np.allclose(relative_points, 0):
        return np.empty((0, point_array.shape[1]), dtype=float)

    angles = np.arctan2(relative_points[:, 1], relative_points[:, 0])
    angles = (angles + 2 * math.pi) % (2 * math.pi)

    segment_radians = math.radians(segment_Angle)
    bin_count = max(1, int(round((2 * math.pi) / segment_radians)))
    bin_width = (2 * math.pi) / bin_count

    edge_map = {}
    for point, angle in zip(point_array, angles):
        bin_index = int(angle // bin_width) % bin_count
        radius = float(np.linalg.norm(point[:2] - center_point))
        if bin_index not in edge_map or radius > edge_map[bin_index][0]:
            edge_map[bin_index] = (radius, point.copy())

    edge_points = [edge_map[index][1] for index in sorted(edge_map)]
    if not edge_points:
        raise ValueError("no edge points found in pointcloud")

    return np.asarray(edge_points, dtype=float)

def get_subject_perimeter(pointcloud, segment_Angle=1):
    '''returns the perimeter of the subject by interpolating between detected edge points
    Args: 
        pointcloud: the pointcloud data
    Returns:
        perimeter: the perimeter of the subject'''

    if len(pointcloud) == 0:
        raise ValueError("pointcloud cannot be empty")
    if not 0 < segment_Angle <= 360:
        raise ValueError(f"segment angle of {segment_Angle} is not valid, must be between 0 and 360")

    point_array = np.asarray(pointcloud, dtype=float)
    if point_array.ndim != 2 or point_array.shape[1] < 2:
        raise ValueError("pointcloud must contain 2D or 3D point data")

    edge_points = _find_edges(point_array, segment_Angle)
    _display_data(edge_points)
    if len(edge_points) < 2:
        return 0.0

    center_point = np.mean(point_array[:, :2], axis=0)
    angles = np.arctan2(edge_points[:, 1] - center_point[1], edge_points[:, 0] - center_point[0])
    ordered_points = edge_points[np.argsort(angles)]

    perimeter = 0.0
    for index in range(len(ordered_points)):
        start = ordered_points[index]
        end = ordered_points[(index + 1) % len(ordered_points)]

        interpolated = np.linspace(start, end, num=10, endpoint=True)
        segment_lengths = np.linalg.norm(np.diff(interpolated, axis=0), axis=1)
        perimeter += float(np.sum(segment_lengths))

    return perimeter