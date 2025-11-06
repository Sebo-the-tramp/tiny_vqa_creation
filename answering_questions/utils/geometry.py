import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon

## Geometry utilities

signs = np.array([
    [-1, -1, -1],
    [-1, -1,  1],
    [-1,  1, -1],
    [-1,  1,  1],
    [ 1, -1, -1],
    [ 1, -1,  1],
    [ 1,  1, -1],
    [ 1,  1,  1],
], dtype=float)

def OBB_to_eight_points(obb):
    """
    Convert an oriented bounding box (OBB) to its eight corner points.

    Parameters:
    obb (np.ndarray): A 3x4 array where the first three columns represent the rotation matrix
                      and the last column represents the center of the OBB.

    Returns:
    np.ndarray: An 8x3 array containing the coordinates of the eight corner points.
    """
    R = np.array(obb["R"])  # Rotation matrix (3x3)
    center = np.array(obb["center"])  # Center of the OBB (3,)
    extents = np.array(obb["extents"])  # Extents along each axis (3,)

    corners = center + (signs * extents/2) @ R.T
    corner_points = [corners[i] for i in range(8)]

    return corner_points

def _norm(v):
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Zero-length vector")
    return v / n

def get_RT(cam, default_up=(0,0,1)):
    # Build a camera that looks along +Y in world space
    eye = np.asarray(cam["eye"], float)
    at  = np.asarray(cam["at"], float)
    upw = np.asarray(cam.get("up", default_up), float)

    f = _norm(at - eye)          # forward (+Y when looking purely along Y)
    r = _norm(np.cross(f, upw))  # right
    u = np.cross(r, f)           # true up

    # World to this Y-forward camera
    R_y = np.vstack([r, -u, f])   # rows are camera axes
    t_y = -R_y @ eye

    return R_y, t_y

def project_points(Xw, cam):
    K = np.array(
        [cam['height'] / (2 * np.tan(cam['fov'] / 2)), 0, cam['width'] / 2,
         0, cam['height'] / (2 * np.tan(cam['fov'] / 2)), cam['height'] / 2,
         0, 0, 1]
    ).reshape(3, 3)    
    R, t = get_RT(cam)
    Xc = (R @ Xw.T + t[:, None]).T
    z = Xc[:, 2]
    uv = (K @ Xc.T).T
    uv = uv[:, :2] / z[:, None]
    return uv, z


def polygon_area(poly):
    if len(poly) < 3:
        return 0.0
    x, y = poly[:,0], poly[:,1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def external_points_2d(points):
    """
    Given Nx2 points, returns only the external ones (convex hull vertices).
    """
    points = np.asarray(points, float)
    if len(points) < 3:
        return points  # all points are external if <3
    hull = ConvexHull(points)
    return points[hull.vertices]