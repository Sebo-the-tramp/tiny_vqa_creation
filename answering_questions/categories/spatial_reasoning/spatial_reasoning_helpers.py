from __future__ import annotations

import random

import numpy as np

from typing import Any, Mapping, Optional, Tuple, Union, List

from utils.helpers import as_vector
from utils.my_exception import ImpossibleToAnswer

from scipy.spatial.transform import Rotation as R


# set random seed for reproducibility
rng = random.Random(42)

Number = Union[int, float]
WorldState = Mapping[str, Any]
QuestionPayload = Mapping[str, Any]
Answer = Union[int, float, str]


from utils.config import get_config


AXIS_TO_NUM = {"X": 0, "Y": 1, "Z": 2}
WORLD_UP_AXIS = get_config()["world_up_axis"]  # 0=X, 1=Y, 2=Z for World Up
WORLD_UP_AXIS_NUM = AXIS_TO_NUM[WORLD_UP_AXIS]


def point_to_plane_distance(point, normal, d):
    """
    Compute the signed distance from a point to a plane.

    Plane: a*x + b*y + c*z = d
    point: (x, y, z)
    normal: (a, b, c)
    """
    point = np.array(point)
    normal = np.array(normal)
    normal = normal / np.linalg.norm(normal)  # ensure unit normal
    return np.dot(normal, point) - d


def get_position(
    world_state: Mapping[str, Any], object_id: str, timestep: str
) -> Optional[Tuple[float, ...]]:
    timestep_world = world_state["simulation"][timestep]
    current_timestep_involved_object = timestep_world["objects"][object_id]["obb"][
        "center"
    ]
    return as_vector(current_timestep_involved_object)


def get_position_camera(
    world_state: Mapping[str, Any], timestep: str
) -> Optional[Tuple[float, ...]]:
    timestep_world = world_state["simulation"][timestep]
    current_timestep_involved_object = timestep_world["camera"]["eye"]
    return as_vector(current_timestep_involved_object)


def get_max_height_from_obb(obb: Mapping[str, Any]) -> float:
    """
    Given an oriented bounding box (obb), return the maximum height (z coordinate) of the object.
    """
    center = np.array(obb["center"])
    extents = np.array(obb["extents"])
    axes = np.array(obb["R"])  # 3x3 rotation matrix
    up = np.array([0.0, 0.0, 1.0])

    # Fast path: choose the sign of each extent by the up-dot for each axis
    signs = np.sign(axes.T @ up)  # shape (3,)
    p_high = center + axes @ (signs * extents)

    return p_high[2]  # return the z coordinate since z is up


def get_min_height_from_obb(obb: Mapping[str, Any]) -> float:
    """
    Given an oriented bounding box (obb), return the minimum height (z coordinate) of the object.
    """
    center = np.array(obb["center"])
    extents = np.array(obb["extents"])
    axes = np.array(obb["R"])  # 3x3 rotation matrix
    up = np.array([0.0, 0.0, 1.0])

    # Fast path: choose the sign of each extent by the up-dot for each axis
    signs = -np.sign(axes.T @ up)  # shape (3,)
    p_low = center + axes @ (signs * extents)

    return max(
        0, p_low[2]
    )  # return the z coordinate since z is up # should not be negative


def get_closest_object(
    world_state: Mapping[str, Any],
    object_id: str,
    object_position_at_time: List[float],
    timestep: str,
) -> str:
    min_distance = float("inf")
    closest_object = None

    for obj_id, obj_data in world_state["objects"].items():
        if obj_id == object_id:
            continue
        obj_position = get_position(world_state, obj_id, timestep)
        if obj_position is None:
            continue
        distance = np.linalg.norm(
            np.array(object_position_at_time) - np.array(obj_position)
        )
        if distance < min_distance:
            min_distance = distance
            closest_object = obj_data
        
    if closest_object is None:
        raise ImpossibleToAnswer("No other visbile objects found in the scene.")

    return closest_object


def get_spatial_relationship_camera_view(obj_1_state, obj_2_state, camera):
    """
    Calculates the positional relationship of O1 relative to O2 from the camera's view.

    Uses a hybrid system:
    - Vertical (Above/Below) is relative to the World Up Axis (Z_world).
    - Horizontal (Left/Right) is relative to the Camera's X_cam axis.
    - Depth (Closer/Further) is relative to the Camera's Z_cam axis.
    """

    rx, ry, rz = camera["at"]
    q = R.from_euler("xyz", [rx, ry, rz], degrees=True)
    R_world_to_cam = q.as_matrix().T  # Transpose to invert rotation

    np_pos1_world = np.array(obj_1_state["obb"]["center"])
    np_pos2_world = np.array(obj_2_state["obb"]["center"])

    # 1. Calculate the Vector from O2 to O1 in WORLD Coordinates
    V_rel_world = np_pos1_world - np_pos2_world

    # 2. World-Relative Vertical (Above/Below)
    # Uses the designated WORLD_UP_AXIS_NUM (e.g., V_rel_world[2] for Z-Up)
    world_vertical_component = V_rel_world[WORLD_UP_AXIS_NUM]
    vertical_adj = []
    if world_vertical_component > obj_1_state["obb"]["extents"][WORLD_UP_AXIS_NUM] * 2:
        vertical_adj.append("Above")
    elif (
        world_vertical_component < -obj_1_state["obb"]["extents"][WORLD_UP_AXIS_NUM] * 2
    ):
        vertical_adj.append("Below")
    else:
        vertical_adj.append("Vertically Aligned")

    # 3. Transform V_rel from World to Camera Coordinates
    V_rel_cam = R_world_to_cam @ V_rel_world

    # 4. Camera-Relative Horizontal (Left/Right) and Depth (Closer/Further)

    # Horizontal (Camera X, V_rel_cam[0])
    horizontal_adj = []
    if V_rel_cam[0] > 0.05:
        horizontal_adj.append("to the Right")
    elif V_rel_cam[0] < -0.05:
        horizontal_adj.append("to the Left")
    else:
        horizontal_adj.append("Horizontally Aligned")

    # Depth (Camera Z, V_rel_cam[2])
    # Z-axis in camera space is the distance from the camera.
    if V_rel_cam[2] > 0.05:
        depth_adj = ["behind"]
    elif V_rel_cam[2] < -0.05:
        depth_adj = ["In front"]
    else:
        depth_adj = ["Equidistant"]

    return horizontal_adj[0].lower(), vertical_adj[0].lower(), depth_adj[0].lower()


def get_all_relational_positional_adjectives():
    horizontals = ["to the Left", "to the Right", "Horizontally Aligned"]
    verticals = ["above", "below", "vertically aligned"]
    depths = ["in front", "behind", "equidistant"]
    total = horizontals + verticals + depths
    return [adj.lower() for adj in total]
