from __future__ import annotations

import random

import numpy as np

from typing import (
    Any,
    Mapping,
    Optional,
    Tuple,
    Union,
)

from utils.helpers import _as_vector

# set random seed for reproducibility
rng = random.Random(42)

Number = Union[int, float]
WorldState = Mapping[str, Any]
QuestionPayload = Mapping[str, Any]
Answer = Union[int, float, str]


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


# _____________________ SPATIAL HELPERS _____________________


def _get_position(
    world_state: Mapping[str, Any], object_id: str, timestep: str
) -> Optional[Tuple[float, ...]]:
    timestep_world = world_state["simulation"][timestep]
    current_timestep_involved_object = timestep_world["objects"][object_id]["obb"][
        "center"
    ]
    return _as_vector(current_timestep_involved_object)


def _get_position_camera(
    world_state: Mapping[str, Any], timestep: str
) -> Optional[Tuple[float, ...]]:
    timestep_world = world_state["simulation"][timestep]
    current_timestep_involved_object = timestep_world["camera"]["eye"]
    return _as_vector(current_timestep_involved_object)


def get_max_height_from_obb(obb: Mapping[str, Any]) -> float:
    """
    Given an oriented bounding box (obb), return the maximum height (y coordinate) of the object.
    """
    center = np.array(obb["center"])
    extents = np.array(obb["extents"])
    axes = np.array(obb["R"])  # 3x3 rotation matrix
    up = np.array([0.0, 0.0, 1.0])

    # Fast path: choose the sign of each extent by the up-dot for each axis
    signs = np.sign(axes.T @ up)  # shape (3,)
    p_high = center + axes @ (signs * extents)

    return p_high[2]  # return the z coordinate since z is up
