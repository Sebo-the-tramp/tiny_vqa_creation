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
    timestamp_world = world_state["simulation_steps"][timestep]
    current_timestamp_involved_object = [0, 0, 0]
    current_timestamp_involved_object[0] = timestamp_world["objects"][object_id][
        "transform"
    ][0]
    current_timestamp_involved_object[1] = timestamp_world["objects"][object_id][
        "transform"
    ][2]
    current_timestamp_involved_object[2] = timestamp_world["objects"][object_id][
        "transform"
    ][1]
    # TODO change back to correct vecottr format now -> XZY, should be XYZ
    return _as_vector(current_timestamp_involved_object)


def _get_position_camera(
    world_state: Mapping[str, Any], timestep: str
) -> Optional[Tuple[float, ...]]:
    timestamp_world = world_state["simulation_steps"][timestep]
    current_timestamp_involved_object = [0, 0, 0]
    current_timestamp_involved_object[0] = timestamp_world["camera"]["transform"][0]
    current_timestamp_involved_object[1] = timestamp_world["camera"]["transform"][2]
    current_timestamp_involved_object[2] = timestamp_world["camera"]["transform"][1]
    # TODO change back to correct vecottr format now -> XZY, should be XYZ
    return _as_vector(current_timestamp_involved_object)
