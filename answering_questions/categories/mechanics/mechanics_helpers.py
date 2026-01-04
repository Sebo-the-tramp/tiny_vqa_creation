from __future__ import annotations


import numpy as np

from typing import Any, Mapping, Optional, Tuple, Union

from utils.helpers import as_vector
from utils.config import get_config

from scipy.spatial.transform import Rotation as R

Number = Union[int, float]
WorldState = Mapping[str, Any]
QuestionPayload = Mapping[str, Any]
Answer = Union[int, float, str]

MOVEMENT_TOLERANCE = get_config()["movement_tolerance"]
FRAME_INTERLEAVE = get_config()["frame_interleave"]
CLIP_LENGTH = get_config()["clip_length"]

## --- Helper functions --- ##

def get_position(
    world_state: Mapping[str, Any], object_id: str, timestep: str
) -> Optional[Tuple[float, ...]]:
    timestep_world = world_state["simulation"][timestep]
    current_timestep_involved_object = timestep_world["objects"][object_id]["obb"][
        "center"
    ]
    return as_vector(current_timestep_involved_object)


def get_rotation(
    world_state: Mapping[str, Any], object_id: str, timestep: str
) -> Optional[Tuple[float, ...]]:
    timestep_world = world_state["simulation"][timestep]
    current_timestep_involved_object = timestep_world["objects"][object_id]["obb"]["R"]
    R_mat = np.array(current_timestep_involved_object)

    # Re-orthogonalize via SVD to enforce det=+1
    U, _, Vt = np.linalg.svd(R_mat)
    R_fixed = U @ Vt
    if np.linalg.det(R_fixed) < 0:  # handle left-handed reflections
        U[:, -1] *= -1
        R_fixed = U @ Vt

    return R.from_matrix(R_fixed).as_euler("xyz", degrees=True)


def is_moving(object_id: str, timestep: str, world_state: Mapping[str, Any]) -> bool:
    return get_speed(object_id, timestep, world_state) > MOVEMENT_TOLERANCE


def get_speed(object_id: str, timestep: str, world_state: Mapping[str, Any]) -> float:
    timestep_world = world_state["simulation"][timestep]
    current_timestep_involved_object = timestep_world["objects"][object_id][
        "kinematics"
    ]["speed"]
    return current_timestep_involved_object


def get_acceleration(
    object_id: str, timestep: str, world_state: Mapping[str, Any]
) -> float:
    timestep_world = world_state["simulation"][timestep]

    # this should work with kinematics_ver_2
    current_timestep_involved_object_velocity = timestep_world["objects"][object_id][
        "kinematics"
    ]

    return current_timestep_involved_object_velocity["accel"]
