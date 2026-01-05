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
    # fix for now
    if current_timestep_involved_object is None:
        return 0.0
    return current_timestep_involved_object


def get_acceleration(
    object_id: str, timestep: str, world_state: Mapping[str, Any]
) -> float:
    timestep_world = world_state["simulation"][timestep]

    # this should work with kinematics_ver_2
    current_timestep_involved_object_velocity = timestep_world["objects"][object_id][
        "kinematics"
    ]["accel"]

    if current_timestep_involved_object_velocity is None:
        return 0.0

    return current_timestep_involved_object_velocity


def get_mask_collisions(world_state: Mapping[str, Any]) -> Optional[np.ndarray]:
    timestep_length = len(world_state["simulation"])
    n_objects = len(world_state["objects"])

    mask = np.zeros((timestep_length, n_objects + 1, n_objects + 1), dtype=np.uint8)

    # basically this mask is a directed symmetric graph for which object is touching which other object
    # we could even just use the upper part, but is faster to use the full matrix
    for t, ts in enumerate(world_state["simulation"].values()):
        pairs = np.asarray(ts["collisions"], dtype=np.uint8)  # shape (M, 2)
        if pairs.shape[0] != 0:
            mask[t, pairs[:, 0], pairs[:, 1]] = 1
            mask[t, pairs[:, 1], pairs[:, 0]] = 1  # symmetric

    return mask
