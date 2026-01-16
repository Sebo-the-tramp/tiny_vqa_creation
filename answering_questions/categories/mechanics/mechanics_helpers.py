from __future__ import annotations


import numpy as np

from typing import Any, Mapping, Optional, Tuple, Union

from utils.helpers import as_vector, iter_objects, minimum_distance_between_OBBs
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


# def get_mask_collisions_score(world_state: Mapping[str, Any]) -> Optional[np.ndarray]:
#     timestep_length = len(world_state["simulation"])
#     n_objects = len(world_state["objects"])

#     mask = np.zeros((timestep_length, n_objects + 1, n_objects + 1), dtype=np.uint8)

#     # basically this mask is a directed symmetric graph for which object is touching which other object
#     # we could even just use the upper part, but is faster to use the full matrix
#     for t, ts in enumerate(world_state["simulation"].values()):
#         pairs = np.asarray(ts["collisions"], dtype=np.uint8)  # shape (M, 2)
#         if pairs.shape[0] != 0:
#             for object_id in ts["objects"].keys():
#                 object_collisions = ts["objects"][object_id]['collisions']
#                 for other_object_id in object_collisions.keys():
#                     collision_point_count = len(object_collisions[other_object_id]['points'])

#                     collision = 1 if collision_point_count > 5 else 0

#                     mask[t, int(object_id), int(other_object_id)] = collision
#                     mask[t, int(other_object_id), int(object_id)] = collision  # symmetric

#     return mask


def get_present_and_far_from_collision(
    world_state: WorldState, timestep: str, collision_object_a_id: int
) -> list:
    """
    Get list of objects that are present and not colliding with the given object at the specified timestep.
    """

    present_and_far_from_collision = []
    present_and_close_to_collision = []

    for object in iter_objects(world_state):
        object_id = int(object["id"])

        distance = minimum_distance_between_OBBs(
            world_state["simulation"][timestep]["objects"][str(object_id)]["obb"],
            world_state["simulation"][timestep]["objects"][str(collision_object_a_id)][
                "obb"
            ],
        )
        if distance > 0.5:  # Threshold distance to consider "far from collision"
            present_and_far_from_collision.append(object["name"])
        else:
            present_and_close_to_collision.append(object["name"])

    return present_and_far_from_collision, present_and_close_to_collision
