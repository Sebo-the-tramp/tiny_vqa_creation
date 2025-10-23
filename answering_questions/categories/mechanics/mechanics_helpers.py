from __future__ import annotations

import random

import numpy as np

from typing import Any, Mapping, Optional, Tuple, Union, List

from utils.helpers import as_vector

# set random seed for reproducibility
rng = random.Random(42)

Number = Union[int, float]
WorldState = Mapping[str, Any]
QuestionPayload = Mapping[str, Any]
Answer = Union[int, float, str]

from utils.helpers import fill_template

from utils.frames_selection import (
    sample_frames_at_timesteps,
    sample_frames_before_timestep,
)


MOVEMENT_TOLERANCE = 1e-3


## --- Helper functions --- ##


def fill_questions(
    question, labels, correct_idx, world_state, timestep, resolved_attributes
) -> List:
    questions = []
    if "single" in question["task_splits"]:
        question_copy = question.copy()
        question_copy["task_splits"] = "single"  # ensure the question knows it's
        fill_template(question_copy, resolved_attributes)
        questions.append(
            [
                question_copy,
                labels,
                correct_idx,
                sample_frames_at_timesteps(world_state, [timestep]),
            ]
        )
    if "multi" in question["task_splits"]:
        question_copy = question.copy()
        question_copy["task_splits"] = "multi"  # ensure the question knows it's
        fill_template(question_copy, resolved_attributes)
        questions.append(
            [
                question_copy,
                labels,
                correct_idx,
                sample_frames_before_timestep(
                    world_state, timestep, num_frames=8, frame_interleave=1
                ),
            ]
        )

    return questions


def get_position(
    world_state: Mapping[str, Any], object_id: str, timestep: str
) -> Optional[Tuple[float, ...]]:
    timestep_world = world_state["simulation"][timestep]
    current_timestep_involved_object = timestep_world["objects"][object_id]["obb"][
        "center"
    ]
    return as_vector(current_timestep_involved_object)


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
    current_timestep_involved_object_velocity = timestep_world["objects"][object_id][
        "kinematics"
    ]["linear_accel_world"]

    acceleration_magnitude = (
        current_timestep_involved_object_velocity[0] ** 2
        + current_timestep_involved_object_velocity[1] ** 2
        + current_timestep_involved_object_velocity[2] ** 2
    ) ** 0.5

    return acceleration_magnitude
