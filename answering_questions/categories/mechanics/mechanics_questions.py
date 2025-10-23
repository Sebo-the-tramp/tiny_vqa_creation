"""
Mock spatial reasoning resolvers.

These helpers extract best-effort spatial answers from the provided world state.
They operate on lightweight metadata (positions, orientations, region tags, etc.)
and fall back to sensible defaults when information is missing.
"""

from __future__ import annotations

from utils.decorators import with_resolved_attributes

from typing import (
    Any,
    Mapping,
    Sequence,
    Tuple,
    Union,
)

import random

random.seed(42)

from utils.my_exception import ImpossibleToAnswer

from utils.all_objects import get_all_objects_names

from utils.helpers import (
    iter_objects,
    distance_between,
    resolve_attributes_visible_at_timestep,
    get_visible_timesteps_for_attributes_min_objects,
    get_continuous_subsequences_min_length,
)

from .mechanics_helpers import get_speed, fill_questions, get_acceleration, get_position

from utils.config import get_config

from utils.bin_creation import create_mc_options_around_gt, uniform_labels

Number = Union[int, float]
Vector = Tuple[float, float, float]
WorldState = Mapping[str, Any]
QuestionPayload = Mapping[str, Any]
Answer = Union[str, float, Vector, Mapping[str, Any], Sequence[str]]

CLIP_LENGTH = get_config()["clip_length"]
MOVEMENT_TOLERANCE = get_config()["movement_tolerance"]

## --- Resolver functions -- ##


@with_resolved_attributes
def F_KINEMATICS_SPEED_OBJECT(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    """Return the velocity of the object referenced in the question."""

    assert len(attributes) == 1 and "OBJECT" in attributes

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        attributes, world_state, min_objects=1
    )
    # if we are in a multi-image setting, we need to ensure there are enough frames
    if len(visible_timesteps) == 0:
        raise ImpossibleToAnswer("No timestep with both objects visible.")

    timestep = random.choice(visible_timesteps[(CLIP_LENGTH - 1):])

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, timestep
    )

    object_id = resolved_attributes["OBJECT"]["choice"]["id"]

    velocity_object_at_timestep = get_speed(object_id, timestep, world_state)

    labels, correct_idx = create_mc_options_around_gt(
        velocity_object_at_timestep, num_answers=4, min_rel_gap=0.3
    )
    labels = [f"{label} m/s" for label in labels]

    return fill_questions(
        question, labels, correct_idx, world_state, timestep, resolved_attributes
    )


@with_resolved_attributes
def F_KINEMATICS_FASTEST_OBJECT_SPEED(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    """Return the velocity of the object referenced in the question."""

    assert len(attributes) == 0

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        attributes, world_state, min_objects=1
    )
    # if we are in a multi-image setting, we need to ensure there are enough frames
    if len(visible_timesteps) == 0:
        raise ImpossibleToAnswer("No timestep with both objects visible.")

    timestep = random.choice(visible_timesteps[(CLIP_LENGTH - 1):])

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, timestep
    )

    highest_speed = -1.0
    for object in iter_objects(world_state):
        object_id = object["id"]
        speed = get_speed(object_id, timestep, world_state)
        if speed > highest_speed:
            highest_speed = speed

    labels, correct_idx = create_mc_options_around_gt(
        highest_speed, num_answers=4, min_rel_gap=0.3
    )
    labels = [f"{label} m/s" for label in labels]

    return fill_questions(
        question, labels, correct_idx, world_state, timestep, resolved_attributes
    )


@with_resolved_attributes
def F_KINEMATICS_ACCEL_OBJECT(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    """Return the velocity of the object referenced in the question."""

    assert len(attributes) == 1 and "OBJECT" in attributes

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        attributes, world_state, min_objects=1
    )
    # if we are in a multi-image setting, we need to ensure there are enough frames
    if len(visible_timesteps) == 0:
        raise ImpossibleToAnswer("No timestep with both objects visible.")

    timestep = random.choice(visible_timesteps[(CLIP_LENGTH - 1):])

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, timestep
    )

    object_id = resolved_attributes["OBJECT"]["choice"]["id"]

    acceleration_object = get_acceleration(object_id, timestep, world_state)

    labels, correct_idx = create_mc_options_around_gt(
        acceleration_object, num_answers=4, min_rel_gap=0.3
    )
    labels = [f"{label} m/s^2" for label in labels]

    return fill_questions(
        question, labels, correct_idx, world_state, timestep, resolved_attributes
    )


@with_resolved_attributes
def F_KINEMATICS_DISTANCE_TRAVELED_INTERVAL(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    """Count objects of a specific type that moved more than a given metric distance."""
    assert len(attributes) == 1 and "OBJECT" in attributes

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        attributes, world_state, min_objects=1
    )

    continuous_subsequences = get_continuous_subsequences_min_length(
        visible_timesteps, min_length=CLIP_LENGTH
    )

    print("continuous_subsequences:", continuous_subsequences)

    visible_timesteps = random.choice(continuous_subsequences)

    # if we are in a multi-image setting, we need to ensure there are enough frames
    if len(visible_timesteps) == 0:
        raise ImpossibleToAnswer("No timestep with both objects visible.")    

    timestep_end = random.choice(visible_timesteps[(CLIP_LENGTH - 1):])
    timestep_start = visible_timesteps[visible_timesteps.index(timestep_end) - (CLIP_LENGTH - 1)]

    print( visible_timesteps)
    print( visible_timesteps.index(timestep_end))
    print( visible_timesteps[visible_timesteps.index(timestep_end) - (CLIP_LENGTH - 1)])

    print(f"Timestep start: {timestep_start}, Timestep end: {timestep_end}")

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, timestep_start
    )

    object_id = resolved_attributes["OBJECT"]["choice"]["id"]

    position_obj_state_timestep_start = get_position(
        world_state, object_id, timestep_start
    )
    position_obj_state_timestep_end = get_position(
        world_state, object_id, timestep_end
    )
    distance = distance_between(
        position_obj_state_timestep_start, position_obj_state_timestep_end
    )    
    
    options, correct_idx = create_mc_options_around_gt(
        distance, num_answers=4, display_decimals=1, lo=0.0, min_rel_gap=0.2
    )
    labels = uniform_labels(options, integer=False, decimals=1)
    labels = [f"{opt} meters" for opt in labels]

    return fill_questions(
        question, labels, correct_idx, world_state, timestep_end, resolved_attributes
    )


@with_resolved_attributes
def F_KINEMATICS_MOVING_OBJECT(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    """Return the velocity of the object referenced in the question."""

    assert len(attributes) == 1 and "OBJECT" in attributes

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        attributes, world_state, min_objects=1
    )
    
    continuous_subsequences = get_continuous_subsequences_min_length(
        visible_timesteps, min_length=CLIP_LENGTH
    )

    visible_timesteps = random.choice(continuous_subsequences)

    # if we are in a multi-image setting, we need to ensure there are enough frames
    if len(visible_timesteps) == 0:
        raise ImpossibleToAnswer("No timestep with both objects visible.")    

    timestep = random.choice(visible_timesteps[(CLIP_LENGTH - 1):])

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, timestep
    )

    object_id = resolved_attributes["OBJECT"]["choice"]["id"]

    index_timestep = visible_timesteps.index(timestep)
    list_of_position = []
    for i in range((CLIP_LENGTH-1), -1, -1):
        current_timestep = visible_timesteps[index_timestep - i]        
        speed = get_position(world_state, object_id, current_timestep)
        list_of_position.append(speed)

    is_moving = False
    for i in range(1, len(list_of_position)):
        dist = distance_between(list_of_position[i-1], list_of_position[i])
        if dist > MOVEMENT_TOLERANCE:
            is_moving = True
            break

    options = ["yes", "no"]
    correct_idx = 0 if is_moving else 1
    labels = options    

    return fill_questions(
        question, labels, correct_idx, world_state, timestep, resolved_attributes
    )


@with_resolved_attributes
def F_KINEMATICS_STILL_OBJECT(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    """Return the velocity of the object referenced in the question."""

    assert len(attributes) == 1 and "OBJECT" in attributes

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        attributes, world_state, min_objects=1
    )
    
    continuous_subsequences = get_continuous_subsequences_min_length(
        visible_timesteps, min_length=CLIP_LENGTH
    )

    visible_timesteps = random.choice(continuous_subsequences)

    # if we are in a multi-image setting, we need to ensure there are enough frames
    if len(visible_timesteps) == 0:
        raise ImpossibleToAnswer("No timestep with both objects visible.")    

    timestep = random.choice(visible_timesteps[(CLIP_LENGTH - 1):])

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, timestep
    )

    object_id = resolved_attributes["OBJECT"]["choice"]["id"]

    index_timestep = visible_timesteps.index(timestep)
    list_of_position = []
    for i in range((CLIP_LENGTH-1), -1, -1):
        current_timestep = visible_timesteps[index_timestep - i]        
        speed = get_position(world_state, object_id, current_timestep)
        list_of_position.append(speed)

    is_still = True
    for i in range(1, len(list_of_position)):
        dist = distance_between(list_of_position[i-1], list_of_position[i])
        if dist > MOVEMENT_TOLERANCE:
            is_still = False
            break

    options = ["yes", "no"]
    correct_idx = 0 if is_still else 1
    labels = options

    return fill_questions(
        question, labels, correct_idx, world_state, timestep, resolved_attributes
    )