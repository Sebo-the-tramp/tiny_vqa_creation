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

from utils.my_exception import ImpossibleToAnswer

from utils.all_objects import get_all_objects_names

from utils.helpers import (
    fill_questions,
    iter_objects,
    distance_between,
    resolve_attributes_visible_at_timestep,
    get_visible_timesteps_for_attributes_min_objects,
    get_continuous_subsequences_min_length,
    is_object_visible_at_timestep,
    get_random_timestep_from_list,
    fill_template,
)

from utils.frames_selection import (
    sample_frames_before_timestep,
)

from .mechanics_helpers import (
    get_speed,    
    get_acceleration,
    get_position,
    get_rotation,
)

from utils.config import get_config

from utils.bin_creation import (
    create_mc_options_around_gt,
    create_mc_object_names_from_dataset,
    uniform_labels,
)

Number = Union[int, float]
Vector = Tuple[float, float, float]
WorldState = Mapping[str, Any]
QuestionPayload = Mapping[str, Any]
Answer = Union[str, float, Vector, Mapping[str, Any], Sequence[str]]

CLIP_LENGTH = get_config()["clip_length"]
FRAME_INTERLEAVE = get_config()["frame_interleave"]

MOVEMENT_TOLERANCE = get_config()["movement_tolerance"]
ROTATION_TOLERANCE = get_config()["rotation_tolerance"]

## --- Resolver functions -- ##


@with_resolved_attributes
def CF_KINEMATICS_SPEED_OBJECT(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    """Return the velocity of the object referenced in the question."""

    assert len(attributes) == 1 and "OBJECT-CF" in attributes

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        attributes, world_state, min_objects=kwargs["current_world_number_of_objects"]
    )

    timestep = get_random_timestep_from_list(visible_timesteps, question)

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, timestep
    )

    object_id = resolved_attributes["OBJECT-CF"]["choice"]["id"]

    velocity_object_at_timestep = get_speed(object_id, timestep, world_state)

    labels, correct_idx = create_mc_options_around_gt(
        velocity_object_at_timestep, num_answers=4
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

    assert len(attributes) == 1 and "OBJECT-CF" in attributes

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        attributes, world_state, min_objects=kwargs["current_world_number_of_objects"]
    )
    # if we are in a multi-image setting, we need to ensure there are enough frames
    timestep = get_random_timestep_from_list(visible_timesteps, question)

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, timestep
    )

    object_id = resolved_attributes["OBJECT-CF"]["choice"]["id"]

    acceleration_object = get_acceleration(object_id, timestep, world_state)

    labels, correct_idx = create_mc_options_around_gt(
        acceleration_object, num_answers=4, display_decimals=2, lo=-100.0, hi=100.0
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
    assert len(attributes) == 1 and "OBJECT-CF" in attributes

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        attributes, world_state, min_objects=kwargs["current_world_number_of_objects"]
    )

    continuous_subsequences = get_continuous_subsequences_min_length(
        visible_timesteps, min_length=CLIP_LENGTH * FRAME_INTERLEAVE
    )

    visible_timesteps = random.choice(continuous_subsequences)

    timestep_end = get_random_timestep_from_list(visible_timesteps, question)
    timestep_start = visible_timesteps[
        visible_timesteps.index(timestep_end)
        - ((CLIP_LENGTH * FRAME_INTERLEAVE) - FRAME_INTERLEAVE)
    ]

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, timestep_start
    )

    object_id = resolved_attributes["OBJECT-CF"]["choice"]["id"]

    position_obj_state_timestep_start = get_position(
        world_state, object_id, timestep_start
    )
    position_obj_state_timestep_end = get_position(world_state, object_id, timestep_end)
    distance = distance_between(
        position_obj_state_timestep_start, position_obj_state_timestep_end
    )

    options, correct_idx = create_mc_options_around_gt(
        distance, num_answers=4, display_decimals=1, lo=0.0
    )
    labels = uniform_labels(options, integer=False, decimals=1)
    labels = [f"{opt} meters" for opt in labels]

    return fill_questions(
        question, labels, correct_idx, world_state, timestep_end, resolved_attributes
    )

@with_resolved_attributes
def F_KINEMATICS_SYSTEM_STABILITY(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    
    """
    Stable: The system has stopped. (False)
    Unstable: The system is currently moving. (True)
    Cyclic: The system has returned to its exact starting position. (False - very rare in physics towers)
    Invisible: The objects have moved out of the frame entirely. (False - assuming they are visible)
    """

    assert len(attributes) == 0

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        attributes, world_state, min_objects=kwargs["current_world_number_of_objects"]
    )

    continuous_subsequences = get_continuous_subsequences_min_length(
        visible_timesteps, min_length=CLIP_LENGTH * FRAME_INTERLEAVE
    )

    visible_timesteps = random.choice(continuous_subsequences)

    is_unstable = random.choice([True, False])

    # basically if the system is unstable, just give a random timestep beside the final ones
    # with the assumption that the frame n+1 will always be stable    
    if(is_unstable):
        # removing the last 3 frames to avoid picking a stable frame
        timestep = get_random_timestep_from_list(visible_timesteps[:-(CLIP_LENGTH * FRAME_INTERLEAVE)], question)
    else:
        # we want to pick the first of the series for which the last frame is the actual last.
        timestep = visible_timesteps[-(CLIP_LENGTH * FRAME_INTERLEAVE)]

    resolved_attributes = resolve_attributes_visible_at_timestep(
        ["OBJECT-CF"], world_state, timestep
    )

    options = [
        "Stable: The system has stopped",
        "Unstable: The system is currently moving",
        "Cyclic: The system has returned to its exact starting position",
        "Invisible: The objects have moved out of the frame entirely"
    ]

    correct_idx = 1 if is_unstable else 0
    labels = options

    return fill_questions(
        question, labels, correct_idx, world_state, timestep, resolved_attributes
    )


## --- COLLISION RESOLVERS --- ##

@with_resolved_attributes
def F_COLLISIONS_OBJ_OBJ_FIRST(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    assert len(attributes) == 1 and "OBJECT-CF" in attributes

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        attributes, world_state, min_objects=kwargs["current_world_number_of_objects"]
    )

    continuous_subsequences = get_continuous_subsequences_min_length(
        visible_timesteps, min_length=CLIP_LENGTH
    )

    visible_timesteps = random.choice(continuous_subsequences)[(CLIP_LENGTH - 1) :]

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, visible_timesteps[0]
    )

    object = resolved_attributes["OBJECT-CF"]["choice"]

    first_collided_object = None
    for timestep in visible_timesteps:
        value = world_state["simulation"][str(timestep)]
        collisions_at_sim_step = value["collisions"]
        for collision in collisions_at_sim_step:
            obj_a = collision[0]
            obj_b = collision[1]
            if obj_a == 0 or obj_b == 0:
                continue  # we are just colliding with the ground
            if obj_a == object["id"] or obj_b == object["id"]:
                if obj_a == object["id"]:
                    first_collided_object = world_state[["objects"]][str(obj_b)]
                else:
                    first_collided_object = world_state[["objects"]][str(obj_b)]
                break

    DATASET = get_all_objects_names()
    present = [
        obj["name"]
        for obj in list(iter_objects(world_state))
        if obj["id"] != object["id"]
    ]

    if first_collided_object is not None:
        labels, idx = create_mc_object_names_from_dataset(
            first_collided_object["name"], present, DATASET
        )
    else:
        labels, idx = create_mc_object_names_from_dataset("No Object", present, DATASET)

    return fill_questions(
        question, labels, idx, world_state, visible_timesteps[0], resolved_attributes
    )


@with_resolved_attributes
def F_COLLISION_OBJECT_OBJECT_FRAME_SINGLE(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    assert len(attributes) == 1 and "OBJECT-CF" in attributes

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        attributes, world_state, min_objects=kwargs["current_world_number_of_objects"]
    )

    choice_collision = 1  # forcing to NOT look for a collision

    collision_timestep = None
    non_collision_timestep = []

    # I just want to catch a collision here
    for timestep in visible_timesteps:
        step_state = world_state["simulation"][str(timestep)]
        collisions_at_sim_step = step_state["collisions"]

        collisions_at_sim_step_ground = [
            collision
            for collision in collisions_at_sim_step
            if (collision[0] != 0 and collision[1] != 0)
        ]

        # if we are looking for no collision, and there is none, we are done
        # this though is sampling the first timestep with no collision, not a random one
        if choice_collision == 0 and len(collisions_at_sim_step) == 0:
            non_collision_timestep.append(timestep)
            continue

        collisions_at_sim_visible_object = []
        for collision in collisions_at_sim_step_ground:
            obj_a = collision[0]
            obj_b = collision[1]
            if obj_a != 0 and obj_b != 0:
                if is_object_visible_at_timestep(
                    str(obj_b), str(timestep), world_state
                ) and is_object_visible_at_timestep(
                    str(obj_a), str(timestep), world_state
                ):
                    collisions_at_sim_visible_object.append(collision)

        if len(collisions_at_sim_visible_object) > 0:
            collision_timestep = timestep
            collision_objects = collisions_at_sim_visible_object
            break

    if collision_timestep is None:
        raise ImpossibleToAnswer("No collision found in the visible timesteps.")

    collision_between_obj_a_b = random.choice(collision_objects)

    collision_object_a_id = collision_between_obj_a_b[0]
    collision_object_b_id = collision_between_obj_a_b[1]

    """ How the object looks like:
    {"OBJECT": {'choice': {'model': 'Olive_Kids_Game_On_Pack_n_Snack', 'sim': 'rho-medium_yms-medium_prs-medium', 'props': {...}, 'volume': 0.02960631065070629, 'mass': 1.6283470392227173, 'description': {...}, 'spawning_region': 'above_ground', 
    'initial_condition': {...}, 'scale': 1.2468836307525635, 'obb_size': None, 'id': '2', 'name': 'Olive_Kids_Game_On_Pack_n_Snack'}, 'category': 'OBJECT'}
    """
    # technically the resolved object should be the one colliding
    collider_object = world_state["objects"][str(collision_object_b_id)]
    colliding_object = world_state["objects"][str(collision_object_a_id)]
    resolved_attributes = {"OBJECT": {"choice": collider_object, "category": "OBJECT"}}

    present = [
        obj["name"]
        for obj in list(iter_objects(world_state))
        if obj["id"] != collider_object["id"]
    ]

    labels, correct_idx = create_mc_object_names_from_dataset(
        colliding_object["name"], present, get_all_objects_names()
    )

    return fill_questions(
        question,
        labels,
        correct_idx,
        world_state,
        collision_timestep,
        resolved_attributes,
    )