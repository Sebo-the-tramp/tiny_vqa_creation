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
import numpy as np

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
    get_timestep_from_idx,
)

from utils.frames_selection import (
    sample_frames_before_timestep,
)

from .mechanics_helpers import (
    get_speed,
    get_acceleration,
    get_position,
    get_rotation,
    get_mask_collisions, 
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
def F_KINEMATICS_SPEED_OBJECT(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    """Return the velocity of the object referenced in the question."""

    assert len(attributes) == 1 and "OBJECT" in attributes

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        attributes, world_state, min_objects=kwargs["current_world_number_of_objects"]
    )

    timestep = get_random_timestep_from_list(visible_timesteps, question)

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, timestep
    )

    object_id = resolved_attributes["OBJECT"]["choice"]["id"]

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

    assert len(attributes) == 1 and "OBJECT" in attributes

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        attributes, world_state, min_objects=kwargs["current_world_number_of_objects"]
    )
    # if we are in a multi-image setting, we need to ensure there are enough frames
    timestep = get_random_timestep_from_list(visible_timesteps, question)

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, timestep
    )

    object_id = resolved_attributes["OBJECT"]["choice"]["id"]

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
    assert len(attributes) == 1 and "OBJECT" in attributes

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

    object_id = resolved_attributes["OBJECT"]["choice"]["id"]

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
def F_KINEMATICS_MOVING_OBJECT(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    assert len(attributes) == 0

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        attributes, world_state, min_objects=kwargs["current_world_number_of_objects"]
    )

    continuous_subsequences = get_continuous_subsequences_min_length(
        visible_timesteps, min_length=CLIP_LENGTH * FRAME_INTERLEAVE
    )

    visible_timesteps = random.choice(continuous_subsequences)

    timestep = get_random_timestep_from_list(visible_timesteps, question)

    resolved_attributes = resolve_attributes_visible_at_timestep(
        ["OBJECT"], world_state, timestep
    )

    object_id = resolved_attributes["OBJECT"]["choice"]["id"]

    index_timestep = visible_timesteps.index(timestep)
    list_of_position = []
    list_of_rotation = []
    for i in range((CLIP_LENGTH - 1), -1, -1):
        current_timestep = visible_timesteps[index_timestep - i]
        position = get_position(world_state, object_id, current_timestep)
        rotation = get_rotation(world_state, object_id, current_timestep)
        list_of_position.append(position)
        list_of_rotation.append(rotation)

    is_moving = False
    for i in range(1, len(list_of_position)):
        dist = distance_between(list_of_position[i - 1], list_of_position[i])
        if dist > MOVEMENT_TOLERANCE:
            is_moving = True
            break
        rot_diff = sum(
            abs(list_of_rotation[i - 1][j] - list_of_rotation[i][j]) for j in range(3)
        )
        if rot_diff > ROTATION_TOLERANCE:
            is_moving = True
            break

    # should return also the correct index, but we chose later based on is_moving
    labels, _ = create_mc_object_names_from_dataset(
        resolved_attributes["OBJECT"]["choice"]["name"],
        ["No Object"],
        get_all_objects_names(),
    )

    if is_moving:
        correct_idx = labels.index(
            resolved_attributes["OBJECT"]["choice"]["name"].lower()
        )
    else:
        correct_idx = labels.index(
            "no object"
        )  # this version is correct because is lowercase

    return fill_questions(
        question, labels, correct_idx, world_state, timestep, resolved_attributes
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
    if is_unstable:
        # removing the last 3 frames to avoid picking a stable frame
        timestep = get_random_timestep_from_list(
            visible_timesteps[: -(CLIP_LENGTH * FRAME_INTERLEAVE)], question
        )
    else:
        # we want to pick the first of the series for which the last frame is the actual last.
        timestep = visible_timesteps[-(CLIP_LENGTH * FRAME_INTERLEAVE)]

    resolved_attributes = resolve_attributes_visible_at_timestep(
        ["OBJECT"], world_state, timestep
    )

    options = [
        "Stable: The system has stopped",
        "Unstable: The system is currently moving",
        "Cyclic: The system has returned to its exact starting position",
        "Invisible: The objects have moved out of the frame entirely",
    ]

    correct_idx = 1 if is_unstable else 0
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
        attributes, world_state, min_objects=kwargs["current_world_number_of_objects"]
    )

    continuous_subsequences = get_continuous_subsequences_min_length(
        visible_timesteps, min_length=CLIP_LENGTH * FRAME_INTERLEAVE
    )

    visible_timesteps = random.choice(continuous_subsequences)

    timestep = get_random_timestep_from_list(visible_timesteps, question)

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, timestep
    )

    object_id = resolved_attributes["OBJECT"]["choice"]["id"]

    index_timestep = visible_timesteps.index(timestep)
    list_of_position = []
    for i in range((CLIP_LENGTH - 1), -1, -1):
        current_timestep = visible_timesteps[index_timestep - i]
        speed = get_position(world_state, object_id, current_timestep)
        list_of_position.append(speed)

    is_still = True
    for i in range(1, len(list_of_position)):
        dist = distance_between(list_of_position[i - 1], list_of_position[i])
        if dist > MOVEMENT_TOLERANCE:
            is_still = False
            break

    options = ["yes", "no"]
    correct_idx = 0 if is_still else 1
    labels = options

    return fill_questions(
        question, labels, correct_idx, world_state, timestep, resolved_attributes
    )


## --- COLLISION RESOLVERS --- ##


@with_resolved_attributes
def F_COLLISIONS_OBJ_OBJ_FIRST(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    
    if kwargs["current_world_number_of_objects"] < 2:
        raise ImpossibleToAnswer("Not enough objects in the scene for a collision to happen.")

    assert len(attributes) == 1 and "OBJECT" in attributes

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        attributes, world_state, min_objects=kwargs["current_world_number_of_objects"]
    )

    # cause here we do not keep into account FRAME_INTERLEAVE THAT WILL BE CRUCIAL FOR LATER...
    continuous_subsequences = get_continuous_subsequences_min_length(
        visible_timesteps, min_length=CLIP_LENGTH
    )

    visible_timesteps = random.choice(continuous_subsequences)[(CLIP_LENGTH - 1) :]

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, visible_timesteps[0]
    )

    object = resolved_attributes["OBJECT"]["choice"]

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
                    first_collided_object = world_state["objects"][str(obj_b)]
                else:
                    first_collided_object = world_state["objects"][str(obj_a)]
                break

    DATASET = get_all_objects_names()
    present = [
        obj["name"]
        for obj in list(iter_objects(world_state))
        if obj["id"] != object["id"]
    ]

    if first_collided_object is not None:
        labels, correct_idx = create_mc_object_names_from_dataset(
            first_collided_object["name"], present, DATASET
        )
    else:
        labels, correct_idx = create_mc_object_names_from_dataset(
            "No Object", present, DATASET
        )

    return fill_questions(
        question,
        labels,
        correct_idx,
        world_state,
        visible_timesteps[-1],
        resolved_attributes,
        visible_timesteps[0],
    )


@with_resolved_attributes
def F_COLLISION_OBJECT_OBJECT_FRAME_SINGLE(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    
    if kwargs["current_world_number_of_objects"] < 2:
        raise ImpossibleToAnswer("Not enough objects in the scene for a collision to happen.")
    
    assert len(attributes) == 1 and "OBJECT" in attributes

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        attributes, world_state, min_objects=kwargs["current_world_number_of_objects"]
    )    
    
    collision_mask = get_mask_collisions(world_state)

    visible_timesteps_index = [world_state['simulation'][i]['frame_idx'] for i in visible_timesteps]
    visible_collision_mask = collision_mask[visible_timesteps_index[0]: visible_timesteps_index[-1]+1, 1:, 1:]

    rows = visible_collision_mask.any(axis=(1,2))
    t_first = np.argmax(rows) if rows.any() else None

    if t_first is not None:
        row_first = visible_collision_mask[t_first]
        idx = np.nonzero(row_first)[0]        # indices of non-zeros
        
        if idx.size > 2:
            raise ImpossibleToAnswer("Too many collisions at the first collision timestep.")
        
        # adding +1 to match the object_id in the simulation file
        collision_object_a_id = idx[0] + 1 if len(idx) > 0 else None
        collision_object_b_id = idx[1] + 1 if len(idx) > 1 else None
        collision_timestep = get_timestep_from_idx(visible_timesteps_index[t_first])

    else:
        raise ImpossibleToAnswer("No collision found in the visible timesteps.")

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


@with_resolved_attributes
def F_COLLISION_OBJECT_OBJECT_FRAME_MULTI(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:

    if kwargs["current_world_number_of_objects"] < 2:
        raise ImpossibleToAnswer("Not enough objects in the scene for a collision to happen.")

    assert len(attributes) == 1 and "OBJECT" in attributes

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        attributes, world_state, min_objects=kwargs["current_world_number_of_objects"]
    )
    
    collision_mask = get_mask_collisions(world_state)

    visible_timesteps_index = [world_state['simulation'][i]['frame_idx'] for i in visible_timesteps]
    visible_collision_mask = collision_mask[visible_timesteps_index[0]: visible_timesteps_index[-1]+1, 1:, 1:]

    rows = visible_collision_mask.any(axis=(1,2))
    t_first = np.argmax(rows) if rows.any() else None

    if t_first is not None:
        row_first = visible_collision_mask[t_first]
        idx = np.nonzero(row_first)[0]        # indices of non-zeros
        
        if idx.size > 2:
            raise ImpossibleToAnswer("Too many collisions at the first collision timestep.")

        # adding +1 to match the object_id in the simulation file
        collision_object_a_id = idx[0] + 1 if len(idx) > 0 else None
        collision_object_b_id = idx[1] + 1 if len(idx) > 1 else None
        collision_timestep = get_timestep_from_idx(visible_timesteps_index[t_first])

    else:
        raise ImpossibleToAnswer("No collision found in the visible timesteps.")
    
    # technically the resolved object should be the one colliding
    collider_object = world_state["objects"][str(collision_object_b_id)]
    # colliding_object = world_state["objects"][str(collision_object_a_id)]
    resolved_attributes = {"OBJECT": {"choice": collider_object, "category": "OBJECT"}}


    frames_og = sample_frames_before_timestep(
        world_state, collision_timestep, num_frames=4, frame_interleave=2
    )

    # Generate one shared permutation
    indices = list(range(len(frames_og)))
    random.shuffle(indices)

    labels = frames_og.copy()

    # Apply same shuffle to both
    frames = [frames_og[i] for i in indices]
    labels = [labels[i] for i in indices]

    correct_idx = labels.index(frames[3])

    fill_template(question, resolved_attributes)

    # only for this time because of the multi-frame choice nature of the question
    question["question"] = question["question"].replace(
        "Consider all frames, but answer only based on the last frame. ", ""
    )

    return [[question, labels, correct_idx, frames, world_state, resolved_attributes]]


# assumption that the object is not colliding at the start, falling and the colliding with the scene
@with_resolved_attributes
def F_COLLISION_OBJECT_SCENE_FRAME_MULTI(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    assert len(attributes) == 1 and "OBJECT" in attributes

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        attributes, world_state, min_objects=kwargs["current_world_number_of_objects"]
    )

    collision_mask = get_mask_collisions(world_state)

    visible_timesteps_index = [world_state['simulation'][i]['frame_idx'] for i in visible_timesteps]
    visible_collision_mask = collision_mask[visible_timesteps_index[0]: visible_timesteps_index[-1]+1, 0, 1:]

    rows = visible_collision_mask.any(axis=(1))
    t_first = np.argmax(rows) if rows.any() else None

    if t_first is not None:        
        collision_timestep = get_timestep_from_idx(visible_timesteps_index[t_first])
        collision_row = collision_mask[t_first, 0, 1:]

        collision_idx = np.nonzero(collision_row)[0]
        if collision_idx.size == 0:
            raise ImpossibleToAnswer("No collision object found.")
        if collision_idx.size > 1:
            raise ImpossibleToAnswer("Multiple collision objects found.")        

        collision_object_id = collision_idx[0] + 1 # to match object_id in the simulation        

    else:
        raise ImpossibleToAnswer("No collision found in the visible timesteps.")

    frames_og = sample_frames_before_timestep(
        world_state, collision_timestep, num_frames=4, frame_interleave=2
    )

    # technically the resolved object should be the one colliding
    resolved_attributes = {
        "OBJECT": {
            "choice": world_state["objects"][str(collision_object_id)],
            "category": "OBJECT",
        }
    }

    # Create the labels copy
    labels = frames_og.copy()

    # Generate one shared permutation
    indices = list(range(len(frames_og)))
    random.shuffle(indices)

    # Apply same shuffle to both
    labels = [labels[i] for i in indices]

    correct_idx = labels.index(frames_og[3])

    fill_template(question, resolved_attributes)

    # only for this time because of the multi-frame choice nature of the question
    question["question"] = question["question"].replace(
        "Consider all frames, but answer only based on the last frame. ", ""
    )

    return [[question, labels, correct_idx, [], world_state, resolved_attributes]]
