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
    distance_between,
    resolve_attributes_visible_at_timestep,
    get_visible_timesteps_for_attributes_min_objects,
    get_continuous_subsequences_min_length,
    get_random_timestep_from_list,
    fill_template,
    get_timestep_from_idx,
    is_object_visible_v3,
    get_visibility_mask
)

from utils.frames_selection import (
    sample_frames_before_timestep,
)

from .mechanics_helpers import (
    get_speed,
    get_acceleration,
    get_position,
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
    """
    Question: What is the speed of the <OBJECT> visible in the image?
    """

    assert len(attributes) == 1 and "OBJECT" in attributes

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        attributes, world_state, min_objects=1
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
    """
    Question: What is the magnitude of acceleration of the <OBJECT> in the image?
    """

    assert len(attributes) == 1 and "OBJECT" in attributes

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        attributes, world_state, min_objects=1
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
    """
    Question: Considering the geometrical center of <OBJECT>, what is the straight-line distance it has traveled during the sequence?
    """
    assert len(attributes) == 1 and "OBJECT" in attributes

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        attributes, world_state, min_objects=1
    )

    continuous_subsequences = get_continuous_subsequences_min_length(
        visible_timesteps, min_length=CLIP_LENGTH * FRAME_INTERLEAVE
    )

    visible_timesteps = random.choice(continuous_subsequences)

    final_timestep = get_random_timestep_from_list(visible_timesteps, question)
    final_timestep_index = world_state["simulation"][final_timestep]["frame_idx"]

    candidates = [
        k for k in (1, 2, 3, 4) if final_timestep_index - (k * (CLIP_LENGTH - 1)) >= 0
    ]
    if len(candidates) == 0:
        raise ImpossibleToAnswer("Not enough previous frames to determine visibility.")

    max_k = max(candidates)
    initial_timestep_index = final_timestep_index - (max_k * (CLIP_LENGTH - 1))
    initial_timestep = get_timestep_from_idx(initial_timestep_index)

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, initial_timestep
    )

    object_id = resolved_attributes["OBJECT"]["choice"]["id"]

    if not is_object_visible_v3(world_state, object_id, final_timestep):
        raise ImpossibleToAnswer("The object is not visible at the end timestep.")

    position_obj_state_timestep_start = get_position(
        world_state, object_id, initial_timestep
    )
    position_obj_state_timestep_end = get_position(
        world_state, object_id, final_timestep
    )
    distance = distance_between(
        position_obj_state_timestep_start, position_obj_state_timestep_end
    )

    options, correct_idx = create_mc_options_around_gt(
        distance, num_answers=4, display_decimals=1, lo=0.0
    )
    labels = uniform_labels(options, integer=False, decimals=1)
    labels = [f"{opt} meters" for opt in labels]

    return fill_questions(
        question,
        labels,
        correct_idx,
        world_state,
        final_timestep,
        resolved_attributes,
        initial_timestep=initial_timestep,
    )


@with_resolved_attributes
def F_KINEMATICS_SYSTEM_STABILITY(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    """
    Question: Analyzing the motion trend across the 8-frame sequence, which statement best describes the system's state at the final frame?
    """

    assert len(attributes) == 0

    # check if at least one object is visible in the last frame
    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        attributes, world_state, min_objects=max(kwargs['current_world_number_of_objects']//2,1), remove_last_n_frames=0
    )    

    is_unstable = random.choice([True, False])

    # basically if the system is unstable, just give a random timestep beside the final ones
    # with the assumption that the frame n+1 will always be stable
    if is_unstable:

        all_timesteps = list(set(list(world_state["simulation"].keys())[:-20]) & set(visible_timesteps))
        
        final_timestep = get_random_timestep_from_list(
            all_timesteps[(CLIP_LENGTH - 1) * FRAME_INTERLEAVE: -10], question
        )        
    else:
        final_timesteps = list(set(list(world_state["simulation"].keys())[-20:]) & set(visible_timesteps))
        if len(final_timesteps) == 0:
            raise ImpossibleToAnswer("No visible timesteps found in the last frames.")
    
        final_timestep = final_timesteps[-1]

    resolved_attributes = resolve_attributes_visible_at_timestep(
        [], world_state, final_timestep
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
        question, labels, correct_idx, world_state, final_timestep, resolved_attributes
    )


## --- COLLISION RESOLVERS --- ##


@with_resolved_attributes
def F_COLLISION_OBJECT_OBJECT_FRAME_SINGLE(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    """Question: Which object is the <OBJECT> colliding with in the frame?"""
    if kwargs["current_world_number_of_objects"] < 2:
        raise ImpossibleToAnswer(
            "Not enough objects in the scene for a collision to happen."
        )

    assert len(attributes) == 1 and "OBJECT" in attributes

    collision_mask = get_mask_collisions(world_state)
    visibility_mask, _ = get_visibility_mask(world_state)

    # adding ground visibility (always visible)
    visibility_mask_T = np.append(
        visibility_mask,
        np.ones((1, visibility_mask.shape[1]), dtype=visibility_mask.dtype),
        axis=0,
    ).T
    visibility_mask_T_extended = (
        visibility_mask_T[:, :, None] * visibility_mask_T[:, None, :]
    )

    # collision_mask AND visibility_mask
    # the collision needs to be visible so
    visible_collision_mask = collision_mask * visibility_mask_T_extended

    rows = visible_collision_mask[:, 1:, 1:].any(axis=(1, 2))
    t_first = np.argmax(rows) if rows.any() else None

    if t_first is not None:
        row_first = visible_collision_mask[t_first, 1:, 1:]
        idx = np.nonzero(row_first)[0]  # indices of non-zeros

        if idx.size > 2:
            raise ImpossibleToAnswer(
                "Too many collisions at the first collision timestep."
            )

        # adding +1 to match the object_id in the simulation file
        collision_object_a_id = idx[0] + 1 if len(idx) > 0 else None
        collision_object_b_id = idx[1] + 1 if len(idx) > 1 else None
        collision_timestep = get_timestep_from_idx(int(t_first))

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

    present_and_not_colliding = []

    collision_mask_timestep = collision_mask[t_first]
    for obj_id in range(1, collision_mask_timestep.shape[0]):
        if (
            collision_mask_timestep[int(collider_object["id"]), obj_id] == 0
            and collision_mask_timestep[obj_id, int(collider_object["id"])] == 0
        ):
            obj = world_state["objects"][str(obj_id)]
            if obj["name"] not in present_and_not_colliding:
                present_and_not_colliding.append(obj["name"])

    # present = []
    # for obj in iter_objects(world_state):
    #     if obj["id"] != colliding_object["id"]:
    #         present.append(obj["name"])

    labels, correct_idx = create_mc_object_names_from_dataset(
        colliding_object["name"], present_and_not_colliding, get_all_objects_names()
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
    """Question: In which frame is the <OBJECT> most likely colliding with another object?"""
    if kwargs["current_world_number_of_objects"] < 2:
        raise ImpossibleToAnswer(
            "Not enough objects in the scene for a collision to happen."
        )

    assert len(attributes) == 1 and "OBJECT" in attributes

    collision_mask = get_mask_collisions(world_state)
    visibility_mask, _ = get_visibility_mask(world_state)

    # adding ground visibility (always visible)
    visibility_mask_T = np.append(
        visibility_mask,
        np.ones((1, visibility_mask.shape[1]), dtype=visibility_mask.dtype),
        axis=0,
    ).T
    visibility_mask_T_extended = (
        visibility_mask_T[:, :, None] * visibility_mask_T[:, None, :]
    )

    # collision_mask AND visibility_mask
    # the collision needs to be visible so
    visible_collision_mask = collision_mask * visibility_mask_T_extended

    rows = visible_collision_mask[:, 1:, 1:].any(axis=(1, 2))
    t_first = np.argmax(rows) if rows.any() else None

    if t_first is not None:
        row_first = visible_collision_mask[t_first, 1:, 1:]
        idx = np.nonzero(row_first)[0]  # indices of non-zeros

        if idx.size > 2:
            raise ImpossibleToAnswer(
                "Too many collisions at the first collision timestep."
            )

        # adding +1 to match the object_id in the simulation file
        collision_object_b_id = idx[1] + 1 if len(idx) > 1 else None
        collision_timestep = get_timestep_from_idx(t_first)

    else:
        raise ImpossibleToAnswer("No collision found in the visible timesteps.")

    # technically the resolved object should be the one colliding
    collider_object = world_state["objects"][str(collision_object_b_id)]
    resolved_attributes = {"OBJECT": {"choice": collider_object, "category": "OBJECT"}}

    frames_og = sample_frames_before_timestep(
        world_state, collision_timestep, num_frames=4, frame_interleave=2
    )

    correct_frame = frames_og[3]

    indices = list(range(len(frames_og)))
    random.shuffle(indices)

    frames = [frames_og[i] for i in indices]
    labels = frames.copy()

    correct_idx = labels.index(correct_frame)

    fill_template(question, resolved_attributes)

    # only for this time because of the multi-frame choice nature of the question
    question["question"] = question["question"].replace(
        "Consider all frames, but answer only based on the last frame. ", ""
    )

    # no frames need to be provided as we already have them in the answer choices
    return [[question, labels, correct_idx, [], world_state, resolved_attributes]]


# assumption that the object is not colliding at the start, falling and the colliding with the scene
@with_resolved_attributes
def F_COLLISION_OBJECT_SCENE_FRAME_MULTI(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    """Question: In which frame is the <OBJECT> most likely colliding with the static scene?"""
    assert len(attributes) == 1 and "OBJECT" in attributes

    collision_mask = get_mask_collisions(world_state)
    visibility_mask, _ = get_visibility_mask(world_state)

    # adding ground visibility (always visible)
    visibility_mask_T = np.append(
        visibility_mask,
        np.ones((1, visibility_mask.shape[1]), dtype=visibility_mask.dtype),
        axis=0,
    ).T
    visibility_mask_T_extended = (
        visibility_mask_T[:, :, None] * visibility_mask_T[:, None, :]
    )

    # collision_mask AND visibility_mask
    # the collision needs to be visible so
    visible_collision_mask = collision_mask * visibility_mask_T_extended

    rows = visible_collision_mask[:, 0, 1:].any(axis=(1))
    t_first = np.argmax(rows) if rows.any() else None

    if t_first is not None:
        collision_timestep = get_timestep_from_idx(t_first)
        collision_row = collision_mask[t_first, 0, 1:]

        # we just get one object colliding with the scene
        collision_idx = np.nonzero(collision_row)[0]

        collision_object_id = int(collision_idx[0]) + 1
        # to match object_id in the simulation

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

    correct_frame = frames_og[3]

    indices = list(range(len(frames_og)))
    random.shuffle(indices)

    frames = [frames_og[i] for i in indices]
    labels = frames.copy()

    correct_idx = labels.index(correct_frame)

    fill_template(question, resolved_attributes)

    # only for this time because of the multi-frame choice nature of the question
    question["question"] = question["question"].replace(
        "Consider all frames, but answer only based on the last frame. ", ""
    )

    return [[question, labels, correct_idx, [], world_state, resolved_attributes]]
