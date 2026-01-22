"""
Mock spatial reasoning resolvers.

These helpers extract best-effort spatial answers from the provided world state.
They operate on lightweight metadata (positions, orientations, region tags, etc.)
and fall back to sensible defaults when information is missing.
"""

from __future__ import annotations

from typing import (
    Any,
    Mapping,
    Sequence,
    Tuple,
    Union,
)

from utils.config import get_config
from utils.my_exception import ImpossibleToAnswer
from utils.decorators import with_resolved_attributes_cf

from utils.helpers import (
    distance_between,
    fill_questions_cf,
    get_timestep_from_idx,
    get_objects_present_and_not_present,
    resolve_attributes_visible_at_timestep,
)

from .mechanics_helpers import (
    get_speed,
    get_position,
    get_acceleration,
)

from utils.bin_creation import (
    uniform_labels,
    create_mc_options_around_gt,
    create_mc_object_names_from_dataset,
)

Vector = Tuple[float, float, float]
WorldState = Mapping[str, Any]
QuestionPayload = Mapping[str, Any]

Number = Union[int, float]
Answer = Union[str, float, Vector, Mapping[str, Any], Sequence[str]]

TIMESTART = get_config()["timestart"]
CLIP_LENGTH = get_config()["clip_length"]
SAMPLING_RATE = get_config()["sampling_rate"]
FRAME_INTERLEAVE = get_config()["frame_interleave"]
MOVEMENT_TOLERANCE = get_config()["movement_tolerance"]
ROTATION_TOLERANCE = get_config()["rotation_tolerance"]
RENDER_STEP = 1.0 / SAMPLING_RATE


## --- Resolver functions -- ##
@with_resolved_attributes_cf
def CF_KINEMATICS_SPEED_OBJECT(
    world_state_og: WorldState,
    world_state_mod: WorldState,
    answer_list_original_data_cf: dict,
    question: QuestionPayload,
    attributes,
    **kwargs,
) -> int:
    """
    Return the velocity of the object referenced in the question."""

    assert len(attributes) == 1 and "OBJECT" in attributes

    timestep_end_index = int(
        answer_list_original_data_cf[0][3][-1]
    )  # this has to be the image to get the question
    timestep_end = get_timestep_from_idx(timestep_end_index)

    # check if the particular question asks something outside of simulation_og
    if timestep_end_index > len(world_state_og["simulation"]) - 1:
        raise ImpossibleToAnswer("Question refers to future timestep.")

    object_id = answer_list_original_data_cf[0][5]["OBJECT"]["choice"]["id"]

    velocity_object_at_timestep = get_speed(object_id, timestep_end, world_state_og)

    labels, correct_idx = create_mc_options_around_gt(
        velocity_object_at_timestep, num_answers=4
    )
    labels = [f"{label} m/s" for label in labels]

    resolved_attributes = resolve_attributes_visible_at_timestep(
        ["OBJECT"], world_state_og, timestep_end
    )

    return fill_questions_cf(
        question,
        labels,
        correct_idx,
        world_state_og,
        world_state_mod,
        timestep_end,
        resolved_attributes,
    )


@with_resolved_attributes_cf
def CF_KINEMATICS_ACCEL_OBJECT(
    world_state_og: WorldState,
    world_state_mod: WorldState,
    answer_list_original_data_cf: dict,
    question: QuestionPayload,
    attributes,
    **kwargs,
) -> int:
    """Return the velocity of the object referenced in the question."""

    assert len(attributes) == 1 and "OBJECT" in attributes

    timestep_end_index = int(
        answer_list_original_data_cf[0][3][-1]
    )  # this has to be the image to get the question
    timestep_end = get_timestep_from_idx(timestep_end_index)

    # check if the particular question asks something outside of simulation_og
    if timestep_end_index > len(world_state_og["simulation"]) - 1:
        raise ImpossibleToAnswer("Question refers to future timestep.")

    object_id = answer_list_original_data_cf[0][5]["OBJECT"]["choice"]["id"]

    acceleration_object = get_acceleration(object_id, timestep_end, world_state_og)

    labels, correct_idx = create_mc_options_around_gt(
        acceleration_object, num_answers=4, display_decimals=2, lo=-100.0, hi=100.0
    )
    labels = [f"{label} m/s^2" for label in labels]

    resolved_attributes = resolve_attributes_visible_at_timestep(
        ["OBJECT"], world_state_og, timestep_end
    )

    return fill_questions_cf(
        question,
        labels,
        correct_idx,
        world_state_og,
        world_state_mod,
        timestep_end,
        resolved_attributes,
    )


@with_resolved_attributes_cf
def CF_KINEMATICS_DISTANCE_TRAVELED_INTERVAL(
    world_state_og: WorldState,
    world_state_mod: WorldState,
    answer_list_original_data_cf: dict,
    question: QuestionPayload,
    attributes,
    **kwargs,
) -> int:
    """Count objects of a specific type that moved more than a given metric distance."""

    assert len(attributes) == 1 and "OBJECT" in attributes

    timestep_end_index = int(answer_list_original_data_cf[0][3][-1])
    timestep_end = get_timestep_from_idx(timestep_end_index)

    # check if the particular question asks something outside of simulation_og
    if timestep_end_index > len(world_state_og["simulation"]) - 1:
        raise ImpossibleToAnswer("Question refers to future timestep.")

    timestep_start_index = answer_list_original_data_cf[0][3][0]
    timestep_start = get_timestep_from_idx(timestep_start_index)

    object_id = answer_list_original_data_cf[0][5]["OBJECT"]["choice"]["id"]

    position_obj_state_timestep_start = get_position(
        world_state_og, object_id, timestep_start
    )
    position_obj_state_timestep_end = get_position(
        world_state_og, object_id, timestep_end
    )
    distance = distance_between(
        position_obj_state_timestep_start, position_obj_state_timestep_end
    )

    options, correct_idx = create_mc_options_around_gt(
        distance, num_answers=4, display_decimals=1, lo=0.0
    )
    labels = uniform_labels(options, integer=False, decimals=1)
    labels = [f"{opt} meters" for opt in labels]

    resolved_attributes = resolve_attributes_visible_at_timestep(
        ["OBJECT"], world_state_og, timestep_start
    )

    return fill_questions_cf(
        question,
        labels,
        correct_idx,
        world_state_og,
        world_state_mod,
        timestep_end,
        resolved_attributes,
        initial_timestep=timestep_start,
    )


## --- COLLISION RESOLVERS --- ##
@with_resolved_attributes_cf
def CF_COLLISION_OBJECT_OBJECT_FRAME_SINGLE_WRONG(
    world_state_og: WorldState,
    world_state_mod: WorldState,
    answer_list_original_data_cf: dict,
    question: QuestionPayload,
    attributes,
    **kwargs,
) -> int:
    assert len(attributes) == 1 and "OBJECT" in attributes

    timestep_end_index = int(
        answer_list_original_data_cf[0][3][-1]
    )  # this has to be the image to get the question
    timestep_end = get_timestep_from_idx(timestep_end_index)

    # check if the particular question asks something outside of simulation_og
    if timestep_end_index > len(world_state_og["simulation"]) - 1:
        raise ImpossibleToAnswer("Question refers to future timestep.")

    object_id = answer_list_original_data_cf[0][5]["OBJECT"]["choice"]["id"]

    collisions_timestep_obj = world_state_og["simulation"][timestep_end]["collisions"]

    obj_colliding = None
    for collision in collisions_timestep_obj:
        if object_id in collision:
            other_object_id = (
                collision[1] if collision[0] == object_id else collision[0]
            )
            if other_object_id != 0:
                obj_colliding = other_object_id
            break    

    if obj_colliding is not None:
        correct_name = world_state_og["objects"][obj_colliding]["name"]

        visible_objects_names_minus_resolved, all_objects_minus_visible_and_non_visible = (
            get_objects_present_and_not_present(
                world_state_mod, timestep_end, [correct_name]
            )
        )

        labels, correct_idx = create_mc_object_names_from_dataset(
            correct_name,
            visible_objects_names_minus_resolved,
            all_objects_minus_visible_and_non_visible,
        )
    else:
        visible_objects_names_minus_resolved, all_objects_minus_visible_and_non_visible = (
            get_objects_present_and_not_present(
                world_state_mod, timestep_end, []
            )
        )
        labels, correct_idx = create_mc_object_names_from_dataset(
            "No Object",
            visible_objects_names_minus_resolved,
            all_objects_minus_visible_and_non_visible,
        )

    resolved_attributes = resolve_attributes_visible_at_timestep(
        ["OBJECT"], world_state_og, timestep_end
    )

    return fill_questions_cf(
        question,
        labels,
        correct_idx,
        world_state_og,
        world_state_mod,
        timestep_end,
        resolved_attributes,
    )



## --- COLLISION RESOLVERS --- ##
@with_resolved_attributes_cf
def CF_COLLISION_OBJECT_OBJECT_FRAME_SINGLE(
    world_state_og: WorldState,
    world_state_mod: WorldState,
    answer_list_original_data_cf: dict,
    question: QuestionPayload,
    attributes,
    **kwargs,
) -> int:
    assert len(attributes) == 1 and "OBJECT" in attributes

    timestep_end_index = int(
        answer_list_original_data_cf[0][3][-1]
    )  # this has to be the image to get the question
    timestep_end = get_timestep_from_idx(timestep_end_index)

    # check if the particular question asks something outside of simulation_og
    if timestep_end_index > len(world_state_og["simulation"]) - 1:
        raise ImpossibleToAnswer("Question refers to future timestep.")

    object_id = answer_list_original_data_cf[0][5]["OBJECT"]["choice"]["id"]

    collisions_timestep_obj = world_state_og["simulation"][timestep_end]["collisions"]

    is_object_colliding = any(int(object_id) in pair for pair in collisions_timestep_obj)

    if is_object_colliding:
        raise ImpossibleToAnswer(
            "Nothing interesting, the object still colliding at the final timestep." 
        )

    visible_objects_names_minus_resolved, all_objects_minus_visible_and_non_visible = (
        get_objects_present_and_not_present(
            world_state_mod, timestep_end, []
        )
    )
    labels, correct_idx = create_mc_object_names_from_dataset(
        "No Object",
        visible_objects_names_minus_resolved,
        all_objects_minus_visible_and_non_visible,
    )

    resolved_attributes = resolve_attributes_visible_at_timestep(
        ["OBJECT"], world_state_og, timestep_end
    )

    return fill_questions_cf(
        question,
        labels,
        correct_idx,
        world_state_og,
        world_state_mod,
        timestep_end,
        resolved_attributes,
    )