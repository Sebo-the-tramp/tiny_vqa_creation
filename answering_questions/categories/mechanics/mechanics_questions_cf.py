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
    is_object_visible,
    get_visibility_mask,
    get_timestep_from_idx,
)

from utils.all_objects import get_all_objects_names

from .mechanics_helpers import (
    get_speed,
    get_position,
    get_acceleration,
    get_mask_collisions,
    get_present_and_far_from_collision,
)

from utils.bin_creation import (
    uniform_labels,
    create_mc_options_around_gt,
    create_mc_object_names_from_dataset,
)

from utils.helpers_cf import (
    get_start_end_timesteps_visible_end    
)

import numpy as np

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


def get_distance_moved_between_timesteps(
    world_state: WorldState,
    object_id: str,
    timestep_start: int,
    timestep_end: int,
) -> float:
    """Calculate the distance moved by an object between two timesteps."""
    position_obj_state_timestep_start = get_position(
        world_state, object_id, timestep_start
    )
    position_obj_state_timestep_end = get_position(
        world_state, object_id, timestep_end
    )
    distance_moved = distance_between(
        position_obj_state_timestep_start, position_obj_state_timestep_end
    )
    return distance_moved



@with_resolved_attributes_cf
def CF_KINEMATICS_SPEED_OBJECT(
    world_state_og: WorldState,
    world_state_mod: WorldState,    
    question: QuestionPayload,
    attributes,
    resolved_attributes,
    **kwargs,
) -> int:
    """Count objects of a specific type that moved more than a given metric distance."""

    assert len(attributes) == 1 and "OBJECT" in attributes

    counterfactual_object_id = kwargs['object_moved_id']    

    # checking if is just visibility at the end timestep
    # cause we have both multi and single object movement counterfactuals
    # multi will check start visibility after else no question will be generated
    timestep_start, timestep_end = \
        get_start_end_timesteps_visible_end(world_state_og, world_state_mod, counterfactual_object_id, steps=[4])
    
    # we need to check that original time start and end are valid
    is_object_visible_og_start = is_object_visible(
        world_state_og, counterfactual_object_id, timestep_start
    )    

    if not is_object_visible_og_start:
        raise ImpossibleToAnswer("1 - Question refers to an object not visible at the start timestep.")
    
    velocity_object_at_timestep_og = get_speed(counterfactual_object_id, timestep_end, world_state_og)
    velocity_object_at_timestep_mod = get_speed(counterfactual_object_id, timestep_end, world_state_mod)

    if abs(velocity_object_at_timestep_og - velocity_object_at_timestep_mod) < 0.5: # if the difference is less than 0.5 m/s we consider no change
        raise ImpossibleToAnswer("7 - No significant change in speed between original and modified simulations.")        

    options, correct_idx = create_mc_options_around_gt(
        velocity_object_at_timestep_mod, num_answers=4, display_decimals=2, lo=0.0
    )
    labels = uniform_labels(options, integer=False, decimals=1)
    labels = [f"{opt} m/s" for opt in labels]

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


@with_resolved_attributes_cf
def CF_KINEMATICS_ACCEL_OBJECT(
    world_state_og: WorldState,
    world_state_mod: WorldState,    
    question: QuestionPayload,
    attributes,
    resolved_attributes,
    **kwargs,
) -> int:
    """Count objects of a specific type that moved more than a given metric distance."""

    assert len(attributes) == 1 and "OBJECT" in attributes

    counterfactual_object_id = kwargs['object_moved_id']    

    # checking if is just visibility at the end timestep
    # cause we have both multi and single object movement counterfactuals
    # multi will check start visibility after else no question will be generated
    timestep_start, timestep_end = \
        get_start_end_timesteps_visible_end(world_state_og, world_state_mod, counterfactual_object_id, steps=[4])
    
    # we need to check that original time start and end are valid
    is_object_visible_og_start = is_object_visible(
        world_state_og, counterfactual_object_id, timestep_start
    )    

    if not is_object_visible_og_start:
        raise ImpossibleToAnswer("1 - Question refers to an object not visible at the start timestep.")
    
    acceleration_object_og = get_acceleration(counterfactual_object_id, timestep_end, world_state_og)
    acceleration_object_mod = get_acceleration(counterfactual_object_id, timestep_end, world_state_mod)

    if abs(acceleration_object_og - acceleration_object_mod) < 0.5: # if the difference is less than 0.5 m/s^2 we consider no change
        raise ImpossibleToAnswer("7 - No significant change in acceleration between original and modified simulations.")        

    options, correct_idx = create_mc_options_around_gt(
        acceleration_object_mod, num_answers=4, display_decimals=2, lo=0.0, min_threshold=0.0
    )
    labels = uniform_labels(options, integer=False, decimals=2)

    labels = [f"{label} m/s^2" for label in labels]

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


# I don't like this as a question...
@with_resolved_attributes_cf
def CF_KINEMATICS_DISTANCE_TRAVELED_INTERVAL(
    world_state_og: WorldState,
    world_state_mod: WorldState,    
    question: QuestionPayload,
    attributes,
    resolved_attributes,
    **kwargs,
) -> int:
    """Count objects of a specific type that moved more than a given metric distance."""

    assert len(attributes) == 1 and "OBJECT" in attributes

    counterfactual_object_id = kwargs['object_moved_id']    

    # checking if is just visibility at the end timestep
    # cause we have both multi and single object movement counterfactuals
    # multi will check start visibility after else no question will be generated
    timestep_start, timestep_end = \
        get_start_end_timesteps_visible_end(world_state_og, world_state_mod, counterfactual_object_id, steps=[4])
    
    # we need to check that original time start and end are valid
    is_object_visible_og_start = is_object_visible(
        world_state_og, counterfactual_object_id, timestep_start
    )    

    if not is_object_visible_og_start:
        raise ImpossibleToAnswer("1 - Question refers to an object not visible at the start timestep.")


    distance_moved_og = get_distance_moved_between_timesteps(
        world_state_og,
        counterfactual_object_id,
        timestep_start,
        timestep_end,
    )

    distance_moved_mod = get_distance_moved_between_timesteps(
        world_state_mod,
        counterfactual_object_id,
        timestep_start,
        timestep_end,
    )

    if abs(distance_moved_og - distance_moved_mod) < 0.2: # if the difference is less than 0.5 meters we consider no change
        raise ImpossibleToAnswer("7 - No significant change in distance traveled between original and modified simulations.")        

    options, correct_idx = create_mc_options_around_gt(
        distance_moved_mod, num_answers=4, display_decimals=1, lo=0.0
    )
    labels = uniform_labels(options, integer=False, decimals=1)
    labels = [f"{opt} meters" for opt in labels]    

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
def CF_COLLISION_OBJECT_OBJECT_FRAME_SINGLE(
    world_state_og: WorldState,
    world_state_mod: WorldState,    
    question: QuestionPayload,
    attributes,
    resolved_attributes,
    **kwargs,
) -> int:
    assert len(attributes) == 1 and "OBJECT" in attributes

    if kwargs["current_world_number_of_objects"] < 2:
        raise ImpossibleToAnswer(
            "Not enough objects in the scene to answer the question."
        )
    
    counterfactual_object_id = kwargs['object_moved_id']    

    # We need to search in the world_mod if the object has been colliding with something 

    collision_mask = get_mask_collisions(world_state_mod)
    visibility_mask, _ = get_visibility_mask(world_state_mod)

    # adding ground visibility (always visible)
    visibility_mask_T = np.append(
        np.ones((1, visibility_mask.shape[1]), dtype=visibility_mask.dtype),
        visibility_mask,
        axis=0,
    ).T
    visibility_mask_T_extended = (
        visibility_mask_T[:, :, None] * visibility_mask_T[:, None, :]
    )

    # collision_mask AND visibility_mask
    # the collision needs to be visible so
    visible_collision_mask = collision_mask * visibility_mask_T_extended

    # I chose the row with counterfactual_object_id -> -1 objects starts from one but array starts from 0
    object_id_int = int(counterfactual_object_id)
    # we need to find the first timestep where there is a collision involving that object
    rows = visible_collision_mask[:, object_id_int, 1:].any(axis=1)
    t_first = np.argmax(rows) if rows.any() else None

    if t_first is not None:
        row_first = visible_collision_mask[t_first, 1:, 1:]
        idx = np.nonzero(row_first)[0]  # indices of non-zeros

        if idx.size > 2:
            raise ImpossibleToAnswer(
                "4 - Too many collisions at the first collision timestep."
            )

        # adding +1 to match the object_id in the simulation file
        collision_object_a_id = idx[0] + 1 if len(idx) > 0 else None
        collision_object_b_id = idx[1] + 1 if len(idx) > 1 else None
        collision_timestep = get_timestep_from_idx(int(t_first))

    else:
        raise ImpossibleToAnswer("6 - No collision found in the visible timesteps.")


    # technically the resolved object should be the one colliding
    collider_object = world_state_mod["objects"][str(collision_object_b_id)]
    colliding_object = world_state_mod["objects"][str(collision_object_a_id)]

    if collision_object_a_id == object_id_int:
        # swap objects so that colliding_object is the one that was moved
        colliding_object = world_state_mod["objects"][str(collision_object_b_id)]
        collider_object = world_state_mod["objects"][str(collision_object_a_id)]

    # I need only to do one more check that in the original sim they were not colliding at that timestep
    if t_first < len(world_state_og["simulation"]):        
        collisions_timestep_obj_og = world_state_og["simulation"][collision_timestep]["collisions"]
        for collision in collisions_timestep_obj_og:
            if (collision_object_a_id in collision) and (collision_object_b_id in collision):
                raise ImpossibleToAnswer("7 - No change in collision between original and modified simulations.")    
    else:
        raise ImpossibleToAnswer("3 - Collision timestep outside of original simulation range.")    

    present_and_far_from_collision, present_and_close_to_collision = (
        get_present_and_far_from_collision(
            world_state_mod, collision_timestep, int(collider_object["id"])
        )
    )

    other_objects_minus_present_and_close = [
        obj_name
        for obj_name in get_all_objects_names()
        if obj_name not in present_and_close_to_collision
        and obj_name != colliding_object["name"]
        and obj_name != collider_object["name"]
    ]

    present_and_far_from_collision_minus_collider = [
        obj_name for obj_name in present_and_far_from_collision if obj_name != collider_object["name"] 
        and obj_name != colliding_object["name"]
    ]

    labels, correct_idx = create_mc_object_names_from_dataset(
        colliding_object["name"],
        present_and_far_from_collision_minus_collider,
        other_objects_minus_present_and_close,
    )

    return fill_questions_cf(
        question,
        labels,
        correct_idx,
        world_state_og,
        world_state_mod,
        collision_timestep,
        resolved_attributes,
    )