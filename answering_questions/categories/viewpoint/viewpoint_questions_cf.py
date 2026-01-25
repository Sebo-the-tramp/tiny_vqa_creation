"""
Mock visibility reasoning resolvers.

These helpers extract best-effort visibility answers from the provided world state.
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
from utils.helpers import fill_questions_cf
from utils.decorators import with_resolved_attributes_cf
from utils.bin_creation import create_mc_object_names_from_dataset

from utils.my_exception import ImpossibleToAnswer

from categories.viewpoint.viewpoint_helpers import get_number_of_visible_objects

from utils.helpers_cf import (
    get_start_end_timesteps_visible_end    
)

Number = Union[int, float]
Vector = Tuple[float, float, float]
WorldState = Mapping[str, Any]
QuestionPayload = Mapping[str, Any]
Answer = Union[str, float, Vector, Mapping[str, Any], Sequence[str]]

CLIP_LENGTH = get_config()["clip_length"]
FRAME_INTERLEAVE = get_config()["frame_interleave"]
MIN_VISIBLE_PIXELS = get_config()["min_pixels_visible"]
TIMESTART = get_config()["timestart"]
SAMPLING_RATE = get_config()["sampling_rate"]
RENDER_STEP = 1.0 / SAMPLING_RATE


@with_resolved_attributes_cf
def CF_VISIBILITY_OBJECT_COUNT(
    world_state_og: WorldState,
    world_state_mod: WorldState,    
    question: QuestionPayload,
    attributes,
    resolved_attributes,
    **kwargs,
) -> int:        

    assert len(attributes) == 0

    counterfactual_object_id = kwargs['object_moved_id']

    # checking if is just visibility at the end timestep
    # cause we have both multi and single object movement counterfactuals
    # multi will check start visibility after else no question will be generated
    _, timestep_end = \
        get_start_end_timesteps_visible_end(world_state_og, world_state_mod, counterfactual_object_id)

    total_visible_objects_og = get_number_of_visible_objects(world_state_og, timestep_end)
    total_visible_objects_mod = get_number_of_visible_objects(world_state_mod, timestep_end)

    # we can only answer if the number of object diminishes else -> impossible for images
    # else we should change the mod to og images
    if total_visible_objects_mod >= total_visible_objects_og:
        raise ImpossibleToAnswer("8 No change in number of visible objects between original and modified world.")

    # balanced options around the initial count
    start = max(0, total_visible_objects_mod - 2)
    shift = abs(total_visible_objects_mod - 2) if total_visible_objects_mod < 2 else 0
    balanced_bins = [
        str(i)
        for i in range(start, total_visible_objects_mod + 2 + shift)
        if i != total_visible_objects_mod
    ]

    labels, correct_idx = create_mc_object_names_from_dataset(
        str(total_visible_objects_mod),
        [],
        balanced_bins,
        num_answers=4,
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

def get_occlusion_percentage(
    world_state: WorldState,
    object_id: str,
    final_timestep: int,
) -> int:
    

    infov_pixels_visible = world_state['simulation'][final_timestep]['objects'][object_id]['infov_pixels_visible']
    infov_pixels_void = world_state['simulation'][final_timestep]['objects'][object_id]['infov_pixels_void']
    infov_pixels = world_state['simulation'][final_timestep]['objects'][object_id]['infov_pixels']

    if infov_pixels_visible < MIN_VISIBLE_PIXELS:
        raise ImpossibleToAnswer("Object is too visible for occlusion question.")
    
    if infov_pixels_void > 0.1 * infov_pixels:
        raise ImpossibleToAnswer("Object is too visible for occlusion question.")

    visibility_object = (infov_pixels_visible + infov_pixels_void) / infov_pixels    

    if visibility_object < 0.25:
        correct_idx = 0
    elif visibility_object < 0.65:
        correct_idx = 1
    elif visibility_object < 0.95:
        correct_idx = 2
    else:
        correct_idx = 3

    labels = [
        "Severely Occluded (0-25% visible)",  # Hard: Requires context/guessing
        "Partially Occluded (25-65% visible)",  # Medium: Major parts missing
        "Slightly Occluded (65-95% visible)",  # Easy: Minor obstructions
        "Fully Visible (>95% visible)",  # Control: Clean object
    ]

    return labels[correct_idx], labels, correct_idx

@with_resolved_attributes_cf
def CF_OCCLUSION_PERCENTAGE_OBJECT(
    world_state_og: WorldState,
    world_state_mod: WorldState,    
    question: QuestionPayload,
    attributes,
    resolved_attributes,
    **kwargs,
) -> int:
        
    assert len(attributes) == 1 and "OBJECT" in attributes        

    # here actually instead of counterfactual_object_id we should check per all object maybe the counterfactual moved 
    # goes in front of another object that was occluding the target object

    answer_string_og = asnwer_string_mod = ""
    chosen_object_id = None

    for object_id in world_state_og["objects"].keys():

        # checking if is just visibility at the end timestep
        # cause we have both multi and single object movement counterfactuals
        # multi will check start visibility after else no question will be generated    

        _, timestep_end = \
        get_start_end_timesteps_visible_end(world_state_og, world_state_mod, object_id)

        answer_string_og, _, _ = get_occlusion_percentage(
            world_state_og,
            object_id,
            timestep_end,
        )  # just to check if we can answer the question

        asnwer_string_mod, labels, correct_idx = get_occlusion_percentage(
            world_state_mod,
            object_id,
            timestep_end,
        )  # just to check if we can answer the question

        if answer_string_og != asnwer_string_mod:
            chosen_object_id = object_id
            break

    if answer_string_og == asnwer_string_mod:
        raise ImpossibleToAnswer("No change in occlusion percentage between original and modified world.")

    resolved_attributes["OBJECT"] = {"choice": world_state_mod["objects"][chosen_object_id], "category": "OBJECT"}     

    return fill_questions_cf(
        question,
        labels,
        correct_idx,
        world_state_og,
        world_state_mod,
        timestep_end,
        resolved_attributes,
    )