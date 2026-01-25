"""
Mock spatial reasoning resolvers.

These helpers extract best-effort spatial answers from the provided world state.
They operate on lightweight metadata (positions, orientations, region tags, etc.)
and fall back to sensible defaults when information is missing.
"""

from __future__ import annotations

import random

from utils.decorators import with_resolved_attributes_cf

from typing import (
    Any,
    Mapping,
    Sequence,
    Tuple,
    Union,
)

from utils.my_exception import ImpossibleToAnswer
from utils.config import get_config

from utils.geometry import get_camera_OBB

from utils.helpers import (
    iter_objects,
    fill_questions_cf,
    is_object_visible,    
    minimum_distance_between_OBBs,    
    get_objects_present_and_not_present,    
)
from .spatial_reasoning_helpers import (    
    get_closest_visible_object,
    get_spatial_relationship_camera_view,
    get_all_relational_positional_adjectives,
)
from utils.bin_creation import (    
    create_mc_options_around_gt,
    create_mc_object_names_from_dataset,
)

from utils.helpers_cf import (
    get_start_end_timesteps_visible_end    
)

Number = Union[int, float]
Vector = Tuple[float, float, float]
WorldState = Mapping[str, Any]
QuestionPayload = Mapping[str, Any]
Answer = Union[str, float, Vector, Mapping[str, Any], Sequence[str]]


VISIBILITY_THRESHOLD = get_config()["visibility_threshold"]
FRAME_INTERLEAVE = get_config()["frame_interleave"]
CLIP_LENGTH = get_config()["clip_length"]
MIN_VISIBLE_PIXELS = get_config()["min_pixels_visible"]
TIMESTART = get_config()["timestart"]
SAMPLING_RATE = get_config()["sampling_rate"]
RENDER_STEP = 1.0 / SAMPLING_RATE


## --- Resolver functions -- ##
## Assumptions: ##
# - All object positions are given by their OBB center
# - Distances are Euclidean distances between object centers unless specified otherwise
# - The valid timesteps are those where all the  objects are visible above VISIBILITY_THRESHOLD


@with_resolved_attributes_cf
def CF_CLOSEST_OBJECT_OBJECT(
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

    # checking if is just visibility at the end timestep
    # cause we have both multi and single object movement counterfactuals
    # multi will check start visibility after else no question will be generated
    _, timestep_end = \
        get_start_end_timesteps_visible_end(world_state_og, world_state_mod, counterfactual_object_id)

    closest_object_og = get_closest_visible_object(world_state_og, counterfactual_object_id, timestep_end)
    closest_object_mod = get_closest_visible_object(world_state_mod, counterfactual_object_id, timestep_end)

    if closest_object_og["id"] == closest_object_mod["id"]:
        return []  # No change in closest object
    
    visible_objects_names_minus_resolved, all_objects_minus_visible_and_non_visible = (
        get_objects_present_and_not_present(
            world_state_mod, timestep_end, [closest_object_mod["name"]]
        )
    )

    labels, correct_idx = create_mc_object_names_from_dataset(
        closest_object_mod["name"],
        visible_objects_names_minus_resolved,
        all_objects_minus_visible_and_non_visible,
        num_answers=4,
    )

    labels = [str(label) for label in labels]

    return fill_questions_cf(
        question,
        labels,
        correct_idx,
        world_state_og,
        world_state_mod,
        timestep_end,
        resolved_attributes,
    )


def get_closest_object_camera(world_state, final_timestep):
    min_distance = float("inf")
    closest_object = None

    camera_state = world_state["simulation"][final_timestep]["camera"]
    camera_OBB = get_camera_OBB(camera_state)

    for obj in iter_objects(world_state):
        obj_id = obj["id"]

        if not is_object_visible(world_state, obj_id, final_timestep):
            continue

        obj_state = world_state["simulation"][final_timestep]["objects"][obj_id]
        object_OBB = obj_state["obb"]

        distance = minimum_distance_between_OBBs(object_OBB, camera_OBB)

        if distance < min_distance:
            min_distance = distance
            closest_object = obj

    if closest_object is None:
        raise ImpossibleToAnswer("1 - No other visbile objects found in the scene.")

    return closest_object

@with_resolved_attributes_cf
def CF_CLOSEST_OBJECT_CAMERA(
    world_state_og: WorldState,
    world_state_mod: WorldState,    
    question: QuestionPayload,
    attributes,
    resolved_attributes,
    **kwargs,
) -> int:    
    """Question: Which object in the image is the closest to the camera?"""
    assert len(attributes) == 0

    if kwargs["current_world_number_of_objects"] < 2:
        raise ImpossibleToAnswer(
            "Not enough objects in the scene to answer the question."
        )
    
    counterfactual_object_id = kwargs['object_moved_id']

    # checking if is just visibility at the end timestep
    # cause we have both multi and single object movement counterfactuals
    # multi will check start visibility after else no question will be generated
    _, timestep_end = \
        get_start_end_timesteps_visible_end(world_state_og, world_state_mod, counterfactual_object_id)

    closest_object_og_camera = get_closest_object_camera(world_state_og, timestep_end)
    closest_object_mod_camera = get_closest_object_camera(world_state_mod, timestep_end)

    if closest_object_og_camera["id"] == closest_object_mod_camera["id"]:
        return []  # No change in closest object
    
    visible_objects_names_minus_resolved, all_objects_minus_visible_and_non_visible = (
        get_objects_present_and_not_present(
            world_state_mod, timestep_end, [closest_object_mod_camera["name"]]
        )
    )

    labels, correct_idx = create_mc_object_names_from_dataset(
        closest_object_mod_camera["name"],
        visible_objects_names_minus_resolved,
        all_objects_minus_visible_and_non_visible,
        num_answers=4,
    )

    labels = [str(label) for label in labels]    

    return fill_questions_cf(
        question,
        labels,
        correct_idx,
        world_state_og,
        world_state_mod,
        timestep_end,
        resolved_attributes,
    )   


def get_extensts_object(world_state, object_id, timestep):
    extents = world_state["simulation"][timestep]["objects"][object_id]["obb"][
        "extents"
    ]
    return extents

@with_resolved_attributes_cf
def CF_SIZE_OBJECT(
    world_state_og: WorldState,
    world_state_mod: WorldState,    
    question: QuestionPayload,
    attributes,
    resolved_attributes,
    **kwargs,
) -> int:
    assert len(attributes) == 1 and "OBJECT" in attributes   
    
    counterfactual_object_id = kwargs['object_moved_id']

    # checking if is just visibility at the end timestep
    # cause we have both multi and single object movement counterfactuals
    # multi will check start visibility after else no question will be generated
    _, timestep_end = \
        get_start_end_timesteps_visible_end(world_state_og, world_state_mod, counterfactual_object_id)

    
    extents_obj_mod = get_extensts_object(world_state_mod, counterfactual_object_id, timestep_end)
    extents_obj_og = get_extensts_object(world_state_og, counterfactual_object_id, timestep_end)

    # if the extents are the same no size change
    if extents_obj_mod == extents_obj_og:
        raise ImpossibleToAnswer("No size change detected.")    

    first_extent = extents_obj_mod[0]

    options, correct_idx = create_mc_options_around_gt(
        first_extent,
        num_answers=4,
        display_decimals=2,
    )

    # we need to make better options per extents
    # we also downsize by (^1/3) to account for the volume scaling to keep
    # all the confounding linearly distant
    scales = [(float(option) / first_extent) ** (1 / 3) for option in options]

    labels = [
        f"{extents_obj_mod[0] * scale:.2f}m x {extents_obj_mod[1] * scale:.2f}m x {extents_obj_mod[2] * scale:.2f}m"
        for scale in scales
    ]    

    return fill_questions_cf(
        question,
        labels,
        correct_idx,
        world_state_og,
        world_state_mod,
        timestep_end,
        resolved_attributes,
    )


def get_biggest_object_by_volume(world_state, timestep):
    biggest_object = None
    biggest_volume = -1.0
    for obj in iter_objects(world_state):
        volume = obj.get("volume", 0.0)

        if volume > biggest_volume and is_object_visible(
            world_state, obj["id"], timestep
        ):
            biggest_volume = volume
            biggest_object = obj

    if biggest_object is None:
        raise ImpossibleToAnswer("1 - No visbile objects found in the scene.")

    return biggest_object


@with_resolved_attributes_cf
def CF_SIZE_OBJECT_BIGGER(
    world_state_og: WorldState,
    world_state_mod: WorldState,    
    question: QuestionPayload,
    attributes,
    resolved_attributes,
    **kwargs,
) -> str:
    assert len(attributes) == 0
    
    if kwargs["current_world_number_of_objects"] < 2:
        raise ImpossibleToAnswer(
            "Not enough objects in the scene to answer the question."
        )
    
    counterfactual_object_id = kwargs['object_moved_id']

    # checking if is just visibility at the end timestep
    # cause we have both multi and single object movement counterfactuals
    # multi will check start visibility after else no question will be generated
    _, timestep_end = \
        get_start_end_timesteps_visible_end(world_state_og, world_state_mod, counterfactual_object_id)
        
    biggest_object_og = get_biggest_object_by_volume(world_state_og, timestep_end)
    biggest_object_mod = get_biggest_object_by_volume(world_state_mod, timestep_end)

    # if the extents are the same no size change
    if biggest_object_mod == biggest_object_og:
        raise ImpossibleToAnswer("2 - No size change detected.")
    
    visible_objects_names_minus_resolved, all_objects_minus_visible_and_non_visible = (
        get_objects_present_and_not_present(
            world_state_mod, timestep_end, [biggest_object_mod["name"]]
        )
    )

    labels, correct_idx = create_mc_object_names_from_dataset(
        biggest_object_mod["name"],
        visible_objects_names_minus_resolved,
        all_objects_minus_visible_and_non_visible,
        num_answers=4,
    )

    resolved_attributes["OBJECT"] = {"choice": biggest_object_mod, "category": "OBJECT"}

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
def CF_LAYOUT_POSITION_OBJECT_OBJECT(
    world_state_og: WorldState,
    world_state_mod: WorldState,    
    question: QuestionPayload,    
    attributes,
    resolved_attributes,
    **kwargs,
) -> str:
    assert (
        len(attributes) == 2 and "OBJECT_1" in attributes and "OBJECT_2" in attributes
    )


    if kwargs["current_world_number_of_objects"] < 2:
        raise ImpossibleToAnswer(
            "Not enough objects in the scene to answer the question."
        )
    
    counterfactual_object_id = kwargs['object_moved_id']    

    # checking if is just visibility at the end timestep
    # cause we have both multi and single object movement counterfactuals
    # multi will check start visibility after else no question will be generated
    _, timestep_end = \
        get_start_end_timesteps_visible_end(world_state_og, world_state_mod, counterfactual_object_id)


    #get position of object in world_og and world_mod at timestep_end
    obj_state_og = world_state_og["simulation"][timestep_end]["objects"][counterfactual_object_id]
    obj_state_mod = world_state_mod["simulation"][timestep_end]["objects"][counterfactual_object_id]

    object_reference = None
    horizontal, vertical, depth, max_movement_adj = None, None, None, None

    for obj in iter_objects(world_state_mod):
        obj_id = obj["id"]

        if obj_id == counterfactual_object_id:
            continue

        if not is_object_visible(world_state_mod, obj_id, timestep_end):
            continue

        obj_state = world_state_mod["simulation"][timestep_end]["objects"][obj_id]

        # check visibility in original world
        if not is_object_visible(world_state_og, obj_id, timestep_end) and \
        not is_object_visible(world_state_mod, obj_id, timestep_end):
            continue

        # now I need to see if there has been a change in spatial relationship
        _, _, _, max_movement_adj_og = get_spatial_relationship_camera_view(
            obj_state_og, obj_state, world_state_og["simulation"][timestep_end]["camera"], timestep_end
        )

        horizontal_mod, vertical_mod, depth_mod, max_movement_adj_mod = get_spatial_relationship_camera_view(
            obj_state_mod, obj_state, world_state_mod["simulation"][timestep_end]["camera"], timestep_end
        )

        if (max_movement_adj_og != max_movement_adj_mod):           
            object_reference = obj 
            horizontal = horizontal_mod
            vertical = vertical_mod
            depth = depth_mod
            max_movement_adj = max_movement_adj_mod
            break        

    if object_reference is None:
        raise ImpossibleToAnswer("6 - No change in spatial relationship detected.")            

    DATASET_RELATIONAL_ADJECTIVES = get_all_relational_positional_adjectives()
    # remove correct answers
    if horizontal in DATASET_RELATIONAL_ADJECTIVES:
        DATASET_RELATIONAL_ADJECTIVES.remove(horizontal)
    if vertical in DATASET_RELATIONAL_ADJECTIVES:
        DATASET_RELATIONAL_ADJECTIVES.remove(vertical)
    if depth in DATASET_RELATIONAL_ADJECTIVES:
        DATASET_RELATIONAL_ADJECTIVES.remove(depth)

    # confounding options
    random.shuffle(DATASET_RELATIONAL_ADJECTIVES)
    confounding_options = DATASET_RELATIONAL_ADJECTIVES[:3]

    correct_idx = random.randint(0, 3)
    labels = (
        confounding_options[:correct_idx]
        + [max_movement_adj]
        + confounding_options[correct_idx:]
    )

    # set them manually
    resolved_attributes = {""
        "OBJECT_1": {"choice": object_reference, "category": "OBJECT"},
        "OBJECT_2": {"choice": world_state_mod["objects"][counterfactual_object_id], "category": "OBJECT"},
    }

    return fill_questions_cf(
        question,
        labels,
        correct_idx,
        world_state_og,
        world_state_mod,
        timestep_end,
        resolved_attributes,
    )
