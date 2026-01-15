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
from utils.all_objects import get_all_objects_names
from utils.config import get_config
from utils.helpers import (
    iter_objects,
    distance_between,
    fill_questions_cf,
    get_visibility_mask,
    is_object_visible_v3,
    get_random_timestep_from_list,
    resolve_attributes_visible_at_timestep,
    get_visible_timesteps_for_attributes_min_objects,
)
from .spatial_reasoning_helpers import (
    get_position,
    get_closest_object,
    get_position_camera,
    get_spatial_relationship_camera_view,
    get_all_relational_positional_adjectives,
)
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
    answer_list_original_data_cf: dict,
    question: QuestionPayload,
    attributes,
    **kwargs,
) -> int:
    assert len(attributes) == 1 and "OBJECT" in attributes

    if kwargs["current_world_number_of_objects"] < 2:
        raise ImpossibleToAnswer(
            "Not enough objects in the scene to answer the question."
        )

    answer_list_original_data_cf = answer_list_original_data_cf[
        1
    ]  # take always the last (video)

    timestep_end_index = int(
        answer_list_original_data_cf[3][-1]
    )  # this has to be the image to get the question
    timestep_end = f"{TIMESTART + float(timestep_end_index) * RENDER_STEP:08.3f}"

    # check if the particular question asks something outside of simulation_og
    if timestep_end_index > len(world_state_og["simulation"]) - 1:
        raise ImpossibleToAnswer("Question refers to future timestep.")

    object_id = answer_list_original_data_cf[5]["OBJECT"]["choice"]["id"]
    visibility_mask, _ = get_visibility_mask(world_state_og)

    # just check that object_id is visible
    if not visibility_mask[int(object_id) - 1][timestep_end_index]:
        raise ImpossibleToAnswer("Object is not visible at the required timestep.")

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state_mod, timestep_end
    )

    object_position_at_time = get_position(world_state_og, object_id, timestep_end)
    closest_object = get_closest_object(
        world_state_og, object_id, object_position_at_time, timestep_end
    )

    presents = [
        obj["name"] for obj in iter_objects(world_state_og) if obj["id"] != object_id
    ]

    labels, correct_idx = create_mc_object_names_from_dataset(
        closest_object["name"], presents, get_all_objects_names(), num_answers=4
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


@with_resolved_attributes_cf
def CF_CLOSEST_OBJECT_CAMERA(
    world_state_og: WorldState,
    world_state_mod: WorldState,
    answer_list_original_data_cf: dict,
    question: QuestionPayload,
    attributes,
    **kwargs,
) -> int:
    if kwargs["current_world_number_of_objects"] < 2:
        raise ImpossibleToAnswer(
            "Not enough objects in the scene to answer the question."
        )

    answer_list_original_data_cf = answer_list_original_data_cf[
        1
    ]  # take always the last (video)

    timestep_end_index = int(
        answer_list_original_data_cf[3][-1]
    )  # this has to be the image to get the question
    timestep_end = f"{TIMESTART + float(timestep_end_index) * RENDER_STEP:08.3f}"

    # check if the particular question asks something outside of simulation_og
    if timestep_end_index > len(world_state_og["simulation"]) - 1:
        raise ImpossibleToAnswer("Question refers to future timestep.")

    closest_object = None
    closest_distance = float("inf")

    for object in iter_objects(world_state_mod):
        object_id = object["id"]
        object_position_at_time = get_position(world_state_mod, object_id, timestep_end)
        camera_position_at_time = get_position_camera(world_state_mod, timestep_end)
        distance = distance_between(object_position_at_time, camera_position_at_time)
        if distance < closest_distance:
            closest_distance = distance
            closest_object = object

    presents = [obj["name"] for obj in iter_objects(world_state_mod)]

    labels, correct_idx = create_mc_object_names_from_dataset(
        closest_object["name"], presents, get_all_objects_names(), num_answers=4
    )
    labels = [str(label) for label in labels]

    resolved_attributes = resolve_attributes_visible_at_timestep(
        [], world_state_mod, timestep_end
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
def CF_SIZE_OBJECT(
    world_state_og: WorldState,
    world_state_mod: WorldState,
    answer_list_original_data_cf: dict,
    question: QuestionPayload,
    attributes,
    **kwargs,
) -> int:
    assert len(attributes) == 1 and "OBJECT" in attributes

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        attributes,
        world_state_mod,
        min_objects=kwargs["current_world_number_of_objects"],
    )

    timestep = get_random_timestep_from_list(visible_timesteps, question)

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state_mod, timestep
    )

    object_id = resolved_attributes["OBJECT"]["choice"]["id"]

    volume_object_cubic_meters = world_state_mod["objects"][object_id]["volume"]
    volume_object_cubic_centimeters = volume_object_cubic_meters * 1e6

    options, correct_idx = create_mc_options_around_gt(
        volume_object_cubic_centimeters,
        num_answers=4,
        display_decimals=2,
    )
    labels = uniform_labels(options, integer=False, decimals=2)
    labels = [str(label) + " cubic centimeters" for label in labels]

    return fill_questions_cf(
        question,
        labels,
        correct_idx,
        world_state_og,
        world_state_mod,
        timestep,
        resolved_attributes,
    )


@with_resolved_attributes_cf
def CF_SIZE_OBJECT_BIGGER(
    world_state_og: WorldState,
    world_state_mod: WorldState,
    answer_list_original_data_cf: dict,
    question: QuestionPayload,
    attributes,
    **kwargs,
) -> str:
    assert len(attributes) == 0

    if kwargs["current_world_number_of_objects"] < 2:
        raise ImpossibleToAnswer(
            "Not enough objects in the scene to answer the question."
        )

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        ["OBJECT"],
        world_state_mod,
        min_objects=kwargs["current_world_number_of_objects"],
    )

    timestep = get_random_timestep_from_list(visible_timesteps, question)

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state_mod, timestep
    )

    # Find the biggest object by volume
    biggest_object = None
    biggest_volume = -1.0
    for obj in iter_objects(world_state_mod):
        volume = obj.get("volume", 0.0)

        if volume > biggest_volume and is_object_visible_v3(world_state_mod, obj["id"], timestep):
            biggest_volume = volume
            biggest_object = obj

    presents = [obj["name"] for obj in iter_objects(world_state_mod)]

    labels, correct_idx = create_mc_object_names_from_dataset(
        biggest_object["name"], presents, get_all_objects_names(), num_answers=4
    )

    return fill_questions_cf(
        question,
        labels,
        correct_idx,
        world_state_og,
        world_state_mod,
        timestep,
        resolved_attributes,
    )


@with_resolved_attributes_cf
def CF_LAYOUT_POSITION_OBJECT_OBJECT(
    world_state_og: WorldState,
    world_state_mod: WorldState,
    answer_list_original_data_cf: dict,
    question: QuestionPayload,
    attributes,
    **kwargs,
) -> str:
    assert (
        len(attributes) == 2 and "OBJECT_1" in attributes and "OBJECT_2" in attributes
    )

    answer_list_original_data_cf = answer_list_original_data_cf[
        1
    ]  # take always the last (video)
    timestep_end_index = int(
        answer_list_original_data_cf[3][-1]
    )  # this has to be the image to get the question
    timestep_end = f"{TIMESTART + float(timestep_end_index) * RENDER_STEP:08.3f}"

    # check if the particular question asks something outside of simulation_og
    if timestep_end_index > len(world_state_og["simulation"]) - 1:
        raise ImpossibleToAnswer("Question refers to future timestep.")

    # is just the opposite in the question so trick to make it work
    object_2 = answer_list_original_data_cf[5]["OBJECT_2"]["choice"]
    object_1 = answer_list_original_data_cf[5]["OBJECT_1"]["choice"]

    horizontal, vertical, depth, max_movement_adj = (
        get_spatial_relationship_camera_view(
            world_state_mod["simulation"][timestep_end]["objects"][object_1["id"]],
            world_state_mod["simulation"][timestep_end]["objects"][object_2["id"]],
            world_state_mod["simulation"][timestep_end]["camera"],
            world_state_mod["simulation"][timestep_end]["frame_idx"],
        )
    )

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

    resolved_attributes = resolve_attributes_visible_at_timestep(
        ["OBJECT_1", "OBJECT_2"], world_state_og, timestep_end
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
