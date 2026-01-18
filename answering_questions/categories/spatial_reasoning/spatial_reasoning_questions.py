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

from utils.config import get_config
from utils.geometry import get_camera_OBB

from utils.my_exception import ImpossibleToAnswer

from utils.helpers import (
    iter_objects,
    fill_questions,
    is_object_visible,
    get_random_timestep_from_list,
    minimum_distance_between_OBBs,
    get_objects_present_and_not_present,
    resolve_attributes_visible_at_timestep,
    get_visible_timesteps_for_attributes_min_objects,
)
from .spatial_reasoning_helpers import (
    get_closest_visible_object,
    get_spatial_relationship_camera_view,
    get_all_relational_positional_adjectives,
)
from utils.bin_creation import (
    uniform_labels,
    create_mc_options_around_gt,
    create_mc_object_names_from_dataset,
)

Number = Union[int, float]
Vector = Tuple[float, float, float]
WorldState = Mapping[str, Any]
QuestionPayload = Mapping[str, Any]
Answer = Union[str, float, Vector, Mapping[str, Any], Sequence[str]]

CLIP_LENGTH = get_config()["clip_length"]
FRAME_INTERLEAVE = get_config()["frame_interleave"]
MIN_VISIBLE_PIXELS = get_config()["min_pixels_visible"]
VISIBILITY_THRESHOLD = get_config()["visibility_threshold"]

## --- Resolver functions -- ##
## Assumptions: ##
# - All object positions are given by their OBB center
# - Distances are Euclidean distances between object centers unless specified otherwise
# - The valid timesteps are those where all the  objects are visible above VISIBILITY_THRESHOLD


@with_resolved_attributes
def F_DISTANCE_OBJECT_OBJECT(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    """Question: What is the distance between the geometrical centers of <OBJECT_1> and the <OBJECT_2>?"""
    assert (
        len(attributes) == 2 and "OBJECT_1" in attributes and "OBJECT_2" in attributes
    )

    if kwargs["current_world_number_of_objects"] < 2:
        raise ImpossibleToAnswer(
            "Not enough objects in the scene to answer the question."
        )

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        attributes, world_state, min_objects=2
    )

    timestep = get_random_timestep_from_list(visible_timesteps, question)

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, timestep
    )

    obj1_id = resolved_attributes["OBJECT_1"]["choice"]["id"]
    obj2_id = resolved_attributes["OBJECT_2"]["choice"]["id"]

    obj1_state = world_state["simulation"][timestep]["objects"][obj1_id]
    obj2_state = world_state["simulation"][timestep]["objects"][obj2_id]

    distance = minimum_distance_between_OBBs(obj1_state["obb"], obj2_state["obb"])

    options, correct_idx = create_mc_options_around_gt(
        distance, num_answers=4, display_decimals=2, lo=0.0
    )
    labels = uniform_labels(options, integer=False, decimals=2)
    labels = [str(label) + " meters" for label in labels]

    return fill_questions(
        question, labels, correct_idx, world_state, timestep, resolved_attributes
    )


@with_resolved_attributes
def F_DISTANCE_OBJECT_CAMERA_DISTANCE(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    """Question: What is the distance between the geometrical center of the <OBJECT> and the camera?"""
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

    object_state = world_state["simulation"][timestep]["objects"][object_id]
    camera_state = world_state["simulation"][timestep]["camera"]

    camera_OBB = get_camera_OBB(camera_state)
    object_OBB = object_state["obb"]

    distance = minimum_distance_between_OBBs(object_OBB, camera_OBB)

    options, correct_idx = create_mc_options_around_gt(
        distance, num_answers=4, display_decimals=2, lo=0.0
    )
    labels = uniform_labels(options, integer=False, decimals=2)
    labels = [str(label) + " meters" for label in labels]

    return fill_questions(
        question, labels, correct_idx, world_state, timestep, resolved_attributes
    )


@with_resolved_attributes
def F_CLOSEST_OBJECT_CAMERA(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    """Question: Which object in the image is the closest to the camera?"""
    assert len(attributes) == 0

    # we need this cause else there cannot be a comparison
    if kwargs["current_world_number_of_objects"] < 2:
        raise ImpossibleToAnswer(
            "Not enough objects in the scene to answer the question."
        )

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        attributes, world_state, min_objects=2
    )

    timestep = get_random_timestep_from_list(visible_timesteps, question)

    closest_object = None
    closest_distance = float("inf")

    for object in iter_objects(world_state):
        object_id = object["id"]

        if not is_object_visible(world_state, object_id, timestep):
            continue

        object_state = world_state["simulation"][timestep]["objects"][object_id]
        camera_state = world_state["simulation"][timestep]["camera"]

        camera_OBB = get_camera_OBB(camera_state)
        object_OBB = object_state["obb"]

        distance = minimum_distance_between_OBBs(object_OBB, camera_OBB)

        if distance < closest_distance:
            closest_distance = distance
            closest_object = object

    visible_objects_names_minus_resolved, all_objects_minus_visible_and_non_visible = (
        get_objects_present_and_not_present(
            world_state, timestep, [closest_object["name"]]
        )
    )

    labels, correct_idx = create_mc_object_names_from_dataset(
        closest_object["name"],
        visible_objects_names_minus_resolved,
        all_objects_minus_visible_and_non_visible,
        num_answers=4,
    )
    labels = [str(label) for label in labels]

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, timestep
    )

    return fill_questions(
        question, labels, correct_idx, world_state, timestep, resolved_attributes
    )


@with_resolved_attributes
def F_CLOSEST_OBJECT_OBJECT(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    """Question: Which object in the image is closest to the geometrical center of the <OBJECT>?"""
    assert len(attributes) == 1 and "OBJECT" in attributes

    if kwargs["current_world_number_of_objects"] < 2:
        raise ImpossibleToAnswer(
            "Not enough objects in the scene to answer the question."
        )

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        attributes, world_state, min_objects=2
    )

    timestep = get_random_timestep_from_list(visible_timesteps, question)

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, timestep
    )

    object_id = resolved_attributes["OBJECT"]["choice"]["id"]

    closest_object = get_closest_visible_object(world_state, object_id, timestep)

    visible_objects_names_minus_resolved, all_objects_minus_visible_and_non_visible = (
        get_objects_present_and_not_present(
            world_state, timestep, [closest_object["name"]]
        )
    )

    labels, correct_idx = create_mc_object_names_from_dataset(
        closest_object["name"],
        visible_objects_names_minus_resolved,
        all_objects_minus_visible_and_non_visible,
        num_answers=4,
    )

    labels = [str(label) for label in labels]

    return fill_questions(
        question, labels, correct_idx, world_state, timestep, resolved_attributes
    )


@with_resolved_attributes
def F_SIZE_OBJECT(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    """Question: What is the volume of the <OBJECT> in the image?"""
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

    extents = world_state["simulation"][timestep]["objects"][object_id]["obb"][
        "extents"
    ]

    first_extent = extents[0]

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
        f"{extents[0] * scale:.2f}m x {extents[1] * scale:.2f}m x {extents[2] * scale:.2f}m"
        for scale in scales
    ]

    return fill_questions(
        question, labels, correct_idx, world_state, timestep, resolved_attributes
    )


@with_resolved_attributes
def F_SIZE_OBJECT_BIGGER(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> str:
    """Question: Which single object in the image has the biggest volume?"""
    assert len(attributes) == 0

    if kwargs["current_world_number_of_objects"] < 2:
        raise ImpossibleToAnswer(
            "Not enough objects in the scene to answer the question."
        )

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        ["OBJECT"],
        world_state,
        min_objects=2,
    )

    timestep = get_random_timestep_from_list(visible_timesteps, question)

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, timestep
    )

    # Find the biggest object by volume
    biggest_object = None
    biggest_volume = -1.0
    total_object_seen = 0

    for obj in iter_objects(world_state):
        volume = obj.get("volume", 0.0)

        if is_object_visible(world_state, obj["id"], timestep) and volume is not None:
            total_object_seen += 1
            if volume > biggest_volume:
                biggest_volume = volume
                biggest_object = obj

    if total_object_seen <= 1:
        raise ImpossibleToAnswer("No visible objects to compare.")

    visible_objects_names_minus_resolved, all_objects_minus_visible_and_non_visible = (
        get_objects_present_and_not_present(
            world_state, timestep, [biggest_object["name"]]
        )
    )

    labels, correct_idx = create_mc_object_names_from_dataset(
        biggest_object["name"],
        visible_objects_names_minus_resolved,
        all_objects_minus_visible_and_non_visible,
        num_answers=4,
    )

    return fill_questions(
        question, labels, correct_idx, world_state, timestep, resolved_attributes
    )


@with_resolved_attributes
def F_SIZE_OBJECT_SMALLER(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> str:
    assert len(attributes) == 0

    if kwargs["current_world_number_of_objects"] < 2:
        raise ImpossibleToAnswer(
            "Not enough objects in the scene to answer the question."
        )

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        ["OBJECT"], world_state, min_objects=2
    )

    timestep = get_random_timestep_from_list(visible_timesteps, question)

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, timestep
    )

    # Find the smallest object by volume
    smallest_object = None
    smallest_volume = 10e6
    total_object_seen = 0

    for obj in iter_objects(world_state):
        volume = obj.get("volume", 0.0)

        if (
            is_object_visible(
                world_state=world_state, obj_id=obj["id"], timestep=timestep
            )
            and volume is not None
        ):
            total_object_seen += 1

            if volume < smallest_volume:
                smallest_volume = volume
                smallest_object = obj

    if total_object_seen <= 1:
        raise ImpossibleToAnswer("No visible objects to compare.")

    visible_objects_names_minus_resolved, all_objects_minus_visible_and_non_visible = (
        get_objects_present_and_not_present(
            world_state, timestep, [smallest_object["name"]]
        )
    )

    labels, correct_idx = create_mc_object_names_from_dataset(
        smallest_object["name"],
        visible_objects_names_minus_resolved,
        all_objects_minus_visible_and_non_visible,
        num_answers=4,
    )

    return fill_questions(
        question, labels, correct_idx, world_state, timestep, resolved_attributes
    )


@with_resolved_attributes
def F_LAYOUT_POSITION_OBJECT_OBJECT(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> str:
    """Question: From the camera’s perspective, where is the <OBJECT_1> relative to the <OBJECT_2>?"""
    assert (
        len(attributes) == 2 and "OBJECT_1" in attributes and "OBJECT_2" in attributes
    )

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        attributes, world_state, min_objects=len(attributes)
    )

    timestep = get_random_timestep_from_list(visible_timesteps, question)

    # I should only be able to resolve the attributes that are not duplicated I hope
    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, timestep
    )

    # is just the opposite in the question so trick to make it work
    object_2 = resolved_attributes["OBJECT_1"]["choice"]
    object_1 = resolved_attributes["OBJECT_2"]["choice"]

    horizontal, vertical, depth, max_movement_adj = (
        get_spatial_relationship_camera_view(
            world_state["simulation"][timestep]["objects"][object_1["id"]],
            world_state["simulation"][timestep]["objects"][object_2["id"]],
            world_state["simulation"][timestep]["camera"],
            world_state["simulation"][timestep]["frame_idx"],
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

    return fill_questions(
        question, labels, correct_idx, world_state, timestep, resolved_attributes
    )
