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

from utils.load_pointclouds import load_scene_pointcloud

from utils.helpers import (
    iter_objects,
    fill_questions,
    distance_between,
    get_random_timestep_from_list,
    resolve_attributes_visible_at_timestep,
    get_visible_timesteps_for_attributes_min_objects,
)

from .spatial_reasoning_helpers import (
    get_position,
    get_closest_object,
    get_position_camera,        
    get_min_height_from_obb,
    get_min_height_from_obb,
    get_min_distance_pointcloud_to_obb,
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


from utils.config import get_config

VISIBILITY_THRESHOLD = get_config()["visibility_threshold"]
FRAME_INTERLEAVE = get_config()["frame_interleave"]
CLIP_LENGTH = get_config()["clip_length"]
MIN_VISIBLE_PIXELS = get_config()["min_pixels_visible"]

## --- Resolver functions -- ##
## Assumptions: ##
# - All object positions are given by their OBB center
# - Distances are Euclidean distances between object centers unless specified otherwise
# - The valid timesteps are those where all the  objects are visible above VISIBILITY_THRESHOLD

@with_resolved_attributes
def F_DISTANCE_OBJECT_OBJECT(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    assert (
        len(attributes) == 2 and "OBJECT_1" in attributes and "OBJECT_2" in attributes
    )

    if kwargs["current_world_number_of_objects"] < 2:
        raise ImpossibleToAnswer("Not enough objects in the scene to answer the question.")

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        attributes, world_state, min_objects=kwargs["current_world_number_of_objects"]
    )
    
    timestep = get_random_timestep_from_list(
        visible_timesteps, question
    )

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, timestep
    )

    obj1_id = resolved_attributes["OBJECT_1"]["choice"]["id"]
    obj2_id = resolved_attributes["OBJECT_2"]["choice"]["id"]

    obj1_pos = get_position(world_state, obj1_id, timestep)
    obj2_pos = get_position(world_state, obj2_id, timestep)

    distance = distance_between(obj1_pos, obj2_pos)

    options, correct_idx = create_mc_options_around_gt(
        distance, num_answers=4, display_decimals=2, lo=0.0
    )
    labels = uniform_labels(options, integer=False, decimals=2)
    labels = [str(label) + " meters" for label in labels]

    return fill_questions(
        question, labels, correct_idx, world_state, timestep, resolved_attributes
    )


@with_resolved_attributes
def F_DISTANCE_OBJECT_GROUND(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    assert len(attributes) == 1 and "OBJECT" in attributes

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        attributes, world_state, min_objects=kwargs["current_world_number_of_objects"]
    )

    timestep = get_random_timestep_from_list(
        visible_timesteps, question
    )

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, timestep
    )

    obj1_id = resolved_attributes["OBJECT"]["choice"]["id"]

    # loading pointcloud
    pointcloud = load_scene_pointcloud(world_state["scene"]["scene"])

    distance_to_scene = get_min_distance_pointcloud_to_obb(
        pointcloud,
        world_state["simulation"][timestep]["objects"][obj1_id]["obb"],
    )    

    options, correct_idx = create_mc_options_around_gt(
        distance_to_scene, num_answers=4, display_decimals=2, lo=0.0
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
    assert len(attributes) == 1 and "OBJECT" in attributes

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        attributes, world_state, min_objects=kwargs["current_world_number_of_objects"]
    )

    timestep = get_random_timestep_from_list(
        visible_timesteps, question
    )

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, timestep
    )

    object_id = resolved_attributes["OBJECT"]["choice"]["id"]

    object_position_at_time = get_position(world_state, object_id, timestep)
    camera_position_at_time = get_position_camera(world_state, timestep)

    distance = distance_between(object_position_at_time, camera_position_at_time)

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
    
    if kwargs["current_world_number_of_objects"] < 2:
        raise ImpossibleToAnswer("Not enough objects in the scene to answer the question.")

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        attributes, world_state, min_objects=kwargs["current_world_number_of_objects"]
    )

    timestep = get_random_timestep_from_list(
        visible_timesteps, question
    )

    closest_object = None
    closest_distance = float("inf")

    for object in iter_objects(world_state):
        object_id = object["id"]
        object_position_at_time = get_position(world_state, object_id, timestep)
        camera_position_at_time = get_position_camera(world_state, timestep)
        distance = distance_between(object_position_at_time, camera_position_at_time)
        if distance < closest_distance:
            closest_distance = distance
            closest_object = object
    
    presents = [obj["name"] for obj in iter_objects(world_state)]

    labels, correct_idx = create_mc_object_names_from_dataset(
        closest_object["name"], presents, get_all_objects_names(), num_answers=4
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
    assert len(attributes) == 1 and "OBJECT" in attributes

    if kwargs["current_world_number_of_objects"] < 2:
        raise ImpossibleToAnswer("Not enough objects in the scene to answer the question.")

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        attributes, world_state, min_objects=kwargs["current_world_number_of_objects"]
    )

    timestep = get_random_timestep_from_list(
        visible_timesteps, question
    )

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, timestep
    )

    object_id = resolved_attributes["OBJECT"]["choice"]["id"]

    object_position_at_time = get_position(world_state, object_id, timestep)
    closest_object = get_closest_object(
        world_state, object_id, object_position_at_time, timestep
    )

    presents = [obj["name"] for obj in iter_objects(world_state) if obj["id"] != object_id]

    labels, correct_idx = create_mc_object_names_from_dataset(
        closest_object["name"], presents, get_all_objects_names(), num_answers=4
    )

    labels = [str(label) for label in labels]

    return fill_questions(
        question, labels, correct_idx, world_state, timestep, resolved_attributes
    )




@with_resolved_attributes
def F_SIZE_OBJECT(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    assert len(attributes) == 1 and "OBJECT" in attributes

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        attributes, world_state, min_objects=kwargs["current_world_number_of_objects"]
    )

    timestep = get_random_timestep_from_list(
        visible_timesteps, question
    )

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, timestep
    )

    object_id = resolved_attributes["OBJECT"]["choice"]["id"]

    volume_object_cubic_meters = world_state["objects"][object_id]["volume"]
    volume_object_cubic_centimeters = volume_object_cubic_meters * 1e6

    options, correct_idx = create_mc_options_around_gt(
        volume_object_cubic_centimeters,
        num_answers=4,
        display_decimals=2,
    )
    labels = uniform_labels(options, integer=False, decimals=2)
    labels = [str(label) + " cubic centimeters" for label in labels]

    return fill_questions(
        question, labels, correct_idx, world_state, timestep, resolved_attributes
    )


@with_resolved_attributes
def F_SIZE_OBJECT_BIGGER(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> str:
    assert len(attributes) == 0

    if kwargs["current_world_number_of_objects"] < 2:
        raise ImpossibleToAnswer("Not enough objects in the scene to answer the question.")

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        ["OBJECT"], world_state, min_objects=kwargs["current_world_number_of_objects"]
    )
   
    timestep = get_random_timestep_from_list(
        visible_timesteps, question
    )

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, timestep
    )

    # Find the biggest object by volume
    biggest_object = None
    biggest_volume = -1.0
    for obj in iter_objects(world_state):
        volume = obj.get("volume", 0.0)
        visible_at_timestep = (
            world_state["simulation"][timestep]["objects"][obj["id"]]["infov_pixels"] > MIN_VISIBLE_PIXELS
            and world_state["simulation"][timestep]["objects"][obj["id"]][
                "fov_visibility"
            ]
            > VISIBILITY_THRESHOLD
        )
        if volume > biggest_volume and visible_at_timestep:
            biggest_volume = volume
            biggest_object = obj

    presents = [obj["name"] for obj in iter_objects(world_state)]

    labels, correct_idx = create_mc_object_names_from_dataset(
        biggest_object["name"], presents, get_all_objects_names(), num_answers=4
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
        raise ImpossibleToAnswer("Not enough objects in the scene to answer the question.")

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        ["OBJECT"], world_state, min_objects=kwargs["current_world_number_of_objects"]
    )

    timestep = get_random_timestep_from_list(
        visible_timesteps, question
    )

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, timestep
    )

    # Find the smallest object by volume
    smallest_object = None
    smallest_volume = 10e6
    for obj in iter_objects(world_state):
        volume = obj.get("volume", 0.0)
        visible_at_timestep = (
            world_state["simulation"][timestep]["objects"][obj["id"]]["infov_pixels"] > MIN_VISIBLE_PIXELS
            and world_state["simulation"][timestep]["objects"][obj["id"]][
                "fov_visibility"
            ]
            > VISIBILITY_THRESHOLD
        )
        if volume < smallest_volume and visible_at_timestep:
            smallest_volume = volume
            smallest_object = obj

    presents = [obj["name"] for obj in iter_objects(world_state)]

    labels, correct_idx = create_mc_object_names_from_dataset(
        smallest_object["name"], presents, get_all_objects_names(), num_answers=4
    )

    return fill_questions(
        question, labels, correct_idx, world_state, timestep, resolved_attributes
    )


@with_resolved_attributes
def F_LAYOUT_POSITION_OBJECT_OBJECT(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> str:
    assert (
        len(attributes) == 2 and "OBJECT_1" in attributes and "OBJECT_2" in attributes
    )

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        attributes, world_state, min_objects=len(attributes)
    )
    
    timestep = get_random_timestep_from_list(
        visible_timesteps, question
    )

    # I should only be able to resolve the attributes that are not duplicated I hope
    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, timestep
    )

    object_1 = resolved_attributes["OBJECT_1"]["choice"]
    object_2 = resolved_attributes["OBJECT_2"]["choice"]

    horizontal, vertical, depth, max_movement_adj = get_spatial_relationship_camera_view(
        world_state["simulation"][timestep]["objects"][object_1['id']],
        world_state["simulation"][timestep]["objects"][object_2['id']],
        world_state["simulation"][timestep]["camera"],
        world_state["simulation"][timestep]["frame_idx"],
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
