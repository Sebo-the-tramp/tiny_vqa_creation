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

random.seed(42)

from utils.my_exception import ImpossibleToAnswer

from utils.all_objects import get_all_objects_names

from utils.helpers import (
    iter_objects,
    distance_between,
    resolve_attributes_visible_at_timestep,
    get_visible_timesteps_for_attributes_min_objects,
)

from .spatial_reasoning_helpers import (
    get_position,
    get_position_camera,
    point_to_plane_distance,
    fill_questions,
    get_closest_object,
    get_min_height_from_obb,
    get_spatial_relationship_camera_view,
    get_all_relational_positional_adjectives
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

## --- Resolver functions -- ##


@with_resolved_attributes
def F_DISTANCE_OBJECT_OBJECT(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    assert (
        len(attributes) == 2 and "OBJECT_1" in attributes and "OBJECT_2" in attributes
    )

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        attributes, world_state, min_objects=2
    )
    # if we are in a multi-image setting, we need to ensure there are enough frames
    if len(visible_timesteps) == 0:
        raise ImpossibleToAnswer("No timestep with both objects visible.")

    if "multi" in question.get("task_splits", ""):
        timestep = random.choice(visible_timesteps[7:])
    else:
        timestep = random.choice(visible_timesteps)

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, timestep
    )

    obj1_id = resolved_attributes["OBJECT_1"]["choice"]["id"]
    obj2_id = resolved_attributes["OBJECT_2"]["choice"]["id"]

    obj1_pos = get_position(world_state, obj1_id, timestep)
    obj2_pos = get_position(world_state, obj2_id, timestep)

    distance = distance_between(obj1_pos, obj2_pos)

    options, correct_idx = create_mc_options_around_gt(
        distance, num_answers=4, display_decimals=1, lo=0.0
    )
    labels = uniform_labels(options, integer=False, decimals=1)
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
        attributes, world_state, min_objects=1
    )
    # if we are in a multi-image setting, we need to ensure there are enough frames
    if len(visible_timesteps) == 0:
        raise ImpossibleToAnswer("No timestep with visible objects.")

    if "multi" in question.get("task_splits", ""):
        timestep = random.choice(visible_timesteps[7:])
    else:
        timestep = random.choice(visible_timesteps)

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, timestep
    )

    obj1_id = resolved_attributes["OBJECT"]["choice"]["id"]

    # TODO check if those are correct assumptions about ground plane
    ground_normal = [0, 0, 1]
    ground_height = 0.0

    distance_to_ground = get_min_height_from_obb(
        world_state["simulation"][timestep]["objects"][obj1_id]["obb"]
    )

    options, correct_idx = create_mc_options_around_gt(
        distance_to_ground, num_answers=4, display_decimals=1, lo=0.0
    )
    labels = uniform_labels(options, integer=False, decimals=1)
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
        attributes, world_state, min_objects=1
    )
    # if we are in a multi-image setting, we need to ensure there are enough frames
    if len(visible_timesteps) == 0:
        raise ImpossibleToAnswer("No timestep with visible objects.")

    if "multi" in question.get("task_splits", ""):
        timestep = random.choice(visible_timesteps[7:])
    else:
        timestep = random.choice(visible_timesteps)

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, timestep
    )

    object_id = resolved_attributes["OBJECT"]["choice"]["id"]

    object_position_at_time = get_position(world_state, object_id, timestep)
    camera_position_at_time = get_position_camera(world_state, timestep)

    distance = distance_between(object_position_at_time, camera_position_at_time)

    options, correct_idx = create_mc_options_around_gt(
        distance, num_answers=4, display_decimals=1, lo=0.0
    )
    labels = uniform_labels(options, integer=False, decimals=1)
    labels = [str(label) + " meters" for label in labels]

    return fill_questions(
        question, labels, correct_idx, world_state, timestep, resolved_attributes
    )


@with_resolved_attributes
def F_DISTANCE_CLOSEST_OBJECT_OBJECT(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    assert len(attributes) == 1 and "OBJECT" in attributes

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        attributes, world_state, min_objects=1
    )
    # if we are in a multi-image setting, we need to ensure there are enough frames
    if len(visible_timesteps) == 0:
        raise ImpossibleToAnswer("No timestep with visible objects.")

    if "multi" in question.get("task_splits", ""):
        timestep = random.choice(visible_timesteps[7:])
    else:
        timestep = random.choice(visible_timesteps)

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, timestep
    )

    object_id = resolved_attributes["OBJECT"]["choice"]["id"]

    object_position_at_time = get_position(world_state, object_id, timestep)
    closest_object = get_closest_object(
        world_state, object_id, object_position_at_time, timestep
    )
    closest_position_at_time = get_position(world_state, closest_object["id"], timestep)

    distance = distance_between(object_position_at_time, closest_position_at_time)

    options, idx = create_mc_options_around_gt(
        distance, num_answers=4, display_decimals=1, lo=0.0
    )
    labels = uniform_labels(options, integer=False, decimals=1)
    labels = [str(label) + " meters" for label in labels]

    return fill_questions(
        question, labels, idx, world_state, timestep, resolved_attributes
    )


@with_resolved_attributes
def F_SIZE_OBJECT(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    assert len(attributes) == 1 and "OBJECT" in attributes

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        attributes, world_state, min_objects=1
    )
    # if we are in a multi-image setting, we need to ensure there are enough frames
    if len(visible_timesteps) == 0:
        raise ImpossibleToAnswer("No timestep with visible objects.")

    if "multi" in question.get("task_splits", ""):
        timestep = random.choice(visible_timesteps[7:])
    else:
        timestep = random.choice(visible_timesteps)

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, timestep
    )

    object_id = resolved_attributes["OBJECT"]["choice"]["id"]

    volume_object = world_state["objects"][object_id]["volume"]

    options, correct_idx = create_mc_options_around_gt(
        volume_object,
        num_answers=4,
        sig_digits=6,
        display_decimals=6,
        lo=0.0,
        min_rel_gap=0.5,
    )
    labels = uniform_labels(options, integer=False, decimals=6)
    labels = [str(label) + " cubic meters" for label in labels]

    return fill_questions(
        question, labels, correct_idx, world_state, timestep, resolved_attributes
    )


@with_resolved_attributes
def F_SIZE_OBJECT_BIGGER(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> str:
    assert len(attributes) == 0

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        ["OBJECT"], world_state, min_objects=1
    )
    # if we are in a multi-image setting, we need to ensure there are enough frames
    if len(visible_timesteps) == 0:
        raise ImpossibleToAnswer("No timestep with visible objects.")

    if "multi" in question.get("task_splits", ""):
        timestep = random.choice(visible_timesteps[7:])
    else:
        timestep = random.choice(visible_timesteps)

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, timestep
    )

    # Find the biggest object by volume
    biggest_object = None
    biggest_volume = -1.0
    for obj in iter_objects(world_state):
        volume = obj.get("volume", 0.0)
        visible_at_timestep = (
            world_state["simulation"][timestep]["objects"][obj["id"]]["infov"]
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

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        ["OBJECT"], world_state, min_objects=1
    )
    # if we are in a multi-image setting, we need to ensure there are enough frames
    if len(visible_timesteps) == 0:
        raise ImpossibleToAnswer("No timestep with visible objects.")

    if "multi" in question.get("task_splits", ""):
        timestep = random.choice(visible_timesteps[7:])
    else:
        timestep = random.choice(visible_timesteps)

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, timestep
    )

    # Find the smallest object by volume
    smallest_object = None
    smallest_volume = 10e6
    for obj in iter_objects(world_state):
        volume = obj.get("volume", 0.0)
        visible_at_timestep = (
            world_state["simulation"][timestep]["objects"][obj["id"]]["infov"]
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
    assert (len(attributes) == 2 and "OBJECT_1" in attributes and "OBJECT_2" in attributes)

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        attributes, world_state, min_objects=2
    )
    # if we are in a multi-image setting, we need to ensure there are enough frames
    if len(visible_timesteps) == 0:
        raise ImpossibleToAnswer("No timestep with visible objects.")

    if "multi" in question.get("task_splits", ""):
        timestep = random.choice(visible_timesteps[7:])
    else:
        timestep = random.choice(visible_timesteps)

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, timestep
    )

    object_1 = resolved_attributes["OBJECT_1"]["choice"]
    object_2 = resolved_attributes["OBJECT_2"]["choice"]

    horizontal, vertical, depth = get_spatial_relationship_camera_view(
        world_state["simulation"][timestep]["objects"][object_1["id"]],
        world_state["simulation"][timestep]["objects"][object_2["id"]],
        world_state["simulation"][timestep]["camera"],
    )

    DATASET_RELATIONAL_ADJECTIVES = get_all_relational_positional_adjectives()
    #remove correct answers
    DATASET_RELATIONAL_ADJECTIVES.remove(horizontal)
    DATASET_RELATIONAL_ADJECTIVES.remove(vertical)
    DATASET_RELATIONAL_ADJECTIVES.remove(depth)
    
    # confounding options
    random.shuffle(DATASET_RELATIONAL_ADJECTIVES)
    confounding_options = DATASET_RELATIONAL_ADJECTIVES[:3]

    correct_idx = random.randint(0, 3)
    labels = confounding_options[:correct_idx] + [horizontal] + confounding_options[correct_idx:]

    return fill_questions(
        question, labels, correct_idx, world_state, timestep, resolved_attributes
    )
    