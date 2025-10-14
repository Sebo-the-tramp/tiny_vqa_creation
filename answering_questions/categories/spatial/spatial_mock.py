"""
Mock spatial reasoning resolvers.

These helpers extract best-effort spatial answers from the provided world state.
They operate on lightweight metadata (positions, orientations, region tags, etc.)
and fall back to sensible defaults when information is missing.
"""

from __future__ import annotations

from utils.decorators import with_resolved_attributes
from utils.bin_creation import (
    create_mc_options_around_gt,
    create_mc_object_names_from_dataset,
    uniform_labels,
)

from utils.all_objects import get_all_objects_names

from typing import (
    Any,
    Mapping,
    Sequence,
    Tuple,
    Union,
)

from utils.helpers import (
    _distance_between,
    _iter_objects,
    _fill_template,
)

from .spatial_helpers import (
    _get_position,
    point_to_plane_distance,
    _get_position_camera,
)

Number = Union[int, float]
Vector = Tuple[float, float, float]
WorldState = Mapping[str, Any]
QuestionPayload = Mapping[str, Any]
Answer = Union[str, float, Vector, Mapping[str, Any], Sequence[str]]

## --- Resolver functions -- ##


@with_resolved_attributes
def F_SPATIAL_OBJECT_GROUND_DISTANCE(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> int:
    assert (
        len(resolved_attributes) == 2
        and "OBJECT" in resolved_attributes
        and "TIME" in resolved_attributes
    )

    timestamp = resolved_attributes["TIME"]["choice"]
    object = resolved_attributes["OBJECT"]["choice"]
    ground_normal = world_state["segments"]["ground"].get(
        "plane_normal", [0, 0, 1]
    )  # TODO change the format eventually
    ground_height = world_state["segments"]["ground"].get("ground_height", 0.0)

    object_position_at_time = _get_position(world_state, object["id"], timestamp)

    distance = point_to_plane_distance(
        object_position_at_time, ground_normal, ground_height
    )

    _fill_template(question, resolved_attributes)
    options, correct_idx = create_mc_options_around_gt(
        distance, num_answers=4, display_decimals=1, lo=0.0
    )
    labels = uniform_labels(options, integer=False, decimals=1)
    labels = [str(label) + " meters" for label in labels]

    return question, labels, correct_idx


@with_resolved_attributes
def F_SPATIAL_OBJECT_OBJECT_DISTANCE(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> int:
    assert (
        len(resolved_attributes) == 3
        and "OBJECT_1" in resolved_attributes
        and "OBJECT_2" in resolved_attributes
        and "TIME" in resolved_attributes
    )

    timestamp = resolved_attributes["TIME"]["choice"]
    object_1 = resolved_attributes["OBJECT_1"]["choice"]
    object_2 = resolved_attributes["OBJECT_2"]["choice"]

    object_1_position_at_time = _get_position(world_state, object_1["id"], timestamp)
    object_2_position_at_time = _get_position(world_state, object_2["id"], timestamp)

    distance = _distance_between(object_1_position_at_time, object_2_position_at_time)

    _fill_template(question, resolved_attributes)
    options, correct_idx = create_mc_options_around_gt(
        distance, num_answers=4, display_decimals=1, lo=0.0
    )
    labels = uniform_labels(options, integer=False, decimals=1)
    labels = [str(label) + " meters" for label in labels]

    return question, labels, correct_idx


@with_resolved_attributes
def F_SPATIAL_OBJECT_CAMERA_DISTANCE(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> int:
    assert (
        len(resolved_attributes) == 3
        and "OBJECT" in resolved_attributes
        and "TIME" in resolved_attributes
        and "CAMERA" in resolved_attributes
    )

    timestamp = resolved_attributes["TIME"]["choice"]
    object = resolved_attributes["OBJECT"]["choice"]

    object_position_at_time = _get_position(world_state, object["id"], timestamp)
    camera_position_at_time = _get_position_camera(world_state, timestamp)

    distance = _distance_between(object_position_at_time, camera_position_at_time)

    _fill_template(question, resolved_attributes)
    options, correct_idx = create_mc_options_around_gt(
        distance, num_answers=4, display_decimals=1, lo=0.0
    )
    labels = uniform_labels(options, integer=False, decimals=1)
    labels = [str(label) + " meters" for label in labels]

    return question, labels, correct_idx


@with_resolved_attributes
def F_SPATIAL_CLOSEST_OBJECT(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> int:
    assert len(resolved_attributes) == 2 and "OBJECT" in resolved_attributes and "TIME"

    timestamp = resolved_attributes["TIME"]["choice"]
    object = resolved_attributes["OBJECT"]["choice"]
    object_position_at_time = _get_position(world_state, object["id"], timestamp)

    closest_object = None
    closest_distance = float("inf")

    _iter_objects_list = list(_iter_objects(world_state))
    for object_iter in _iter_objects_list:
        if object_iter["id"] != object["id"]:
            distance = _distance_between(
                object_position_at_time,
                _get_position(world_state, object_iter["id"], timestamp),
            )
            if distance < closest_distance:
                closest_distance = distance
                closest_object = object_iter

    print(
        f"Closest object to {object['id']} at time {timestamp} is {closest_object} at distance {closest_distance}"
    )

    _fill_template(question, resolved_attributes)
    # here we need to have

    DATASET = get_all_objects_names()
    present = [obj["name"] for obj in _iter_objects_list if obj["id"] != object["id"]]

    labels, idx = create_mc_object_names_from_dataset(
        closest_object["name"], present, DATASET
    )
    print(f"Labels: {labels}, idx: {idx}")

    return question, labels, idx


@with_resolved_attributes
def F_SPATIAL_COUNTING_OBJECTS_CLOSE(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> int:
    assert (
        len(resolved_attributes) == 2
        and "TIME" in resolved_attributes
        and "DISTANCE" in resolved_attributes
    )

    timestamp = resolved_attributes["TIME"]["choice"]
    distance_threshold = resolved_attributes["DISTANCE"]["choice"]

    camera_position_at_time = _get_position_camera(world_state, timestamp)

    count = 0
    for object_iter in _iter_objects(world_state):
        object_position_at_time = _get_position(
            world_state, object_iter["id"], timestamp
        )
        distance = _distance_between(object_position_at_time, camera_position_at_time)
        if distance <= distance_threshold:
            count += 1

    _fill_template(question, resolved_attributes)
    options, correct_idx = create_mc_options_around_gt(
        count, num_answers=4, display_decimals=0, lo=0.0
    )
    labels = uniform_labels(options, integer=True, decimals=0)
    labels = [str(label) for label in labels]

    return question, labels, correct_idx
