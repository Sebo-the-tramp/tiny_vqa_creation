"""
Mock volume reasoning resolvers.

These helpers extract best-effort volume answers from the provided world state.
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


Number = Union[int, float]
Vector = Tuple[float, float, float]
WorldState = Mapping[str, Any]
QuestionPayload = Mapping[str, Any]
Answer = Union[str, float, Vector, Mapping[str, Any], Sequence[str]]

from utils.all_objects import get_all_objects_names
from utils.decorators import with_resolved_attributes
from utils.frames_selection import uniformly_sample_frames

from utils.bin_creation import (
    create_mc_options_around_gt,
    uniform_labels,
)

from utils.bin_creation import (
    create_mc_object_names_from_dataset,
)

from utils.helpers import (
    _iter_objects,
    iter_visible_objects_at_time,
)


## --- Resolver functions -- ##


@with_resolved_attributes
def F_VOLUME_COUNTING(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> str:
    assert len(resolved_attributes) == 1 and "VOLUME" in resolved_attributes

    volume_threshold = resolved_attributes["VOLUME"]["choice"]

    count = 0

    objects = _iter_objects(world_state)

    for object in objects:
        if object["volume"] >= volume_threshold:
            count += 1

    options, correct_idx = create_mc_options_around_gt(
        count, num_answers=4, display_decimals=0, lo=0.0
    )
    labels = uniform_labels(options, integer=True, decimals=0)
    labels = [str(label) for label in labels]

    return question, labels, correct_idx, uniformly_sample_frames(world_state)


@with_resolved_attributes
def F_VOLUME_ATTRIBUTE(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> str:
    assert len(resolved_attributes) == 1 and "OBJECT" in resolved_attributes

    # TODO make sure that objects is visible at time T, before getting the choice§
    object = resolved_attributes["OBJECT"]["choice"]

    volume = object["volume"] * 1e6  # convert from m3 to cm3

    options, correct_idx = create_mc_options_around_gt(
        volume, num_answers=4, display_decimals=2, lo=0.01, min_rel_gap=0.4
    )
    labels = uniform_labels(options, integer=False, decimals=1)
    labels = [f"{label} cm^3" for label in labels]

    return question, labels, correct_idx, uniformly_sample_frames(world_state)


@with_resolved_attributes
def F_VOLUME_ATTRIBUTE_MOST_VOLUME(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> str:
    assert len(resolved_attributes) == 0

    largest_volume = -1.0
    object = None
    for obj in _iter_objects(world_state):
        volume = obj["volume"] * 1e6  # convert from m3 to cm3
        if volume > largest_volume:
            largest_volume = volume
            object = obj

    if object is None:
        raise ValueError("No objects found in the world state.")

    DATASET = get_all_objects_names()
    present = [
        obj["name"] for obj in _iter_objects(world_state) if obj["id"] != object["id"]
    ]

    options, correct_idx = create_mc_object_names_from_dataset(
        object["name"], present, DATASET
    )

    return question, options, correct_idx, uniformly_sample_frames(world_state)
