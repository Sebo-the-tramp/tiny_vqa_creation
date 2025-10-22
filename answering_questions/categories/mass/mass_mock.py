"""
Mock mass reasoning resolvers.

These helpers extract best-effort mass answers from the provided world state.
They operate on lightweight metadata (positions, orientations, region tags, etc.)
and fall back to sensible defaults when information is missing.
"""

from __future__ import annotations

from utils.decorators import with_resolved_attributes
from utils.frames_selection import uniformly_sample_frames

from utils.bin_creation import (
    create_mc_object_names_from_dataset,
    create_mc_options_around_gt,
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
    _iter_objects,
    _iter_visible_objects,
)


# from .mass_helpers import (
# )

Number = Union[int, float]
Vector = Tuple[float, float, float]
WorldState = Mapping[str, Any]
QuestionPayload = Mapping[str, Any]
Answer = Union[str, float, Vector, Mapping[str, Any], Sequence[str]]

## --- Resolver functions -- ##


@with_resolved_attributes
def F_MASS_COUNTING(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> int:
    assert len(resolved_attributes) == 1 and "MASS" in resolved_attributes

    mass_threshold = resolved_attributes["MASS"]["choice"]

    count = 0
    for obj in _iter_visible_objects(world_state):
        if obj["mass"] >= mass_threshold:
            count += 1

    options, correct_idx = create_mc_options_around_gt(
        count, num_answers=4, display_decimals=0, lo=0.0
    )
    labels = uniform_labels(options, integer=True, decimals=0)
    labels = [str(label) for label in labels]

    return question, labels, correct_idx, uniformly_sample_frames(world_state)


@with_resolved_attributes
def F_MASS_ATTRIBUTE(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> str:
    assert len(resolved_attributes) == 1 and "OBJECT" in resolved_attributes

    object = resolved_attributes["OBJECT"]["choice"]
    mass = object["mass"]

    options, correct_idx = create_mc_options_around_gt(
        mass, num_answers=4, display_decimals=1, lo=0.1, min_rel_gap=0.4
    )
    labels = uniform_labels(options, integer=False, decimals=1)
    labels = [str(label) for label in labels]

    return question, labels, correct_idx, uniformly_sample_frames(world_state)


@with_resolved_attributes
def F_MASS_ATTRIBUTE_HEAVIEST(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> str:
    assert len(resolved_attributes) == 0

    heaviest_mass = -1.0
    object = None
    for obj in _iter_objects(world_state):
        if "mass" not in obj:
            raise ValueError(f"Object {obj['id']} is missing 'mass' attribute.")
        if obj["mass"] > heaviest_mass:
            heaviest_mass = obj["mass"]
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


@with_resolved_attributes
def F_MASS_ATTRIBUTE_LIGHTEST(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> str:
    assert len(resolved_attributes) == 0

    lightest_mass = 10e10  # very high outside of our expected range
    object = None
    for obj in _iter_objects(world_state):
        if "mass" not in obj:
            raise ValueError(f"Object {obj['id']} is missing 'mass' attribute.")
        if obj["mass"] < lightest_mass:
            lightest_mass = obj["mass"]
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


@with_resolved_attributes
def F_DENSITY_COUNTING(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> str:
    assert len(resolved_attributes) == 1 and "DENSITY" in resolved_attributes

    density_threshold = resolved_attributes["DENSITY"]["choice"]

    count = 0

    for obj in _iter_objects(world_state):
        min_object_density = obj["description"]["material"]["density_kg_per_m3"]["min"]
        max_object_density = obj["description"]["material"]["density_kg_per_m3"]["max"]
        mean_object_density = (min_object_density + max_object_density) / 2.0
        if mean_object_density > density_threshold:
            count += 1

    options, correct_idx = create_mc_options_around_gt(
        count, num_answers=4, display_decimals=0, lo=0.1
    )
    labels = uniform_labels(options, integer=True, decimals=0)
    labels = [str(label) for label in labels]

    return question, labels, correct_idx, uniformly_sample_frames(world_state)


@with_resolved_attributes
def F_DENSITY_ATTRIBUTE(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> str:
    assert len(resolved_attributes) == 1 and "OBJECT" in resolved_attributes

    # TODO check here as well about visibility at time T before getting the choice
    object = resolved_attributes["OBJECT"]["choice"]

    min_object_density = object["description"]["material"]["density_kg_per_m3"]["min"]
    max_object_density = object["description"]["material"]["density_kg_per_m3"]["max"]
    mean_object_density = (min_object_density + max_object_density) / 2.0

    options, correct_idx = create_mc_options_around_gt(
        mean_object_density, num_answers=4, display_decimals=1, lo=0.1
    )
    labels = uniform_labels(options, integer=False, decimals=1)

    return question, labels, correct_idx, uniformly_sample_frames(world_state)
