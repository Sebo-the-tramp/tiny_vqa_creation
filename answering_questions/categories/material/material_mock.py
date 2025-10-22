"""
Mock material reasoning resolvers.

These helpers extract best-effort material answers from the provided world state.
They operate on lightweight metadata (positions, orientations, region tags, etc.)
and fall back to sensible defaults when information is missing.
"""

from __future__ import annotations

from utils.decorators import with_resolved_attributes
from utils.frames_selection import sample_frames_at_timesteps

from utils.bin_creation import (
    create_mc_options_around_gt,
    create_mc_object_names_from_dataset,
    uniform_labels,
)


from typing import (
    Any,
    Mapping,
    Sequence,
    Tuple,
    Union,
)

from utils.helpers import (
    _iter_objects,
    iter_visible_objects_at_time,
    get_object_state_at_timestep,
    get_all_objects_state_at_time,
    _shuffle_array,
    _get_total_timesteps,
    _get_total_images,
    get_random_timestep,
)

from .material_helpers import (
    get_all_materials,
    get_all_materials_in_scene,
)

from utils.all_objects import get_all_objects_names

Number = Union[int, float]
Vector = Tuple[float, float, float]
WorldState = Mapping[str, Any]
QuestionPayload = Mapping[str, Any]
Answer = Union[str, float, Vector, Mapping[str, Any], Sequence[str]]

DEFORMATION_UNIT = "MPa"

## --- Resolver functions -- ##


@with_resolved_attributes
def F_MATERIALS_COUNTING(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> int:
    assert len(resolved_attributes) == 1 and "MATERIAL" in resolved_attributes

    material = resolved_attributes["MATERIAL"]["choice"]

    timestep = get_random_timestep(world_state)

    count = 0
    for obj in iter_visible_objects_at_time(world_state, timestep):
        if obj["description"]["material_group"] == material:
            count += 1

    options, correct_idx = create_mc_options_around_gt(
        count, num_answers=4, display_decimals=0, lo=0.0
    )
    labels = uniform_labels(options, integer=True, decimals=0)

    return (
        question,
        labels,
        correct_idx,
        sample_frames_at_timesteps(world_state, timesteps=[timestep]),
    )


@with_resolved_attributes
def F_MATERIALS_ATTRIBUTE(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> int:
    assert len(resolved_attributes) == 1 and "OBJECT" in resolved_attributes

    object = resolved_attributes["OBJECT"]["choice"]
    timestep = get_random_timestep(world_state)

    material = "unknown"
    for obj in _iter_objects(world_state):
        if obj["id"] == object["id"]:
            material = obj["description"]["material_group"]
            break

    material_present = get_all_materials_in_scene(world_state)

    labels, correct_idx = create_mc_object_names_from_dataset(
        material, material_present, get_all_materials()
    )

    return (
        question,
        labels,
        correct_idx,
        sample_frames_at_timesteps(world_state, timesteps=[timestep]),
    )


@with_resolved_attributes
def F_MATERIALS_COMPARISON_HARD(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> int:
    assert len(resolved_attributes) == 0

    timestep = get_random_timestep(world_state)

    max_youngs_modulus = -1.0
    harder_object = None

    for obj in iter_visible_objects_at_time(world_state, timestep):
        mean_young_modulus_obeject = (
            obj["description"]["material"]["youngs_modulus_pa"]["min"]
            + obj["description"]["material"]["youngs_modulus_pa"]["max"]
        ) / 2.0
        if mean_young_modulus_obeject > max_youngs_modulus:
            max_youngs_modulus = mean_young_modulus_obeject
            harder_object = obj

    correct_answer = harder_object["model"]

    present = [
        obj["model"]
        for obj in list(_iter_objects(world_state))
        if obj["model"] != correct_answer
    ]

    options, correct_idx = create_mc_object_names_from_dataset(
        correct_answer, present, get_all_objects_names()
    )
    return (
        question,
        options,
        correct_idx,
        sample_frames_at_timesteps(world_state, timesteps=[timestep]),
    )


@with_resolved_attributes
def F_MATERIALS_COMPARISON_SOFT(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> int:
    assert len(resolved_attributes) == 0

    timestep = get_random_timestep(world_state)

    max_youngs_modulus = 10e16
    softer_object = None

    for obj in iter_visible_objects_at_time(world_state, timestep):
        mean_young_modulus_obeject = (
            obj["description"]["material"]["youngs_modulus_pa"]["min"]
            + obj["description"]["material"]["youngs_modulus_pa"]["max"]
        ) / 2.0
        if mean_young_modulus_obeject < max_youngs_modulus:
            max_youngs_modulus = mean_young_modulus_obeject
            softer_object = obj

    correct_answer = softer_object["model"]

    present = [
        obj["model"]
        for obj in list(_iter_objects(world_state))
        if obj["model"] != correct_answer
    ]

    options, correct_idx = create_mc_object_names_from_dataset(
        correct_answer, present, get_all_objects_names()
    )
    return (
        question,
        options,
        correct_idx,
        sample_frames_at_timesteps(world_state, timesteps=[timestep]),
    )
