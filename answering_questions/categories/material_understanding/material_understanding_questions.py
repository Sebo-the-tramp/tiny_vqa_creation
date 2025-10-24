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

from utils.all_objects import get_all_objects_names, get_all_materials

from utils.helpers import (
    iter_objects,
    distance_between,
    resolve_attributes_visible_at_timestep,
    get_visible_timesteps_for_attributes_min_objects,
    get_continuous_subsequences_min_length,
)

from .material_understanding_helpers import (
    get_speed,
    fill_questions,
    get_acceleration,
    get_position,
)

from utils.config import get_config

from utils.bin_creation import (
    create_mc_options_around_gt,
    create_mc_options_around_gt_log,
    uniform_labels,
    create_mc_object_names_from_dataset,
)

Number = Union[int, float]
Vector = Tuple[float, float, float]
WorldState = Mapping[str, Any]
QuestionPayload = Mapping[str, Any]
Answer = Union[str, float, Vector, Mapping[str, Any], Sequence[str]]

CLIP_LENGTH = get_config()["clip_length"]
MOVEMENT_TOLERANCE = get_config()["movement_tolerance"]
VISIBILITY_THRESHOLD = get_config()["visibility_threshold"]

## --- Resolver functions -- ##


@with_resolved_attributes
def F_MASS_OBJECT(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    assert len(attributes) == 1 and "OBJECT" in attributes

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

    object = resolved_attributes["OBJECT"]["choice"]

    mass = object["mass"]

    options, correct_idx = create_mc_options_around_gt(
        mass, num_answers=4, display_decimals=1, lo=0.0
    )
    labels = uniform_labels(options, integer=False, decimals=1)
    labels = [str(label) + " kgs" for label in labels]

    return fill_questions(
        question, labels, correct_idx, world_state, timestep, resolved_attributes
    )


@with_resolved_attributes
def F_MASS_HEAVIEST_OBJECT(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    assert len(attributes) == 0

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        ["OBJECT"], world_state, min_objects=1
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

    heaviest_visible_object = None
    for obj in iter_objects(world_state):
        obj_state = world_state["simulation"][timestep]["objects"][obj["id"]]

        is_object_visible = (
            obj_state["infov"] and obj_state["fov_visibility"] >= VISIBILITY_THRESHOLD
        )

        if (
            heaviest_visible_object is None
            or obj["mass"] > heaviest_visible_object["mass"]
        ) and is_object_visible:
            heaviest_visible_object = obj

    presents = [obj["name"] for obj in iter_objects(world_state)]

    labels, correct_idx = create_mc_object_names_from_dataset(
        heaviest_visible_object["name"],
        presents,
        get_all_objects_names(),
        num_answers=4,
    )

    return fill_questions(
        question, labels, correct_idx, world_state, timestep, resolved_attributes
    )


@with_resolved_attributes
def F_MASS_LIGHTEST_OBJECT(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    assert len(attributes) == 0

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        ["OBJECT"], world_state, min_objects=1
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

    lightest_visible_object = None
    for obj in iter_objects(world_state):
        obj_state = world_state["simulation"][timestep]["objects"][obj["id"]]

        is_object_visible = (
            obj_state["infov"] and obj_state["fov_visibility"] >= VISIBILITY_THRESHOLD
        )

        if (
            lightest_visible_object is None
            or obj["mass"] < lightest_visible_object["mass"]
        ) and is_object_visible:
            lightest_visible_object = obj

    presents = [obj["name"] for obj in iter_objects(world_state)]

    labels, correct_idx = create_mc_object_names_from_dataset(
        lightest_visible_object["name"],
        presents,
        get_all_objects_names(),
        num_answers=4,
    )

    return fill_questions(
        question, labels, correct_idx, world_state, timestep, resolved_attributes
    )


@with_resolved_attributes
def F_MASS_COMPARE_OBJECTS(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    assert (
        len(attributes) == 2 and "OBJECT_1" in attributes and "OBJECT_2" in attributes
    )

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        ["OBJECT"], world_state, min_objects=1
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

    obj1 = resolved_attributes["OBJECT_1"]["choice"]
    obj2 = resolved_attributes["OBJECT_2"]["choice"]

    options = ["yes", "no"]

    if obj1["mass"] < obj2["mass"]:
        correct_idx = 1  # no
    else:
        correct_idx = 0  # yes

    return fill_questions(
        question, options, correct_idx, world_state, timestep, resolved_attributes
    )


@with_resolved_attributes
def F_PHYSICS_PROPERTY_DENSITY_OBJECT(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    assert len(attributes) == 1 and "OBJECT"

    # First we find the pairs of objects visible
    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        ["OBJECT"], world_state, min_objects=1
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

    object = resolved_attributes["OBJECT"]["choice"]

    density = object["props"]["rhos"]

    options, correct_idx = create_mc_options_around_gt(
        density, num_answers=4, display_decimals=1, lo=0.0
    )
    labels = uniform_labels(options, integer=False, decimals=1)
    labels = [str(label) + " kg/m^3" for label in labels]

    return fill_questions(
        question, labels, correct_idx, world_state, timestep, resolved_attributes
    )


@with_resolved_attributes
def F_PHYSICS_PROPERTY_YOUNG_MODULUS_OBJECT(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    assert len(attributes) == 1 and "OBJECT"

    # First we find the pairs of objects visible
    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        ["OBJECT"], world_state, min_objects=1
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

    object = resolved_attributes["OBJECT"]["choice"]

    youngs_modulus = object["props"]["yms"]

    options, correct_idx = create_mc_options_around_gt_log(
        youngs_modulus, num_answers=4, display_decimals=1, lo=0.0
    )
    labels = uniform_labels(options, integer=False, decimals=1)
    labels = [str(label) + " Pa" for label in labels]

    return fill_questions(
        question, labels, correct_idx, world_state, timestep, resolved_attributes
    )


@with_resolved_attributes
def F_PHYSICS_PROPERTY_POISSON_RATIO_OBJECT(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    assert len(attributes) == 1 and "OBJECT"

    # First we find the pairs of objects visible
    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        ["OBJECT"], world_state, min_objects=1
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

    object = resolved_attributes["OBJECT"]["choice"]

    poisson_ratio = object["props"]["prs"]

    options, correct_idx = create_mc_options_around_gt(
        poisson_ratio, num_answers=4, display_decimals=2
    )
    labels = uniform_labels(options, integer=False, decimals=2)
    labels = [str(label) for label in labels]

    return fill_questions(
        question, labels, correct_idx, world_state, timestep, resolved_attributes
    )


@with_resolved_attributes
def F_MATERIAL_IDENTIFICATION_OBJECT(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    assert len(attributes) == 1 and "OBJECT"

    # First we find the pairs of objects visible
    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        ["OBJECT"], world_state, min_objects=1
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

    object = resolved_attributes["OBJECT"]["choice"]
    material = object["description"]["material_group"]

    MATERIALS_ALL = get_all_materials()

    present = [
        obj["description"]["material_group"] for obj in iter_objects(world_state)
    ]

    options, correct_idx = create_mc_object_names_from_dataset(
        material, present, MATERIALS_ALL, num_answers=4
    )

    return fill_questions(
        question, options, correct_idx, world_state, timestep, resolved_attributes
    )
