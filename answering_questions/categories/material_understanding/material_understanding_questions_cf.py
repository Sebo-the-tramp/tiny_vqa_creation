"""
Mock material understanding resolvers.

These helpers extract best-effort material answers from the provided world state.
"""

from __future__ import annotations

from utils.decorators import with_resolved_attributes_cf

from typing import (
    Any,
    Mapping,
    Sequence,
    Tuple,
    Union,
)


from utils.my_exception import ImpossibleToAnswer

from utils.all_objects import get_all_objects_names, get_all_materials

from utils.helpers import (
    fill_questions_cf,
    iter_objects,
    get_random_timestep_from_list,
    resolve_attributes_visible_at_timestep,
    get_visible_timesteps_for_attributes_min_objects,
)

from utils.config import get_config

from utils.bin_creation import (
    create_mc_options_around_gt,
    create_mc_options_around_gt_log,
    create_mc_options_around_gt_poisson_ratio,
    uniform_labels,
    create_mc_object_names_from_dataset,
)

from .material_understanding_helpers import get_material_dataset_different_from_target

Number = Union[int, float]
Vector = Tuple[float, float, float]
WorldState = Mapping[str, Any]
QuestionPayload = Mapping[str, Any]
Answer = Union[str, float, Vector, Mapping[str, Any], Sequence[str]]

CLIP_LENGTH = get_config()["clip_length"]
MOVEMENT_TOLERANCE = get_config()["movement_tolerance"]
VISIBILITY_THRESHOLD = get_config()["visibility_threshold"]
THRESHOLD_DIFFERENCE_PERCENTAGE = get_config()["threshold_difference_percentage"]
MIN_VISIBLE_PIXELS = get_config()["min_pixels_visible"]
MAX_ALLOWED_DIFFERENCE_YOUNGS_MODULUS = get_config()["max_allowed_difference_youngs_modulus"]
MAX_ALLOWED_DIFFERENCE_POISSON_RATIO = get_config()["max_allowed_difference_poisson_ratio"]

## --- Resolver functions -- ##


@with_resolved_attributes_cf
def CF_MASS_OBJECT(
    world_state_og: WorldState, world_state_mod: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    assert len(attributes) == 1 and "OBJECT-CF" in attributes

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        attributes, world_state_mod, min_objects=kwargs["current_world_number_of_objects"]
    )

    timestep = get_random_timestep_from_list(visible_timesteps, question)

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state_mod, timestep
    )

    object = resolved_attributes["OBJECT-CF"]["choice"]

    mass = object["mass"]

    options, correct_idx = create_mc_options_around_gt(
        mass, num_answers=4, display_decimals=2, lo=0.0
    )
    labels = uniform_labels(options, integer=False, decimals=2)
    labels = [str(label) + " kgs" for label in labels]

    return fill_questions_cf(
        question, labels, correct_idx, world_state_og, world_state_mod, timestep, resolved_attributes
    )


@with_resolved_attributes_cf
def CF_MASS_HEAVIEST_OBJECT(
    world_state_og: WorldState, world_state_mod: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    assert len(attributes) == 0

    if kwargs["current_world_number_of_objects"] < 2:
        raise ImpossibleToAnswer("Not enough objects in the scene.")

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        ["OBJECT-CF"], world_state_mod, min_objects=kwargs["current_world_number_of_objects"]
    )

    timestep = get_random_timestep_from_list(visible_timesteps, question)

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state_mod, timestep
    )

    objects_masses = []

    for obj in iter_objects(world_state_mod):
        obj_state = world_state_mod["simulation"][timestep]["objects"][obj["id"]]

        is_object_visible = (
            obj_state["infov_pixels"] > MIN_VISIBLE_PIXELS
            and obj_state["fov_visibility"] >= VISIBILITY_THRESHOLD
        )

        if is_object_visible:
            objects_masses.append((obj["mass"], obj))

    if len(objects_masses) < 2:
        raise ImpossibleToAnswer("Not enough visible objects in the scene.")

    object_ordered_by_mass = sorted(objects_masses, key=lambda x: x[0], reverse=True)
    heaviest_object_mass, heaviest_visible_object = object_ordered_by_mass[0]
    second_heaviest_object_mass, _ = object_ordered_by_mass[1]

    if (
        heaviest_object_mass - second_heaviest_object_mass
        < THRESHOLD_DIFFERENCE_PERCENTAGE * second_heaviest_object_mass
    ):
        raise ImpossibleToAnswer("No single heaviest object in the scene.")

    presents = [obj["name"] for obj in iter_objects(world_state_mod)]

    labels, correct_idx = create_mc_object_names_from_dataset(
        heaviest_visible_object["name"],
        presents,
        get_all_objects_names(),
        num_answers=4,
    )

    return fill_questions_cf(
        question, labels, correct_idx, world_state_og, world_state_mod, timestep, resolved_attributes
    )


@with_resolved_attributes_cf
def CF_MASS_LIGHTEST_OBJECT(
    world_state_og: WorldState, world_state_mod: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    assert len(attributes) == 0

    if kwargs["current_world_number_of_objects"] < 2:
        raise ImpossibleToAnswer("Not enough objects in the scene.")

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        ["OBJECT-CF"], world_state_mod, min_objects=1
    )

    timestep = get_random_timestep_from_list(visible_timesteps, question)

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state_mod, timestep
    )

    objects_masses = []

    for obj in iter_objects(world_state_mod):
        obj_state = world_state_mod["simulation"][timestep]["objects"][obj["id"]]

        is_object_visible = (
            obj_state["infov_pixels"] > MIN_VISIBLE_PIXELS
            and obj_state["fov_visibility"] >= VISIBILITY_THRESHOLD
        )

        if is_object_visible:
            objects_masses.append((obj["mass"], obj))

    if len(objects_masses) < 2:
        raise ImpossibleToAnswer("Not enough visible objects in the scene.")

    object_ordered_by_mass = sorted(objects_masses, key=lambda x: x[0], reverse=True)
    lightest_object_mass, lightest_visible_object = object_ordered_by_mass[-1]
    second_lightest_object_mass, _ = object_ordered_by_mass[-2]

    if (
        lightest_object_mass - second_lightest_object_mass
        < THRESHOLD_DIFFERENCE_PERCENTAGE * second_lightest_object_mass
    ):
        raise ImpossibleToAnswer("No single lightest object in the scene.")

    presents = [obj["name"] for obj in iter_objects(world_state_mod)]

    labels, correct_idx = create_mc_object_names_from_dataset(
        lightest_visible_object["name"],
        presents,
        get_all_objects_names(),
        num_answers=4,
    )

    return fill_questions_cf(
        question, labels, correct_idx, world_state_og, world_state_mod, timestep, resolved_attributes
    )
