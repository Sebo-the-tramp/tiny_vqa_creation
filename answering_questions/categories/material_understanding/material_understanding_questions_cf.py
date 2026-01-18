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


from utils.config import get_config

from utils.my_exception import ImpossibleToAnswer

from utils.helpers import (
    iter_objects,
    fill_questions_cf,
    is_object_visible,
    get_timestep_from_idx,
    get_objects_present_and_not_present,
    resolve_attributes_visible_at_timestep,
)

from utils.bin_creation import (
    create_mc_options_around_gt,
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
MIN_VISIBLE_PIXELS = get_config()["min_pixels_visible"]
VISIBILITY_THRESHOLD = get_config()["visibility_threshold"]
THRESHOLD_DIFFERENCE_PERCENTAGE = get_config()["threshold_difference_percentage"]
MAX_ALLOWED_DIFFERENCE_POISSON_RATIO = get_config()[
    "max_allowed_difference_poisson_ratio"
]


## --- Resolver functions -- ##


@with_resolved_attributes_cf
def CF_MASS_OBJECT(
    world_state_og: WorldState,
    world_state_mod: WorldState,
    answer_list_original_data_cf: dict,
    question: QuestionPayload,
    attributes,
    **kwargs,
) -> int:
    assert len(attributes) == 1 and "OBJECT" in attributes

    timestep_end_index = int(
        answer_list_original_data_cf[0][3][-1]
    )  # this has to be the image to get the question
    timestep_end = get_timestep_from_idx(timestep_end_index)

    timestep_start_index = int(
        answer_list_original_data_cf[0][3][0]
    )  # this has to be the image to get the question
    timestep_start = get_timestep_from_idx(timestep_start_index)

    # check if the particular question asks something outside of simulation_og
    if timestep_end_index > len(world_state_og["simulation"]) - 1:
        raise ImpossibleToAnswer("Question refers to future timestep.")

    object_id = answer_list_original_data_cf[0][5]["OBJECT"]["choice"]["id"]

    object = world_state_og["objects"][object_id]

    if not is_object_visible(
        world_state=world_state_og, obj_id=object_id, timestep=timestep_end
    ):
        raise ImpossibleToAnswer("Object is not visible.")

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state_mod, timestep_end
    )

    mass = object["mass"]

    options, correct_idx = create_mc_options_around_gt(
        mass, num_answers=4, display_decimals=2, lo=0.0
    )
    labels = uniform_labels(options, integer=False, decimals=2)
    labels = [str(label) + " kgs" for label in labels]

    return fill_questions_cf(
        question,
        labels,
        correct_idx,
        world_state_og,
        world_state_mod,
        timestep_end,
        resolved_attributes,
        initial_timestep=timestep_start,
    )


@with_resolved_attributes_cf
def CF_MASS_HEAVIEST_OBJECT(
    world_state_og: WorldState,
    world_state_mod: WorldState,
    answer_list_original_data_cf: dict,
    question: QuestionPayload,
    attributes,
    **kwargs,
) -> int:
    assert len(attributes) == 0

    if kwargs["current_world_number_of_objects"] < 2:
        raise ImpossibleToAnswer("Not enough objects in the scene.")

    timestep_end_index = int(
        answer_list_original_data_cf[0][3][-1]
    )  # this has to be the image to get the question
    timestep_end = get_timestep_from_idx(timestep_end_index)

    timestep_start_index = int(
        answer_list_original_data_cf[0][3][0]
    )  # this has to be the image to get the question
    timestep_start = get_timestep_from_idx(timestep_start_index)

    # check if the particular question asks something outside of simulation_og
    if timestep_end_index > len(world_state_og["simulation"]) - 1:
        raise ImpossibleToAnswer("Question refers to future timestep.")

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state_mod, timestep_end
    )

    objects_masses = []

    for obj in iter_objects(world_state_mod):
        if is_object_visible(
            world_state=world_state_mod, obj_id=obj["id"], timestep=timestep_end
        ):
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

    visible_objects_names_minus_resolved, all_objects_minus_visible_and_non_visible = (
        get_objects_present_and_not_present(
            world_state_mod, timestep_end, [heaviest_visible_object["name"]]
        )
    )

    labels, correct_idx = create_mc_object_names_from_dataset(
        heaviest_visible_object["name"],
        visible_objects_names_minus_resolved,
        all_objects_minus_visible_and_non_visible,
        num_answers=4,
    )

    return fill_questions_cf(
        question,
        labels,
        correct_idx,
        world_state_og,
        world_state_mod,
        timestep_end,
        resolved_attributes,
        initial_timestep=timestep_start,
    )


@with_resolved_attributes_cf
def CF_MASS_LIGHTEST_OBJECT(
    world_state_og: WorldState,
    world_state_mod: WorldState,
    answer_list_original_data_cf: dict,
    question: QuestionPayload,
    attributes,
    **kwargs,
) -> int:
    assert len(attributes) == 0

    if kwargs["current_world_number_of_objects"] < 2:
        raise ImpossibleToAnswer("Not enough objects in the scene.")

    timestep_end_index = int(
        answer_list_original_data_cf[0][3][-1]
    )  # this has to be the image to get the question
    timestep_end = get_timestep_from_idx(timestep_end_index)

    timestep_start_index = int(
        answer_list_original_data_cf[0][3][0]
    )  # this has to be the image to get the question
    timestep_start = get_timestep_from_idx(timestep_start_index)

    # check if the particular question asks something outside of simulation_og
    if timestep_end_index > len(world_state_og["simulation"]) - 1:
        raise ImpossibleToAnswer("Question refers to future timestep.")

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state_mod, timestep_end
    )

    objects_masses = []

    for obj in iter_objects(world_state_mod):
        if is_object_visible(
            world_state=world_state_mod, obj_id=obj["id"], timestep=timestep_end
        ):
            objects_masses.append((obj["mass"], obj))

    if len(objects_masses) < 2:
        raise ImpossibleToAnswer("Not enough visible objects in the scene.")

    object_ordered_by_mass = sorted(objects_masses, key=lambda x: x[0], reverse=True)
    lightest_object_mass, lightest_visible_object = object_ordered_by_mass[-1]
    second_lightest_object_mass, _ = object_ordered_by_mass[-2]

    if (
        second_lightest_object_mass - lightest_object_mass
        < THRESHOLD_DIFFERENCE_PERCENTAGE * second_lightest_object_mass
    ):
        raise ImpossibleToAnswer("No single lightest object in the scene.")

    visible_objects_names_minus_resolved, all_objects_minus_visible_and_non_visible = (
        get_objects_present_and_not_present(
            world_state_mod, timestep_end, [lightest_visible_object["name"]]
        )
    )

    labels, correct_idx = create_mc_object_names_from_dataset(
        lightest_visible_object["name"],
        visible_objects_names_minus_resolved,
        all_objects_minus_visible_and_non_visible,
        num_answers=4,
    )

    return fill_questions_cf(
        question,
        labels,
        correct_idx,
        world_state_og,
        world_state_mod,
        timestep_end,
        resolved_attributes,
        initial_timestep=timestep_start,
    )
