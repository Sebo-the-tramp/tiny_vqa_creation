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

from utils.helpers_cf import (
    get_start_end_timesteps_visible_end    
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
    question: QuestionPayload,
    attributes,
    resolved_attributes,
    **kwargs,
) -> int:
    assert len(attributes) == 1 and "OBJECT" in attributes

    counterfactual_object_id = kwargs['object_moved_id']
    
    # checking if is just visibility at the end timestep
    # cause we have both multi and single object movement counterfactuals
    # multi will check start visibility after else no question will be generated    

    _, timestep_end = \
        get_start_end_timesteps_visible_end(world_state_og, world_state_mod, counterfactual_object_id)    

    mass_obj_og = world_state_og["objects"][counterfactual_object_id]["mass"]
    mass_obj_mod = world_state_mod["objects"][counterfactual_object_id]["mass"]

    if mass_obj_og == mass_obj_mod:
        raise ImpossibleToAnswer("Mass did not change in the counterfactual.")    
    
    options, correct_idx = create_mc_options_around_gt(
        mass_obj_mod, num_answers=4, display_decimals=2, lo=0.0
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
    )


def get_heaviest_object_at_timestep(
    world_state: WorldState,
    timestep: int,
) -> Tuple[Number, Mapping[str, Any]]:
    objects_masses = []

    for obj in iter_objects(world_state):
        if is_object_visible(world_state, obj["id"], timestep):
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

    return heaviest_object_mass, heaviest_visible_object


@with_resolved_attributes_cf
def CF_MASS_HEAVIEST_OBJECT(
    world_state_og: WorldState,
    world_state_mod: WorldState,    
    question: QuestionPayload,
    attributes,
    resolved_attributes,
    **kwargs,
) -> int:
    assert len(attributes) == 0       

    counterfactual_object_id = kwargs['object_moved_id']
    
    # checking if is just visibility at the end timestep
    # cause we have both multi and single object movement counterfactuals
    # multi will check start visibility after else no question will be generated    

    _, timestep_end = \
        get_start_end_timesteps_visible_end(world_state_og, world_state_mod, counterfactual_object_id)    

    _, heaviest_object_og = get_heaviest_object_at_timestep(
        world_state_og,
        timestep_end,
    )

    _, heaviest_object_mod = get_heaviest_object_at_timestep(
        world_state_mod,
        timestep_end,
    )

    # if there is no change in which is the heaviest okay
    # but if the heaviest in both is the one that changed mass then we can consider it valid

    if heaviest_object_og["id"] != heaviest_object_mod["id"]:
        if (
            heaviest_object_og["id"] != counterfactual_object_id
            and heaviest_object_mod["id"] != counterfactual_object_id
        ):
            raise ImpossibleToAnswer(
                " 2 - Heaviest object did not change in the counterfactual."
            )
        
    visible_objects_names_minus_resolved, all_objects_minus_visible_and_non_visible = (
        get_objects_present_and_not_present(
            world_state_mod, timestep_end, [heaviest_object_mod["name"]]
        )
    )

    resolved_attributes['OBJECT'] = {"choice": heaviest_object_mod, "category": "OBJECT"}    

    labels, correct_idx = create_mc_object_names_from_dataset(
        heaviest_object_mod["name"],
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
        resolved_attributes
    )