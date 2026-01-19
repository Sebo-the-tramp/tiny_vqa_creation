"""
Mock material understanding resolvers.

These helpers extract best-effort material answers from the provided world state.
"""

from __future__ import annotations

import math

from typing import (
    Any,
    Mapping,
    Sequence,
    Tuple,
    Union,
)

from utils.my_exception import ImpossibleToAnswer

from utils.config import get_config
from utils.decorators import with_resolved_attributes

from .material_understanding_helpers import get_material_dataset_different_from_target

from utils.helpers import (
    iter_objects,
    fill_questions,
    is_object_visible,
    get_random_timestep_from_list,
    get_objects_present_and_not_present,
    resolve_attributes_visible_at_timestep,
    get_visible_timesteps_for_attributes_min_objects,
)

from utils.bin_creation import (
    uniform_labels,
    create_mc_options_around_gt,
    create_mc_options_around_gt_log,
    create_mc_object_names_from_dataset,
    create_mc_options_around_gt_poisson_ratio,
)

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
MAX_ALLOWED_DIFFERENCE_YOUNGS_MODULUS_LOG = get_config()[
    "max_allowed_difference_youngs_modulus_log"
]
MAX_ALLOWED_DIFFERENCE_POISSON_RATIO = get_config()[
    "max_allowed_difference_poisson_ratio"
]

MIN_LOG_DIFF = get_config()["min_log_difference_youngs_modulus_highest"]

## --- Resolver functions -- ##


@with_resolved_attributes
def F_MASS_OBJECT(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    """Question: What is the mass of the <OBJECT> seen in the image?"""
    assert len(attributes) == 1 and "OBJECT" in attributes

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        attributes, world_state, min_objects=1
    )

    timestep = get_random_timestep_from_list(visible_timesteps, question)

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, timestep
    )

    object = resolved_attributes["OBJECT"]["choice"]

    mass = object["mass"]

    options, correct_idx = create_mc_options_around_gt(
        mass, num_answers=4, display_decimals=2, lo=0.0
    )
    labels = uniform_labels(options, integer=False, decimals=2)
    labels = [str(label) + " kgs" for label in labels]

    return fill_questions(
        question, labels, correct_idx, world_state, timestep, resolved_attributes
    )


@with_resolved_attributes
def F_MASS_HEAVIEST_OBJECT(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    """Question: Which single object seen in the image has the greatest mass?"""
    assert len(attributes) == 0

    if kwargs["current_world_number_of_objects"] < 2:
        raise ImpossibleToAnswer("Not enough objects in the scene.")

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        ["OBJECT"],
        world_state,
        min_objects=min(kwargs["current_world_number_of_objects"], 3),
    )

    timestep = get_random_timestep_from_list(visible_timesteps, question)

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, timestep
    )

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

    visible_objects_names_minus_resolved, all_objects_minus_visible_and_non_visible = (
        get_objects_present_and_not_present(
            world_state, timestep, [heaviest_visible_object["name"]]
        )
    )

    labels, correct_idx = create_mc_object_names_from_dataset(
        heaviest_visible_object["name"],
        visible_objects_names_minus_resolved,
        all_objects_minus_visible_and_non_visible,
        num_answers=4,
    )

    return fill_questions(
        question, labels, correct_idx, world_state, timestep, resolved_attributes
    )


@with_resolved_attributes
def F_MASS_LIGHTEST_OBJECT(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    """Question: Which single object seen in the image has the least mass?"""
    assert len(attributes) == 0

    if kwargs["current_world_number_of_objects"] < 2:
        raise ImpossibleToAnswer("Not enough objects in the scene.")

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        ["OBJECT"],
        world_state,
        min_objects=min(kwargs["current_world_number_of_objects"], 3),
    )

    timestep = get_random_timestep_from_list(visible_timesteps, question)

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, timestep
    )

    objects_masses = []

    for obj in iter_objects(world_state):
        if is_object_visible(world_state, obj["id"], timestep):
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
            world_state, timestep, [lightest_visible_object["name"]]
        )
    )

    labels, correct_idx = create_mc_object_names_from_dataset(
        lightest_visible_object["name"],
        visible_objects_names_minus_resolved,
        all_objects_minus_visible_and_non_visible,
        num_answers=4,
    )

    return fill_questions(
        question, labels, correct_idx, world_state, timestep, resolved_attributes
    )


@with_resolved_attributes
def F_PHYSICS_PROPERTY_DENSITY_OBJECT(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    """Question: What is the average (or effective) density of the <OBJECT> seen in the image?"""
    assert len(attributes) == 1 and "OBJECT"

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        ["OBJECT"], world_state, min_objects=1
    )

    timestep = get_random_timestep_from_list(visible_timesteps, question)

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
def F_PHYSICS_PROPERTY_DENSITY_OBJECT_RELATIVE(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    """Question: Which object seen in the image has the highest effective density?"""
    assert len(attributes) == 0

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        ["OBJECT"], world_state, min_objects=1
    )

    timestep = get_random_timestep_from_list(visible_timesteps, question)

    denser_object = None
    for obj in iter_objects(world_state):
        if is_object_visible(world_state, obj["id"], timestep):
            if (
                denser_object is None
                or obj["props"]["rhos"] > denser_object["props"]["rhos"]
            ):
                denser_object = obj

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, timestep
    )
    if denser_object is None:
        raise ImpossibleToAnswer("No objects found in the scene.")

    visible_objects_names_minus_resolved, all_objects_minus_visible_and_non_visible = (
        get_objects_present_and_not_present(
            world_state, timestep, [denser_object["name"]]
        )
    )

    labels, correct_idx = create_mc_object_names_from_dataset(
        denser_object["name"],
        visible_objects_names_minus_resolved,
        all_objects_minus_visible_and_non_visible,
        num_answers=4,
    )

    return fill_questions(
        question, labels, correct_idx, world_state, timestep, resolved_attributes
    )


@with_resolved_attributes
def F_PHYSICS_PROPERTY_YOUNG_MODULUS_OBJECT(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    """Question: What is the Young's modulus of the <OBJECT> seen in the image?"""
    assert len(attributes) == 1 and "OBJECT"

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        ["OBJECT"], world_state, min_objects=1
    )
    # if we are in a multi-image setting, we need to ensure there are enough frames
    timestep = get_random_timestep_from_list(visible_timesteps, question)

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, timestep
    )

    object = resolved_attributes["OBJECT"]["choice"]

    youngs_modulus = object["props"]["yms"]

    options, correct_idx = create_mc_options_around_gt_log(
        youngs_modulus, num_answers=4, display_decimals=1, lo=0.0, min_threshold=10000
    )
    labels = uniform_labels(options, integer=False, decimals=1)
    labels = [str(label) + " Pa" for label in labels]

    return fill_questions(
        question, labels, correct_idx, world_state, timestep, resolved_attributes
    )


# THE FOLLOWINGS ARE JUST AN EXPEIMENTS #


@with_resolved_attributes
def F_PHYSICS_PROPERTY_YOUNG_MODULUS_OBJECT_SCIENTIFIC_NOTATION(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    """Question: What is the Young's modulus of the <OBJECT> seen in the image, expressed in scientific notation?"""
    assert len(attributes) == 1 and "OBJECT"

    results = F_PHYSICS_PROPERTY_YOUNG_MODULUS_OBJECT(
        world_state, question, kwargs["destination_simulation_id_path"]
    )

    new_results = []

    for task in results:
        question = task[0]
        labels = task[1]
        correct_idx = task[2]
        frames = task[3]
        world_state = task[4]
        resolved_attributes = task[5]

        new_labels = []
        for label in labels:
            value_str = label.replace(" Pa", "")
            value = float(value_str)
            if value == 0:
                new_label = "0 x 10^0 Pa"
            else:
                exponent = int(math.floor(math.log10(abs(value))))
                mantissa = value / (10**exponent)
                new_label = f"{mantissa:.2f}x10^{exponent} Pa"
            new_labels.append(new_label)

        new_results.append(
            [
                question,
                new_labels,
                correct_idx,
                frames,
                world_state,
                resolved_attributes,
            ]
        )

    return new_results


@with_resolved_attributes
def F_PHYSICS_PROPERTY_YOUNG_MODULUS_OBJECT_METRIC_PREFIX(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    """Question: What is the Young's modulus of the <OBJECT> seen in the image, expressed in scientific notation?"""
    assert len(attributes) == 1 and "OBJECT"

    results = F_PHYSICS_PROPERTY_YOUNG_MODULUS_OBJECT(
        world_state, question, kwargs["destination_simulation_id_path"]
    )

    new_results = []

    for task in results:
        question = task[0]
        labels = task[1]
        correct_idx = task[2]
        frames = task[3]
        world_state = task[4]
        resolved_attributes = task[5]

        new_labels = []
        for label in labels:
            value_str = label.replace(" Pa", "")
            value = float(value_str)
            new_value = value / 1e6
            if new_value < 1:
                formatted_value = f"{new_value:.3f}"
            else:
                formatted_value = f"{new_value:.2f}"
            formatted_value = formatted_value.replace(".", ",")
            new_label = f"{formatted_value} MPa"
            new_labels.append(new_label)

        new_results.append(
            [
                question,
                new_labels,
                correct_idx,
                frames,
                world_state,
                resolved_attributes,
            ]
        )

    return new_results


@with_resolved_attributes
def F_PHYSICS_PROPERTY_YOUNG_MODULUS_OBJECT_SIMILAR(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    """Question: Which object seen in the image has a Young's Modulus most similar to that of the <OBJECT>?"""
    assert len(attributes) == 1 and "OBJECT" in attributes

    if kwargs["current_world_number_of_objects"] < 2:
        raise ImpossibleToAnswer("Not enough objects in the scene.")

    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        ["OBJECT"], world_state, min_objects=2
    )
    timestep = get_random_timestep_from_list(visible_timesteps, question)

    resolved = resolve_attributes_visible_at_timestep(attributes, world_state, timestep)
    ref_obj = resolved["OBJECT"]["choice"]
    ref_obj_name =  ref_obj['name']
    ref_yms = ref_obj["props"]["yms"]

    if ref_yms <= 0:
        raise ImpossibleToAnswer("Invalid Young's modulus.")

    similar_objects = []

    for candidate in iter_objects(world_state):
        if candidate["id"] == ref_obj["id"]:
            continue

        cand_yms = candidate["props"]["yms"]
        if cand_yms <= 0:
            continue

        log_diff = abs(math.log10(cand_yms) - math.log10(ref_yms))

        if log_diff <= MAX_ALLOWED_DIFFERENCE_YOUNGS_MODULUS_LOG and is_object_visible(
            world_state, candidate["id"], timestep
        ):
            similar_objects.append(candidate)

    if len(similar_objects) == 0:
        raise ImpossibleToAnswer("No similar object found.")

    if len(similar_objects) > 1:
        raise ImpossibleToAnswer("Multiple similar objects found. Ambiguous.")

    target = similar_objects[0]

    visible_objects_names_minus_resolved, all_objects_minus_visible_and_non_visible = (
        get_objects_present_and_not_present(world_state, timestep, [target["name"], ref_obj_name])
    )

    labels, correct_idx = create_mc_object_names_from_dataset(
        target["name"],
        visible_objects_names_minus_resolved,
        all_objects_minus_visible_and_non_visible,
        num_answers=4,
    )

    return fill_questions(
        question, labels, correct_idx, world_state, timestep, resolved
    )


@with_resolved_attributes
def F_PHYSICS_PROPERTY_YOUNG_MODULUS_OBJECT_SIMILAR_NON_TECHNICAL(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    """Question: Which object seen in the image has a softness most similar to that of the <OBJECT>?"""
    # better to reuse the previous function
    return F_PHYSICS_PROPERTY_YOUNG_MODULUS_OBJECT_SIMILAR(
        world_state, question, kwargs["destination_simulation_id_path"]
    )


@with_resolved_attributes
def F_PHYSICS_PROPERTY_YOUNG_MODULUS_HIGHEST(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    """Question: Which object seen in the image exhibits the highest Young's Modulus?"""
    assert len(attributes) == 0

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        ["OBJECT"], world_state, min_objects=2
    )
    # if we are in a multi-image setting, we need to ensure there are enough frames
    timestep = get_random_timestep_from_list(visible_timesteps, question)

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, timestep
    )

    visible_objects = []

    # 1. First Pass: Collect all valid candidates
    for obj in iter_objects(world_state):
        if is_object_visible(world_state, obj["id"], timestep):
            # Store tuple of (Young's Modulus, Object)
            # Handle 0 or negative modulus edge cases if necessary
            ym = obj["props"]["yms"]
            if ym > 0:
                visible_objects.append((ym, obj))

    # 2. Logic: Sort and Compare
    if not visible_objects:
        raise ImpossibleToAnswer("No visible objects found.")
    else:
        # Sort by Young's Modulus descending (Highest first)
        visible_objects.sort(key=lambda x: x[0], reverse=True)

        best_ym, best_obj = visible_objects[0]

        # If there is only one object, it is the winner
        if len(visible_objects) == 1:
            highest_modulus_object = best_obj
        else:
            second_best_ym = visible_objects[1][0]

            # LOGARITHMIC COMPARISON
            # Check if the best is significantly distinct from the second best
            # abs() not strictly needed since we sorted, but good practice
            log_diff = math.log10(best_ym) - math.log10(second_best_ym)

            if log_diff < MIN_LOG_DIFF:
                raise ImpossibleToAnswer(
                    f"Ambiguous result: Top two objects have similar stiffness "
                    f"(Log Diff: {log_diff:.3f})."
                )
            else:
                highest_modulus_object = best_obj

    visible_objects_names_minus_resolved, all_objects_minus_visible_and_non_visible = (
        get_objects_present_and_not_present(
            world_state, timestep, [highest_modulus_object["name"]]
        )
    )

    labels, correct_idx = create_mc_object_names_from_dataset(
        highest_modulus_object["name"],
        visible_objects_names_minus_resolved,
        all_objects_minus_visible_and_non_visible,
        num_answers=4,
    )

    return fill_questions(
        question, labels, correct_idx, world_state, timestep, resolved_attributes
    )


@with_resolved_attributes
def F_PHYSICS_PROPERTY_YOUNG_MODULUS_HIGHEST_NON_TECHNICAL(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    """Question: Which object seen in the image is the stiffest?"""
    # better to reuse the previous function
    return F_PHYSICS_PROPERTY_YOUNG_MODULUS_HIGHEST(
        world_state, question, kwargs["destination_simulation_id_path"]
    )


def is_close_to_edges(E, edges, tol):
    if E is None or not math.isfinite(E) or E <= 0:
        return True # treat invalid as ambiguous

    # Convert percentage → log10 tolerance
    tol = math.log10(1.0 + tol / 100.0)

    logE = math.log10(E)
    for edge in edges:
        if edge <= 0:
            continue
            if abs(logE - math.log10(edge)) <= tol:
                return True

            return False

@with_resolved_attributes
def F_PHYSICS_PROPERTY_YOUNG_MODULUS_OBJECT_HIGH_LEVEL(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    """Question: Which attribute best describes the <OBJECT> seen in the image in terms of deformability?"""
    assert len(attributes) == 1 and "OBJECT" in attributes

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        ["OBJECT"], world_state, min_objects=1
    )
    # if we are in a multi-image setting, we need to ensure there are enough frames
    timestep = get_random_timestep_from_list(visible_timesteps, question)

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, timestep
    )

    object = resolved_attributes["OBJECT"]["choice"]
    youngs_modulus = object["props"]["yms"]

    E = youngs_modulus  # in Pascals

    if is_close_to_edges(E, (1e8, 1e7, 1e5), 0.20):
        raise ImpossibleToAnswer("Too close to edges!")

    # 1. RIGID (The "High" Histogram Bars)
    # > 100 MPa (10^8)
    # Captures your Metal/Hard Plastic/Wood objects.
    if E >= 1e8:
        correct_idx = 0

    # 2. FLEXIBLE (The "Middle" Histogram Bar)
    # 10 MPa to 100 MPa (10^7 - 10^8)
    # Captures your Rubber/Tough Leather objects.
    elif E >= 1e7:
        correct_idx = 1

    # 3. SOFT (The "Lego Box" Bin)
    # 100 kPa to 10 MPa (10^5 - 10^7)
    # This captures the 500k Lego Box.
    # It distinguishes "Structural Foam" from "Mushy Stuff".
    elif E >= 1e5:
        correct_idx = 2

    # 4. EXTREMELY SOFT (The "Plush Toy" Bin)
    # < 100 kPa (< 10^5)
    # This captures the 60k Plush Toy.
    else:
        correct_idx = 3

    labels = [
        "Rigid (Holds shape perfectly)",
        "Flexible (Bendable but tough)",
        "Soft (Deformable like stiff foam)",
        "Very Soft (No resistance, like a plush toy)",
    ]

    return fill_questions(
        question, labels, correct_idx, world_state, timestep, resolved_attributes
    )


@with_resolved_attributes
def F_PHYSICS_PROPERTY_POISSON_RATIO_OBJECT(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    """Question: What is the Poisson ratio of the <OBJECT> seen in the image?"""
    assert len(attributes) == 1 and "OBJECT"

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        ["OBJECT"], world_state, min_objects=1
    )

    timestep = get_random_timestep_from_list(visible_timesteps, question)

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, timestep
    )

    object = resolved_attributes["OBJECT"]["choice"]

    poisson_ratio = object["props"]["prs"]

    options, correct_idx = create_mc_options_around_gt_poisson_ratio(
        poisson_ratio, num_answers=4, display_decimals=2, lo=0.0, hi=0.5
    )
    labels = uniform_labels(options, integer=False, decimals=2)
    labels = [str(label) for label in labels]

    return fill_questions(
        question, labels, correct_idx, world_state, timestep, resolved_attributes
    )


@with_resolved_attributes
def F_PHYSICS_PROPERTY_POISSON_RATIO_OBJECT_SIMILAR(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    """Question: Which object has a Poisson ratio most similar to that of the <OBJECT> seen in the image?"""
    assert len(attributes) == 1 and "OBJECT"

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        ["OBJECT"], world_state, min_objects=1
    )

    timestep = get_random_timestep_from_list(visible_timesteps, question)

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, timestep
    )

    ref_object = resolved_attributes["OBJECT"]["choice"]
    ref_object_name = resolved_attributes["OBJECT"]["choice"]['name'] 
    poisson_ratio = ref_object["props"]["prs"]

    similar_object = None
    similar_object_count = 0

    for obj in iter_objects(world_state):
        if obj["id"] == ref_object["id"]:
            continue  # skip the same object

        difference = abs(obj["props"]["prs"] - poisson_ratio)

        if difference < MAX_ALLOWED_DIFFERENCE_POISSON_RATIO and is_object_visible(
            world_state, obj["id"], timestep
        ):
            similar_object = obj
            similar_object_count += 1

        if similar_object_count >= 2:
            raise ImpossibleToAnswer(
                "Too many similar objects in the scene. Ambiguous question."
            )

    if similar_object is None:
        raise ImpossibleToAnswer("No similar object found in the scene.")
        # similar_object = {"name": "None of the objects", "props": {"prs": -1}}

    visible_objects_names_minus_resolved, all_objects_minus_visible_and_non_visible = (
        get_objects_present_and_not_present(
            world_state, timestep, [similar_object["name"], ref_object_name]
        )
    )

    labels, correct_idx = create_mc_object_names_from_dataset(
        similar_object["name"],
        visible_objects_names_minus_resolved,
        all_objects_minus_visible_and_non_visible,
        num_answers=4,
    )

    return fill_questions(
        question, labels, correct_idx, world_state, timestep, resolved_attributes
    )


@with_resolved_attributes
def F_PHYSICS_PROPERTY_POISSON_RATIO_OBJECT_SIMILAR_NON_TECHNICAL(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    """Question: Which object acts most like the <OBJECT> seen in the image in terms of how it bulges sideways when squeezed?"""
    # better to reuse the previous function
    return F_PHYSICS_PROPERTY_POISSON_RATIO_OBJECT_SIMILAR(
        world_state, question, kwargs["destination_simulation_id_path"]
    )


@with_resolved_attributes
def F_PHYSICS_PROPERTY_POISSON_RATIO_HIGHEST(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    """Question: Which object seen in the image exhibits the largest Poisson ratio?"""
    assert len(attributes) == 0

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        ["OBJECT"], world_state, min_objects=2
    )

    timestep = get_random_timestep_from_list(visible_timesteps, question)

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, timestep
    )

    highest_poisson_ratio_object = None
    highest_poisson_ratio = -float("inf")
    highest_poisson_ratio_count = 0

    for obj in iter_objects(world_state):
        if obj["props"]["prs"] >= highest_poisson_ratio and is_object_visible(
            world_state, obj["id"], timestep
        ):
            highest_poisson_ratio = obj["props"]["prs"]
            highest_poisson_ratio_object = obj
            highest_poisson_ratio_count += 1

    if highest_poisson_ratio_object is None:
        raise ImpossibleToAnswer("No objects found in the scene.")

    if highest_poisson_ratio_count >= 2:
        raise ImpossibleToAnswer(
            "Too many objects with similar highest Poisson's ratio. Ambiguous question."
        )

    visible_objects_names_minus_resolved, all_objects_minus_visible_and_non_visible = (
        get_objects_present_and_not_present(
            world_state, timestep, [highest_poisson_ratio_object["name"]]
        )
    )

    labels, correct_idx = create_mc_object_names_from_dataset(
        highest_poisson_ratio_object["name"],
        visible_objects_names_minus_resolved,
        all_objects_minus_visible_and_non_visible,
        num_answers=4,
    )

    return fill_questions(
        question, labels, correct_idx, world_state, timestep, resolved_attributes
    )


@with_resolved_attributes
def F_PHYSICS_PROPERTY_POISSON_RATIO_HIGHEST_NON_TECHNICAL(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    """Question: Which object seen in the image bulges out the most when you press on it?"""
    # better to reuse the previous function
    return F_PHYSICS_PROPERTY_POISSON_RATIO_HIGHEST(
        world_state, question, kwargs["destination_simulation_id_path"]
    )


@with_resolved_attributes
def F_PHYSICS_PROPERTY_POISSON_HIGH_LEVEL(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    """Question: If the <OBJECT> seen in the image were compressed vertically, how would its horizontal dimensions change?"""
    assert len(attributes) == 1 and "OBJECT"

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        ["OBJECT"], world_state, min_objects=1
    )

    timestep = get_random_timestep_from_list(visible_timesteps, question)

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, timestep
    )

    object = resolved_attributes["OBJECT"]["choice"]

    poisson_ratio = object["props"]["prs"]

    if poisson_ratio < 0.0:
        correct_idx = 0  # Auxetic
    if poisson_ratio <= 0.1:
        correct_idx = 1  # Porous/Cork-like
    elif poisson_ratio <= 0.4:
        correct_idx = 2  # Standard Solid (Metal/Plastic)
    else:
        correct_idx = 3  # Rubber-like/Incompressible

    labels = [
        "It would contract inwards",  # Distractor
        "It would barely change width",  # < 0.1
        "It would expand sideways a moderate amount",  # 0.1 - 0.4
        "It would bulge out significantly",  # > 0.4
    ]

    return fill_questions(
        question, labels, correct_idx, world_state, timestep, resolved_attributes
    )


@with_resolved_attributes
def F_MATERIAL_IDENTIFICATION_SIMILAR_OBJECT(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    """Question: Which object seen in the image is made of a material most similar to that of the <OBJECT>?"""
    assert len(attributes) == 1 and "OBJECT"

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        ["OBJECT"], world_state, min_objects=1
    )
    timestep = get_random_timestep_from_list(visible_timesteps, question)

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, timestep
    )

    object = resolved_attributes["OBJECT"]["choice"]
    object_name = resolved_attributes["OBJECT"]["choice"]['name'] 
    material = object["description"]["material_group"]

    object_similar = None
    similar_object_count = 0

    for obj in iter_objects(world_state):
        if (
            obj["description"]["material_group"] == material
            and obj["id"] != object["id"]
            and is_object_visible(world_state, obj["id"], timestep)
        ):
            object_similar = obj
            similar_object_count += 1

    if similar_object_count == 0:
        raise ImpossibleToAnswer("No similar object found in the scene.")
        # object_similar = {"name": "None of the objects"}

    if similar_object_count > 1:
        raise ImpossibleToAnswer(
            "Too many similar objects in the scene. Ambiguous question."
        )

    visible_objects_names_minus_resolved, all_objects_minus_visible_and_non_visible = (
        get_objects_present_and_not_present(
            world_state, timestep, [object_similar["name"], object_name]
        )
    )

    options, correct_idx = create_mc_object_names_from_dataset(
        object_similar["name"],
        visible_objects_names_minus_resolved,
        all_objects_minus_visible_and_non_visible,
        num_answers=4,
    )

    return fill_questions(
        question, options, correct_idx, world_state, timestep, resolved_attributes
    )


@with_resolved_attributes
def F_MATERIAL_IDENTIFICATION_OBJECT_LEVEL_1(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    """Question: What material is the <OBJECT> seen in the image made of?"""
    assert len(attributes) == 1 and "OBJECT"

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        ["OBJECT"], world_state, min_objects=1
    )
    timestep = get_random_timestep_from_list(visible_timesteps, question)

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, timestep
    )

    object = resolved_attributes["OBJECT"]["choice"]
    material = object["description"]["material_group"]

    MATERIALS_ALL, target_material_level_1 = get_material_dataset_different_from_target(
        material, target_level=1
    )

    present = []

    options, correct_idx = create_mc_object_names_from_dataset(
        target_material_level_1, present, MATERIALS_ALL, num_answers=4
    )

    return fill_questions(
        question, options, correct_idx, world_state, timestep, resolved_attributes
    )


@with_resolved_attributes
def F_MATERIAL_IDENTIFICATION_OBJECT_LEVEL_2(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    """Question: What material is the <OBJECT> seen in the image made of?"""
    assert len(attributes) == 1 and "OBJECT"

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        ["OBJECT"], world_state, min_objects=1
    )
    timestep = get_random_timestep_from_list(visible_timesteps, question)

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, timestep
    )

    object = resolved_attributes["OBJECT"]["choice"]
    material = object["description"]["material_group"]

    MATERIALS_ALL, target_material_level_2 = get_material_dataset_different_from_target(
        material, target_level=2
    )

    present = []

    options, correct_idx = create_mc_object_names_from_dataset(
        target_material_level_2, present, MATERIALS_ALL, num_answers=4
    )

    return fill_questions(
        question, options, correct_idx, world_state, timestep, resolved_attributes
    )


@with_resolved_attributes
def F_MATERIAL_IDENTIFICATION_OBJECT_LEVEL_3(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    """Question: What material is the <OBJECT> seen in the image made of?"""
    assert len(attributes) == 1 and "OBJECT"

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        ["OBJECT"], world_state, min_objects=1
    )
    timestep = get_random_timestep_from_list(visible_timesteps, question)

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, timestep
    )

    object = resolved_attributes["OBJECT"]["choice"]
    material = object["description"]["material_group"]

    MATERIALS_ALL, target_material_level_3 = get_material_dataset_different_from_target(
        material, target_level=3
    )

    present = []

    for obj in iter_objects(world_state):
        if obj["description"].get("material_group", None) is None:
            print("Patched material for object:", obj["model"])
        present.append(obj["description"].get("material_group", None))

    options, correct_idx = create_mc_object_names_from_dataset(
        target_material_level_3, present, MATERIALS_ALL, num_answers=4
    )

    return fill_questions(
        question, options, correct_idx, world_state, timestep, resolved_attributes
    )
