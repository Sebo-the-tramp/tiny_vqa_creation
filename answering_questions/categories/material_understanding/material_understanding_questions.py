"""
Mock material understanding resolvers.

These helpers extract best-effort material answers from the provided world state.
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

from utils.my_exception import ImpossibleToAnswer

from utils.all_objects import get_all_objects_names

from utils.helpers import (
    fill_questions,
    iter_objects,
    get_random_timestep_from_list,
    resolve_attributes_visible_at_timestep,
    get_visible_timesteps_for_attributes_min_objects,
    is_object_visible,
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

import math

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

## --- Resolver functions -- ##


@with_resolved_attributes
def F_MASS_OBJECT(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
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
        obj_state = world_state["simulation"][timestep]["objects"][obj["id"]]

        if is_object_visible(obj_state):
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
        obj_state = world_state["simulation"][timestep]["objects"][obj["id"]]

        if is_object_visible(obj_state):
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
def F_PHYSICS_PROPERTY_DENSITY_OBJECT(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
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
    assert len(attributes) == 0

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        ["OBJECT"], world_state, min_objects=1
    )

    timestep = get_random_timestep_from_list(visible_timesteps, question)

    denser_object = None
    for object in iter_objects(world_state):
        obj_state = world_state["simulation"][timestep]["objects"][object["id"]]

        if is_object_visible(obj_state):
            if (
                denser_object is None
                or object["props"]["rhos"] > denser_object["props"]["rhos"]
            ):
                denser_object = object

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, timestep
    )
    if denser_object is None:
        raise ImpossibleToAnswer("No objects found in the scene.")

    presents = [obj["name"] for obj in iter_objects(world_state)]
    labels, correct_idx = create_mc_object_names_from_dataset(
        denser_object["name"],
        presents,
        get_all_objects_names(),
        num_answers=4,
    )

    return fill_questions(
        question, labels, correct_idx, world_state, timestep, resolved_attributes
    )


@with_resolved_attributes
def F_PHYSICS_PROPERTY_YOUNG_MODULUS_OBJECT(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
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


@with_resolved_attributes
def F_PHYSICS_PROPERTY_YOUNG_MODULUS_OBJECT_SIMILAR(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    assert len(attributes) == 1 and "OBJECT" in attributes

    if kwargs["current_world_number_of_objects"] < 2:
        raise ImpossibleToAnswer("Not enough objects in the scene.")

    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        ["OBJECT"], world_state, min_objects=2
    )
    timestep = get_random_timestep_from_list(visible_timesteps, question)

    resolved = resolve_attributes_visible_at_timestep(attributes, world_state, timestep)
    ref_obj = resolved["OBJECT"]["choice"]
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
        cand_state = world_state["simulation"][timestep]["objects"][candidate["id"]]

        if log_diff <= MAX_ALLOWED_DIFFERENCE_YOUNGS_MODULUS_LOG and is_object_visible(
            cand_state
        ):
            similar_objects.append(candidate)

    if len(similar_objects) == 0:
        raise ImpossibleToAnswer("No similar object found.")

    if len(similar_objects) > 1:
        raise ImpossibleToAnswer("Multiple similar objects found. Ambiguous.")

    target = similar_objects[0]

    presents = [
        obj["name"] for obj in iter_objects(world_state) if obj["id"] != ref_obj["id"]
    ]
    labels, correct_idx = create_mc_object_names_from_dataset(
        target["name"],
        presents,
        get_all_objects_names(),
        num_answers=4,
    )

    return fill_questions(
        question, labels, correct_idx, world_state, timestep, resolved
    )


@with_resolved_attributes
def F_PHYSICS_PROPERTY_YOUNG_MODULUS_OBJECT_SIMILAR_NON_TECHNICAL(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    # better to reuse the previous function
    return F_PHYSICS_PROPERTY_YOUNG_MODULUS_OBJECT_SIMILAR(
        world_state, question, kwargs["destination_simulation_id_path"]
    )


@with_resolved_attributes
def F_PHYSICS_PROPERTY_YOUNG_MODULUS_HIGHEST(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    assert len(attributes) == 0

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        ["OBJECT"], world_state, min_objects=1
    )
    # if we are in a multi-image setting, we need to ensure there are enough frames
    timestep = get_random_timestep_from_list(visible_timesteps, question)

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, timestep
    )

    # search for another object with similar young's modulus
    highest_modulus_object = None
    highest_modulus = -float("inf")
    highest_modulus_count = 0
    MIN_DIFFERENCE_YOUNGS_MODULUS_PERCENTAGE = (
        0.1  # to avoid selecting objects with very similar modulus
    )

    for obj in iter_objects(world_state):
        obj_state = world_state["simulation"][timestep]["objects"][obj["id"]]

        is_object_visible = (
            obj_state["infov_pixels"] > MIN_VISIBLE_PIXELS
            and obj_state["fov_visibility"] >= VISIBILITY_THRESHOLD
        )

        if (
            obj["props"]["yms"]
            > highest_modulus
            + MIN_DIFFERENCE_YOUNGS_MODULUS_PERCENTAGE * highest_modulus
            and is_object_visible
        ):
            highest_modulus = obj["props"]["yms"]
            highest_modulus_object = obj
            highest_modulus_count += 1

    if highest_modulus_count > 1:
        raise ImpossibleToAnswer(
            "Too many objects with similar highest Young's modulus. Ambiguous question."
        )

    if highest_modulus_object is None:
        raise ImpossibleToAnswer("No objects found in the scene.")

    presents = [obj["name"] for obj in iter_objects(world_state)]
    labels, correct_idx = create_mc_object_names_from_dataset(
        highest_modulus_object["name"],
        presents,
        get_all_objects_names(),
        num_answers=4,
    )

    return fill_questions(
        question, labels, correct_idx, world_state, timestep, resolved_attributes
    )


@with_resolved_attributes
def F_PHYSICS_PROPERTY_YOUNG_MODULUS_HIGHEST_NON_TECHNICAL(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    # better to reuse the previous function
    return F_PHYSICS_PROPERTY_YOUNG_MODULUS_HIGHEST(
        world_state, question, kwargs["destination_simulation_id_path"]
    )


@with_resolved_attributes
def F_PHYSICS_PROPERTY_YOUNG_MODULUS_BEHAVIOR(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
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
    E_steel = 200e9
    r = E / E_steel

    # Revised thresholds for better semantic mapping
    if r > 0.3:
        correct_idx = 0  # Metal/Glass Tier (> 60 GPa)
    elif r > 0.01:
        correct_idx = 1  # Structural Tier (Wood, Hard Plastic, Bone) (> 2 GPa)
    elif r > 0.001:
        correct_idx = 2  # Flexible Plastic Tier (Soft Polyethylene) (> 200 MPa)
    else:
        correct_idx = 3  # Soft/Rubbery Tier (< 200 MPa)

    labels = [
        "It would behave like metal (Extremely Rigid)",
        "It would behave like wood or hard plastic (Rigid)",
        "It would behave like flexible plastic (Bendable)",
        "It would behave like rubber or foam (Squishy)",
    ]

    return fill_questions(
        question, labels, correct_idx, world_state, timestep, resolved_attributes
    )


@with_resolved_attributes
def F_PHYSICS_PROPERTY_YOUNG_MODULUS_OBJECT_HIGH_LEVEL(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
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
    poisson_ratio = ref_object["props"]["prs"]

    similar_object = None
    similar_object_count = 0

    for obj in iter_objects(world_state):
        if obj["id"] == ref_object["id"]:
            continue  # skip the same object

        obj_state = world_state["simulation"][timestep]["objects"][obj["id"]]

        is_object_visible = (
            obj_state["infov_pixels"] > MIN_VISIBLE_PIXELS
            and obj_state["fov_visibility"] >= VISIBILITY_THRESHOLD
        )

        difference = abs(obj["props"]["prs"] - poisson_ratio)

        if difference < MAX_ALLOWED_DIFFERENCE_POISSON_RATIO and is_object_visible:
            similar_object = obj
            similar_object_count += 1

        if similar_object_count >= 2:
            raise ImpossibleToAnswer(
                "Too many similar objects in the scene. Ambiguous question."
            )

    if similar_object is None:
        raise ImpossibleToAnswer("No similar object found in the scene.")
        # similar_object = {"name": "None of the objects", "props": {"prs": -1}}

    presents = [
        obj["name"]
        for obj in iter_objects(world_state)
        if obj["id"] != ref_object["id"]
    ]
    labels, correct_idx = create_mc_object_names_from_dataset(
        similar_object["name"],
        presents,
        get_all_objects_names(),
        num_answers=4,
    )

    return fill_questions(
        question, labels, correct_idx, world_state, timestep, resolved_attributes
    )


@with_resolved_attributes
def F_PHYSICS_PROPERTY_POISSON_RATIO_OBJECT_SIMILAR_NON_TECHNICAL(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    # better to reuse the previous function
    return F_PHYSICS_PROPERTY_POISSON_RATIO_OBJECT_SIMILAR(
        world_state, question, kwargs["destination_simulation_id_path"]
    )


@with_resolved_attributes
def F_PHYSICS_PROPERTY_POISSON_RATIO_HIGHEST(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    assert len(attributes) == 0

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        ["OBJECT"], world_state, min_objects=1
    )

    timestep = get_random_timestep_from_list(visible_timesteps, question)

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, timestep
    )

    highest_poisson_ratio_object = None
    highest_poisson_ratio = -float("inf")
    highest_poisson_ratio_count = 0

    for obj in iter_objects(world_state):
        obj_state = world_state["simulation"][timestep]["objects"][obj["id"]]

        is_object_visible = (
            obj_state["infov_pixels"] > MIN_VISIBLE_PIXELS
            and obj_state["fov_visibility"] >= VISIBILITY_THRESHOLD
        )

        if obj["props"]["prs"] >= highest_poisson_ratio and is_object_visible:
            highest_poisson_ratio = obj["props"]["prs"]
            highest_poisson_ratio_object = obj
            highest_poisson_ratio_count += 1

    if highest_poisson_ratio_object is None:
        raise ImpossibleToAnswer("No objects found in the scene.")

    if highest_poisson_ratio_count >= 2:
        raise ImpossibleToAnswer(
            "Too many objects with similar highest Poisson's ratio. Ambiguous question."
        )

    presents = [obj["name"] for obj in iter_objects(world_state)]
    labels, correct_idx = create_mc_object_names_from_dataset(
        highest_poisson_ratio_object["name"],
        presents,
        get_all_objects_names(),
        num_answers=4,
    )

    return fill_questions(
        question, labels, correct_idx, world_state, timestep, resolved_attributes
    )


@with_resolved_attributes
def F_PHYSICS_PROPERTY_POISSON_RATIO_HIGHEST_NON_TECHNICAL(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    # better to reuse the previous function
    return F_PHYSICS_PROPERTY_POISSON_RATIO_HIGHEST(
        world_state, question, kwargs["destination_simulation_id_path"]
    )


@with_resolved_attributes
def F_PHYSICS_PROPERTY_POISSON_HIGH_LEVEL(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
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

    present = []

    object_similar = None
    similar_object_count = 0

    for obj in iter_objects(world_state):
        obj_state = world_state["simulation"][timestep]["objects"][obj["id"]]

        if (
            obj["description"]["material_group"] == material
            and obj["id"] != object["id"]
            and is_object_visible(obj_state)
        ):
            object_similar = obj
            similar_object_count += 1

    present = []
    for obj in iter_objects(world_state):
        if object_similar is not None and obj["id"] == object_similar["id"]:
            continue  # skip the similar object
        present.append(obj["name"])

    if similar_object_count == 0:
        raise ImpossibleToAnswer("No similar object found in the scene.")
        # object_similar = {"name": "None of the objects"}

    if similar_object_count > 1:
        raise ImpossibleToAnswer(
            "Too many similar objects in the scene. Ambiguous question."
        )

    options, correct_idx = create_mc_object_names_from_dataset(
        object_similar["name"], present, get_all_objects_names(), num_answers=4
    )

    return fill_questions(
        question, options, correct_idx, world_state, timestep, resolved_attributes
    )


@with_resolved_attributes
def F_MATERIAL_IDENTIFICATION_OBJECT_LEVEL_1(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
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
