"""
Mock material reasoning resolvers.

These helpers extract best-effort material answers from the provided world state.
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


from typing import (
    Any,
    Mapping,
    Sequence,
    Tuple,
    Union,
)

from utils.helpers import (
    _iter_objects,
    _fill_template,
    get_object_state_at_timestep,
    get_all_objects_state_at_time,
    _shuffle_array,
    _get_total_timesteps,
    _get_total_images,
)

from .material_helpers import (
    get_all_materials,
    get_all_materials_in_scene,
)

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

    count = 0
    for obj in _iter_objects(world_state):
        if "material" in obj and obj["material"] == material:
            count += 1

    _fill_template(question, resolved_attributes)

    options, correct_idx = create_mc_options_around_gt(
        count, num_answers=4, display_decimals=0, lo=0.0
    )
    labels = uniform_labels(options, integer=True, decimals=0)

    return question, labels, correct_idx


@with_resolved_attributes
def F_MATERIALS_ATTRIBUTE(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> int:
    assert len(resolved_attributes) == 1 and "OBJECT" in resolved_attributes

    object = resolved_attributes["OBJECT"]["choice"]

    material = "unknown"
    for obj in _iter_objects(world_state):
        if obj["id"] == object["id"]:
            material = obj.get("material", "unknown")
            break

    _fill_template(question, resolved_attributes)

    DATASET = get_all_materials()
    material_present = get_all_materials_in_scene(world_state)

    print("material:", material)
    print("material_present:", material_present)
    print("DATASET:", DATASET)

    labels, correct_idx = create_mc_object_names_from_dataset(
        material, material_present, DATASET
    )

    return question, labels, correct_idx


@with_resolved_attributes
def F_DEFORMATION_ATTRIBUTE(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> int:
    assert (
        len(resolved_attributes) == 2
        and "OBJECT" in resolved_attributes
        and "TIME" in resolved_attributes
    )

    object = resolved_attributes["OBJECT"]["choice"]
    time = resolved_attributes["TIME"]["choice"]

    objsect_state = get_object_state_at_timestep(world_state, object["id"], time)
    stress = objsect_state["stress"]

    _fill_template(question, resolved_attributes)

    options, correct_idx = create_mc_options_around_gt(
        stress, num_answers=4, display_decimals=1, lo=0.0
    )
    labels = uniform_labels(options, integer=False, decimals=1)
    labels = [str(label) + " " + DEFORMATION_UNIT for label in labels]

    return question, labels, correct_idx


@with_resolved_attributes
def F_DEFORMATION_COUNTING(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> int:
    assert len(resolved_attributes) == 1 and "TIME" in resolved_attributes

    timestep = resolved_attributes["TIME"]["choice"]
    count = 0

    objects_states = get_all_objects_state_at_time(world_state, timestep=timestep)

    for obj_state in objects_states:
        if "stress" in obj_state and obj_state["stress"] > 0.0:
            count += 1

    _fill_template(question, resolved_attributes)

    options, correct_idx = create_mc_options_around_gt(
        count, num_answers=4, display_decimals=0, lo=0.0
    )
    labels = uniform_labels(options, integer=True, decimals=0)

    return question, labels, correct_idx


@with_resolved_attributes
def F_DEFORMATION_COUNTING_THRESHOLD(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> int:
    assert (
        len(resolved_attributes) == 2
        and "TIME" in resolved_attributes
        and "STRESS-THRESHOLD" in resolved_attributes
    )

    timestep = resolved_attributes["TIME"]["choice"]
    threshold = float(resolved_attributes["STRESS-THRESHOLD"]["choice"])

    objects_state = get_all_objects_state_at_time(world_state, timestep=timestep)

    count = 0
    for obj_id, obj in objects_state.items():
        max_stress = max(
            (state["stress"] for state in obj.get("states", []) if "stress" in state),
            default=0.0,
        )
        if max_stress > threshold:
            count += 1

    _fill_template(question, resolved_attributes)

    options, correct_idx = create_mc_options_around_gt(
        count, num_answers=4, display_decimals=0, lo=0.0
    )
    labels = uniform_labels(options, integer=True, decimals=0)

    return question, labels, correct_idx


@with_resolved_attributes
def F_DEFORMATION_OBJECT_IMAGE_MOST(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> int:
    assert len(resolved_attributes) == 1 and "OBJECT" in resolved_attributes

    object = resolved_attributes["OBJECT"]["choice"]

    # get 4 random states out of 8 possible images and then compare their relative
    # world state
    number_of_images = _get_total_images()

    images = sorted(_shuffle_array([x for x in range(number_of_images)])[:4])
    total_timesteps = _get_total_timesteps()
    timestep_delta = round((total_timesteps / number_of_images) * 0.01, 2)

    iterate_timesteps = [f"{round(i * timestep_delta, 2):.2f}" for i in images]

    max_stress = -1.0  # I think stress can be + or -
    correct_idx = -1

    for i, timestep in enumerate(iterate_timesteps):
        objsect_state = get_object_state_at_timestep(world_state, object["id"], timestep)
        stress = objsect_state["stress"]
        if stress > max_stress:
            max_stress = stress
            correct_idx = i

    # if all have zero stress
    if correct_idx == -1:
        raise ValueError(
            "All images have zero stress, cannot determine the one with most stress"
        )

    assert correct_idx != -1
    _fill_template(question, resolved_attributes)
    options = [f"Image {i}" for i in range(len(iterate_timesteps))]

    # TODO Note here there could be ties, we should handle that better
    # for now we just return the first one with max stress

    return question, options, correct_idx


@with_resolved_attributes
def F_DEFORMATION_ANY_OBJECT_IMAGE_MOST(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> int:
    assert len(resolved_attributes) == 0

    number_of_images = _get_total_images()

    images = sorted(_shuffle_array([x for x in range(number_of_images)])[:4])
    total_timesteps = _get_total_timesteps()
    timestep_delta = round((total_timesteps / number_of_images) * 0.01, 2)

    iterate_timesteps = [f"{round(i * timestep_delta, 2):.2f}" for i in images]

    max_stress = -1.0  # I think stress can be + or -
    correct_idx = -1

    for i, timestep in enumerate(iterate_timesteps):
        for obj in _iter_objects(world_state):
            objsect_state = get_object_state_at_timestep(world_state, obj["id"], timestep)
            stress = objsect_state["stress"]
            if stress > max_stress:
                max_stress = stress
                correct_idx = i

    # if all have zero stress
    if correct_idx == -1:
        raise ValueError(
            "All images have zero stress, cannot determine the one with most stress"
        )

    assert correct_idx != -1
    _fill_template(question, resolved_attributes)
    options = [f"Image {i}" for i in range(len(iterate_timesteps))]

    # TODO Note here there could be ties, we should handle that better
    # for now we just return the first one with max stress

    return question, options, correct_idx
