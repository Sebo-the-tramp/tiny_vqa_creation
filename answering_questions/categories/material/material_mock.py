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

from utils.all_objects import get_all_objects_names

from typing import (
    Any,
    Mapping,
    Sequence,
    Tuple,
    Union,
)

from utils.helpers import (
    _distance_between,
    _iter_objects,
    _fill_template,
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

## --- Resolver functions -- ##


@with_resolved_attributes
def F_MATERIALS_COUNTING(
     world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> int:
    assert (
        len(resolved_attributes) == 1
        and "MATERIAL" in resolved_attributes
    )

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
    assert (
        len(resolved_attributes) == 1
        and "OBJECT" in resolved_attributes
    )

    object = resolved_attributes["OBJECT"]["choice"]

    material = "unknown"
    for obj in _iter_objects(world_state):
        if obj["id"] == object["id"]:
            material = obj.get("material", "unknown")
            break

    _fill_template(question, resolved_attributes)

    DATASET = get_all_materials()
    material_present = get_all_materials_in_scene(world_state)

    labels, correct_idx = create_mc_object_names_from_dataset(
        material,  material_present, DATASET
    )

    return question, labels, correct_idx