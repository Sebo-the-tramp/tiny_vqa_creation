"""
Mock visibility reasoning resolvers.

These helpers extract best-effort visibility answers from the provided world state.
They operate on lightweight metadata (positions, orientations, region tags, etc.)
and fall back to sensible defaults when information is missing.
"""

from __future__ import annotations


from typing import (
    Any,
    Mapping,
    Sequence,
    Tuple,
    Union,
)


Number = Union[int, float]
Vector = Tuple[float, float, float]
WorldState = Mapping[str, Any]
QuestionPayload = Mapping[str, Any]
Answer = Union[str, float, Vector, Mapping[str, Any], Sequence[str]]

from utils.decorators import with_resolved_attributes

from utils.bin_creation import (
    create_mc_options_around_gt,
    create_mc_object_names_from_dataset,
    uniform_labels,
)

from utils.helpers import (    
    _iter_objects,
    _fill_template,
    get_object_state_at_timestep
)

## --- Resolver functions -- ##

@with_resolved_attributes
def F_VISIBILITY_COUNTING(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> int:
    assert (
        len(resolved_attributes) == 1
        and "TIME" in resolved_attributes
    )

    visible_count = 0
    timestep = resolved_attributes["TIME"]["choice"]

    for obj in _iter_objects(world_state):
        obj_state = get_object_state_at_timestep(world_state, obj['id'], timestep)
        if obj_state is not None and obj_state.get("is_visible_from_camera", False):
            visible_count += 1

    _fill_template(question, resolved_attributes)

    options, correct_idx = create_mc_options_around_gt(
        visible_count, num_answers=4, display_decimals=0, lo=0.0
    )
    labels = uniform_labels(options, integer=True, decimals=0)

    return question, labels, correct_idx


@with_resolved_attributes
def F_VISIBILITY_COUNTING_CATEGORY(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> int:
    assert (
        len(resolved_attributes) == 2
        and "TIME" in resolved_attributes
        and "CATEGORY" in resolved_attributes
    )

    category = resolved_attributes["CATEGORY"]["choice"]
    timestep = resolved_attributes["TIME"]["choice"]

    visible_count = 0

    for obj in _iter_objects(world_state):
        if obj.get("category", "") != category:
            continue
        obj_state = get_object_state_at_timestep(world_state, obj['id'], timestep)
        if obj_state is not None and obj_state.get("is_visible_from_camera", False):
            visible_count += 1

    _fill_template(question, resolved_attributes)

    options, correct_idx = create_mc_options_around_gt(
        visible_count, num_answers=4, display_decimals=0, lo=0.0
    )
    labels = uniform_labels(options, integer=True, decimals=0)

    return question, labels, correct_idx
