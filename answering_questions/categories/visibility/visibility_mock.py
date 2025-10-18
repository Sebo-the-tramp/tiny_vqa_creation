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
    uniform_labels,
)

from utils.helpers import _iter_objects, _fill_template, get_object_state_at_timestep

## --- Resolver functions -- ##


@with_resolved_attributes
def F_VISIBILITY_COUNTING(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> int:
    assert len(resolved_attributes) == 1 and "TIME" in resolved_attributes

    visible_count = 0
    timestep = resolved_attributes["TIME"]["choice"]

    for obj in _iter_objects(world_state):
        obj_state = get_object_state_at_timestep(world_state, obj["id"], timestep)
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
        and "OBJECT-CATEGORY" in resolved_attributes
    )

    category = resolved_attributes["OBJECT-CATEGORY"]["choice"]
    timestep = resolved_attributes["TIME"]["choice"]

    visible_count = 0

    for obj in _iter_objects(world_state):
        if obj.get("category", "") != category:
            continue
        obj_state = get_object_state_at_timestep(world_state, obj["id"], timestep)
        if obj_state is not None and obj_state.get("is_visible_from_camera", False):
            visible_count += 1

    _fill_template(question, resolved_attributes)

    options, correct_idx = create_mc_options_around_gt(
        visible_count, num_answers=4, display_decimals=0, lo=0.0
    )
    labels = uniform_labels(options, integer=True, decimals=0)

    return question, labels, correct_idx


@with_resolved_attributes
def F_VISIBILITY_COUNTING_OBJECTS_HIDDEN_SCENE(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> int:
    # TODO with amodal mask

    return "", ["A.", "B.", "C.", "D."], 0


@with_resolved_attributes
def F_VISIBILITY_COUNTING_OBJECTS_ALWAYS_VISIBEL(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> int:
    # assuming always all objects are visible
    count_object_visible = list(_iter_objects(world_state))

    for timestep, value in world_state["simulation_steps"].items():
        for obj in count_object_visible:
            object_state = get_object_state_at_timestep(
                world_state, obj["id"], timestep
            )
            if not object_state["is_visible_from_camera"]:
                count_object_visible.remove(obj)
                if count_object_visible == []:
                    break

    _fill_template(question, resolved_attributes)

    options, correct_idx = create_mc_options_around_gt(
        len(count_object_visible), num_answers=4, display_decimals=0, lo=0.0
    )
    labels = uniform_labels(options, integer=True, decimals=0)

    return question, labels, correct_idx
