"""
Mock temporal reasoning resolvers.

These helpers extract best-effort temporal answers from the provided world state.
They operate on lightweight metadata (positions, orientations, region tags, etc.)
and fall back to sensible defaults when information is missing.
"""

from __future__ import annotations

import numpy as np

from utils.config import get_config
from utils.decorators import with_resolved_attributes
from utils.all_objects import get_all_objects_names
from utils.my_exception import ImpossibleToAnswer

from typing import (
    Any,
    Mapping,
    Sequence,
    Tuple,
    Union,
)

from utils.helpers import fill_questions, resolve_attributes_visible_at_timestep

from utils.bin_creation import create_mc_object_names_from_dataset
from categories.persistence.persistence_helpers import (
    get_visibility_mask,
    get_visibility_change,
    get_optimal_timestep_interval,
)


Number = Union[int, float]
Vector = Tuple[float, float, float]
WorldState = Mapping[str, Any]
QuestionPayload = Mapping[str, Any]
Answer = Union[str, float, Vector, Mapping[str, Any], Sequence[str]]

SAMPLING_RATE = get_config()["sampling_rate"]
RENDER_STEP = 1.0 / SAMPLING_RATE
FRAME_INTERLEAVE = 2  # custom only for temporal questions (heuristic)
MIN_PIXELS_VISIBLE = get_config()["min_pixels_visible"]
CLIP_LENGTH = get_config()["clip_length"]


@with_resolved_attributes
def F_PERSISTENCE_OBJECT_PRESENT(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> Sequence[str]:
    """Which object is seen during the frames, but not visible in the last frame?"""

    assert len(attributes) == 0

    visibility_mask, _, _ = get_visibility_mask(world_state)

    changes_in_visibility = get_visibility_change(visibility_mask)
    changes_across_time = np.abs(changes_in_visibility).sum(axis=1)

    object_name = ""
    final_timestep = None

    if sum(changes_across_time >= 1) > 1:
        raise ImpossibleToAnswer("Multiple significant changes detected.")

    elif sum(changes_across_time >= 1) == 0:
        raise ImpossibleToAnswer("No significant changes detected.")

    else:
        object_index = np.where(changes_across_time >= 1)[0][0]
        object_name = world_state["objects"][str(object_index + 1)]["name"]

        obj_changes = changes_in_visibility[object_index]

        disappearance_indices = np.where(obj_changes == -1)[0]

        if len(disappearance_indices) == 0:
            # The object moved (sum > 0), but no -1 found.
            # It must have only appeared (value 1).
            raise ImpossibleToAnswer(
                f"Object '{object_name}' appeared but never disappeared."
            )

        # this modification could be strange but maybe useful, else we could do CLIP_LENGTH//3 for shorter hidden intervals
        first_disappearance_idx = min(disappearance_indices[0] + (FRAME_INTERLEAVE * CLIP_LENGTH // 2), len(world_state["simulation"]) - 1)
        final_timestep = list(world_state["simulation"].keys())[first_disappearance_idx]

    if (
        final_timestep is None
        or object_name == ""
        or first_disappearance_idx < FRAME_INTERLEAVE * CLIP_LENGTH
    ):
        raise ImpossibleToAnswer(
            "Could not determine the disappeared object or final timestep."
        )

    initial_timestep = list(world_state["simulation"].keys())[
        first_disappearance_idx - FRAME_INTERLEAVE * CLIP_LENGTH
    ]

    labels, correct_idx = create_mc_object_names_from_dataset(
        object_name,
        [],
        get_all_objects_names(),
        num_answers=4,
    )

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, final_timestep
    )

    return fill_questions(
        question,
        labels,
        correct_idx,
        world_state,
        final_timestep,
        resolved_attributes,
        initial_timestep=initial_timestep,
    )


# this is also okay
@with_resolved_attributes
def F_PERSISTENCE_OBJECT_DISAPPEAR(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> Sequence[str]:
    return F_PERSISTENCE_OBJECT_PRESENT(
        world_state, question, kwargs["destination_simulation_id_path"]
    )


@with_resolved_attributes
def F_PERSISTENCE_OBJECT_TOTAL_COUNT(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> Sequence[str]:
    """How many objects are there in the last frame in total, including those currently hidden and/or out of frame?"""

    assert len(attributes) == 0

    visibility_mask, _, _ = get_visibility_mask(world_state)
    total_visible_objects = np.sum(visibility_mask, axis=0)

    initial_timestep_index, initial_timestep, final_timestep_index, final_timestep = (
        get_optimal_timestep_interval(world_state)
    )

    if final_timestep_index - initial_timestep_index < CLIP_LENGTH:
        raise ImpossibleToAnswer(
            "Not enough timesteps before visibility drop to answer the question."
        )

    final_timestep = list(world_state["simulation"].keys())[final_timestep_index]
    initial_timestep = list(world_state["simulation"].keys())[initial_timestep_index]

    # this is not the initial count, but the count at the timestep before disappearing
    count_objects_initial = total_visible_objects[initial_timestep_index]

    # balanced options around the initial count
    start = max(0, count_objects_initial - 2)
    shift = abs(count_objects_initial - 2) if count_objects_initial < 2 else 0
    balanced_bins = [
        str(i)
        for i in range(start, count_objects_initial + 2 + shift)
        if i != count_objects_initial
    ]

    labels, correct_idx = create_mc_object_names_from_dataset(
        str(count_objects_initial),
        [],
        balanced_bins,
        num_answers=4,
    )

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, final_timestep
    )

    return fill_questions(
        question,
        labels,
        correct_idx,
        world_state,
        final_timestep,
        resolved_attributes,
        initial_timestep=initial_timestep,
    )


@with_resolved_attributes
def F_PERSISTENCE_OBJECT_TOTAL_COUNT_HIDDEN(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> Sequence[str]:
    """How many objects are present but not visible in the last frame?"""

    assert len(attributes) == 0

    visibility_mask, _, _ = get_visibility_mask(world_state)
    total_visible_objects = np.sum(visibility_mask, axis=0)

    initial_timestep_index, initial_timestep, final_timestep_index, final_timestep = (
        get_optimal_timestep_interval(world_state)
    )

    if final_timestep_index - initial_timestep_index < CLIP_LENGTH:
        raise ImpossibleToAnswer(
            "Not enough timesteps before visibility drop to answer the question."
        )

    final_timestep = list(world_state["simulation"].keys())[final_timestep_index]
    initial_timestep = list(world_state["simulation"].keys())[initial_timestep_index]

    # this is not the initial count, but the count at the timestep before disappearing
    count_objects_initial = total_visible_objects[initial_timestep_index]
    count_objects_final = total_visible_objects[final_timestep_index]

    hidden = count_objects_initial - count_objects_final

    # balanced options around the initial count
    start = max(0, hidden - 2)
    shift = abs(hidden - 2) if hidden < 2 else 0
    balanced_bins = [str(i) for i in range(start, hidden + 2 + shift) if i != hidden]

    labels, correct_idx = create_mc_object_names_from_dataset(
        str(hidden),
        [],
        balanced_bins,
        num_answers=4,
    )

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, final_timestep
    )

    return fill_questions(
        question,
        labels,
        correct_idx,
        world_state,
        final_timestep,
        resolved_attributes,
        initial_timestep=initial_timestep,
    )
