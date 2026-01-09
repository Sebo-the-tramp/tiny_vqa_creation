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

    visibility_mask, _ = get_visibility_mask(world_state)

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
        first_disappearance_idx = disappearance_indices[0]
        # check if the index after is also not visible else remove cause we are not sure
        visible_obj_t_init = visibility_mask[object_index, first_disappearance_idx]
        visible_obj_t_next = visibility_mask[object_index, first_disappearance_idx + 1]
        if visible_obj_t_init == 0 and visible_obj_t_next != 0:
            # If the object is not visible in both frames, we can't be sure about its disappearance
            raise ImpossibleToAnswer(
                f"Not robust enough to determine disappearance."
            )
        
        first_disappearance_idx += 1  # move to the frame where it is not visible
        final_timestep = list(world_state["simulation"].keys())[first_disappearance_idx]

    # change the frame interleave or maybe not?
    # we can have it -> stochastically will change, and then persistence should not
    # be bound at the granularity of the frames only at object present or not
    curr_frame_interleave = first_disappearance_idx // CLIP_LENGTH

    if (
        final_timestep is None
        or object_name == ""
        or first_disappearance_idx < curr_frame_interleave * CLIP_LENGTH
    ):
        raise ImpossibleToAnswer(
            "Could not determine the disappeared object or final timestep."
        )

    initial_timestep = list(world_state["simulation"].keys())[
        first_disappearance_idx - (curr_frame_interleave * CLIP_LENGTH)
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

    visibility_mask, _ = get_visibility_mask(world_state)

    initial_timestep_index, initial_timestep, final_timestep_index, final_timestep, _, _ = (
        get_optimal_timestep_interval(world_state)
    )

    if (final_timestep_index + 1) - initial_timestep_index < CLIP_LENGTH:
        raise ImpossibleToAnswer(
            "Not enough timesteps before visibility drop to answer the question."
        )

    # this is not the initial count, but the count at the timestep before disappearing
    # you cannot just count the objects at the beginning you should also take into account from beginning to end, max, and also based 
    # on how many frames you are skipping -> KEEP GOING FROM HERE!
    curr_frame_interleave = (final_timestep_index + 1 - initial_timestep_index) // CLIP_LENGTH
    visibility_mask_valid = visibility_mask[:, initial_timestep_index:final_timestep_index + 1][:, ::curr_frame_interleave]
    objects_seen_at_least_once_mask = np.any(visibility_mask_valid, axis=1)
    total_unique_objects_seen = np.sum(objects_seen_at_least_once_mask)

    # balanced options around the initial count
    start = max(0, total_unique_objects_seen - 2)
    shift = abs(total_unique_objects_seen - 2) if total_unique_objects_seen < 2 else 0
    balanced_bins = [
        str(i)
        for i in range(start, total_unique_objects_seen + 2 + shift)
        if i != total_unique_objects_seen
    ]

    labels, correct_idx = create_mc_object_names_from_dataset(
        str(total_unique_objects_seen),
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

    visibility_mask, _ = get_visibility_mask(world_state)
    total_visible_objects = np.sum(visibility_mask, axis=0)

    initial_timestep_index, initial_timestep, final_timestep_index, final_timestep, _, _ = (
        get_optimal_timestep_interval(world_state)
    )

    if (final_timestep_index + 1) - initial_timestep_index < CLIP_LENGTH:
        raise ImpossibleToAnswer(
            "Not enough timesteps before visibility drop to answer the question."
        )

    # this is not the initial count, but the count at the timestep before disappearing
    # you cannot just count the objects at the beginning you should also take into account from beginning to end, max, and also based 
    # on how many frames you are skipping -> KEEP GOING FROM HERE!
    curr_frame_interleave = (final_timestep_index + 1 - initial_timestep_index) // CLIP_LENGTH
    visibility_mask_valid = visibility_mask[:, initial_timestep_index:final_timestep_index + 1][:, ::curr_frame_interleave]
    objects_seen_at_least_once_mask = np.any(visibility_mask_valid, axis=1)
    total_unique_objects_seen = np.sum(objects_seen_at_least_once_mask)

    count_objects_final = total_visible_objects[final_timestep_index]

    hidden = total_unique_objects_seen - count_objects_final

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
