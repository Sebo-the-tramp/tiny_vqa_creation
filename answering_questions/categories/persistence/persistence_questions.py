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
from utils.my_exception import ImpossibleToAnswer

from typing import (
    Any,
    Mapping,
    Sequence,
    Tuple,
    Union,
)

from utils.helpers import (
    fill_questions,
    get_timestep_from_idx,
    get_visibility_mask_soft,
    get_objects_present_and_not_present,
    resolve_attributes_visible_at_timestep,
)

from utils.bin_creation import create_mc_object_names_from_dataset
from categories.persistence.persistence_helpers import (
    get_maximum_windows_for_each_object,
    choose_best_window_object_id,
    has_min_consecutive_visibility,
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


def _count_objects_with_min_consecutive_visibility(
    visibility_mask_window: np.ndarray,
) -> int:
    return int(
        sum(
            has_min_consecutive_visibility(row, 0, len(row) - 1)
            for row in visibility_mask_window
        )
    )


@with_resolved_attributes
def F_PERSISTENCE_OBJECT_PRESENT(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> Sequence[str]:
    """Question: Which object is seen during the frames, but not visible in the last frame?"""

    assert len(attributes) == 0

    object_proposed = get_maximum_windows_for_each_object(world_state)

    # check fo unique object that appears and then disappears for time interval
    chosen_object_id = choose_best_window_object_id(world_state, object_proposed)

    if chosen_object_id is None:
        raise ImpossibleToAnswer("No object found that appears and then disappears.")

    final_timestep_index = object_proposed[str(chosen_object_id)][3]
    initial_timestep_index = object_proposed[str(chosen_object_id)][2]

    if final_timestep_index == -1 or initial_timestep_index == -1:
        raise ImpossibleToAnswer("No object found that appears and then disappears.")

    if final_timestep_index < initial_timestep_index:
        raise ImpossibleToAnswer("No object found that appears and then disappears.")

    final_timestep = get_timestep_from_idx(final_timestep_index)
    initial_timestep = get_timestep_from_idx(initial_timestep_index)

    obj_name = world_state["objects"][str(chosen_object_id)]["name"]

    _, all_objects_minus_visible_and_non_visible = get_objects_present_and_not_present(
        world_state, final_timestep, [obj_name]
    )

    labels, correct_idx = create_mc_object_names_from_dataset(
        obj_name,
        [],
        all_objects_minus_visible_and_non_visible,
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
    """Question: Which object disappears and does not reappear in the last frame?"""
    return F_PERSISTENCE_OBJECT_PRESENT(
        world_state, question, kwargs["destination_simulation_id_path"]
    )


@with_resolved_attributes
def F_PERSISTENCE_OBJECT_TOTAL_COUNT(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> Sequence[str]:
    """Question: How many objects are there in the last frame in total, including those currently hidden and/or out of frame?"""

    assert len(attributes) == 0

    # order for longest window
    visibility_mask, _ = get_visibility_mask_soft(world_state)
    object_proposed = get_maximum_windows_for_each_object(world_state)
    chosen_object_id = choose_best_window_object_id(world_state, object_proposed)

    if chosen_object_id is None:
        raise ImpossibleToAnswer("No object found that appears and then disappears.")

    final_timestep_index = object_proposed[str(chosen_object_id)][3]
    initial_timestep_index = object_proposed[str(chosen_object_id)][2]

    final_timestep = get_timestep_from_idx(final_timestep_index)
    initial_timestep = get_timestep_from_idx(initial_timestep_index)

    if final_timestep_index == -1 or initial_timestep_index == -1:
        raise ImpossibleToAnswer("No object found that appears and then disappears.")

    if final_timestep_index < initial_timestep_index:
        raise ImpossibleToAnswer("No object found that appears and then disappears.")

    curr_frame_interleave = ((final_timestep_index) - initial_timestep_index) // (
        CLIP_LENGTH - 1
    )  # there seems to be a problem here
    visibility_objects_window = visibility_mask[
        :, initial_timestep_index : final_timestep_index + 1
    ][:, ::curr_frame_interleave]
    total_unique_objects_seen = _count_objects_with_min_consecutive_visibility(
        visibility_objects_window
    )

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
    """Question: How many objects are present but not visible in the last frame?"""

    assert len(attributes) == 0

    # order for longest window
    visibility_mask, _ = get_visibility_mask_soft(world_state)
    object_proposed = get_maximum_windows_for_each_object(world_state)
    chosen_object_id = choose_best_window_object_id(world_state, object_proposed)

    if chosen_object_id is None:
        raise ImpossibleToAnswer("No object found that appears and then disappears.")

    final_timestep_index = object_proposed[str(chosen_object_id)][3]
    initial_timestep_index = object_proposed[str(chosen_object_id)][2]

    final_timestep = get_timestep_from_idx(final_timestep_index)
    initial_timestep = get_timestep_from_idx(initial_timestep_index)

    if final_timestep_index == -1 or initial_timestep_index == -1:
        raise ImpossibleToAnswer("No object found that appears and then disappears.")

    if final_timestep_index < initial_timestep_index:
        raise ImpossibleToAnswer("No object found that appears and then disappears.")

    curr_frame_interleave = ((final_timestep_index) - initial_timestep_index) // (
        CLIP_LENGTH - 1
    )  # there seems to be a problem here
    visibility_objecst_window = visibility_mask[
        :, initial_timestep_index : final_timestep_index + 1
    ][:, ::curr_frame_interleave]
    total_unique_objects_seen = _count_objects_with_min_consecutive_visibility(
        visibility_objecst_window
    )
    count_objects_final = visibility_objecst_window[:, -1].sum()

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
