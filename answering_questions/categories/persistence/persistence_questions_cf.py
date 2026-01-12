"""
Mock temporal reasoning resolvers.

These helpers extract best-effort temporal answers from the provided world state.
They operate on lightweight metadata (positions, orientations, region tags, etc.)
and fall back to sensible defaults when information is missing.
"""

from __future__ import annotations

import numpy as np

from typing import (
    Any,
    Mapping,
    Sequence,
    Tuple,
    Union,
)

from utils.config import get_config
from utils.helpers import fill_questions_cf, resolve_attributes_visible_at_timestep
from utils.decorators import with_resolved_attributes_cf
from utils.my_exception import ImpossibleToAnswer
from utils.bin_creation import create_mc_object_names_from_dataset

from categories.persistence.persistence_helpers import get_visibility_mask

Number = Union[int, float]
Vector = Tuple[float, float, float]
WorldState = Mapping[str, Any]
QuestionPayload = Mapping[str, Any]
Answer = Union[str, float, Vector, Mapping[str, Any], Sequence[str]]

SAMPLING_RATE = get_config()["sampling_rate"]
RENDER_STEP = 1.0 / SAMPLING_RATE
FRAME_INTERLEAVE = 4  # custom only for temporal questions (heuristic)
MIN_PIXELS_VISIBLE = get_config()["min_pixels_visible"]
CLIP_LENGTH = get_config()["clip_length"]
TIMESTART = 0.01


@with_resolved_attributes_cf
def CF_PERSISTENCE_OBJECT_TOTAL_COUNT(
    world_state_og: WorldState,
    world_state_mod: WorldState,
    answer_list_original_data_cf: dict,
    question: QuestionPayload,
    attributes,
    **kwargs,
) -> Sequence[str]:
    """How many objects are there in the last frame in total, including those currently hidden and/or out of frame?"""

    # note -> if the simulation timestep length is longer in the modified than in the og, then
    # we do not have corresponding data for the modified simulation to compare against
    if len(world_state_og["simulation"]) < len(world_state_mod["simulation"]):
        raise ImpossibleToAnswer(
            "Modified simulation has fewer timesteps than original; cannot compare."
        )

    # this could be a list of things no? only if there are single and multi images though which we will not have here
    frames_images = answer_list_original_data_cf[0][3]
    frames_int = [int(x) for x in frames_images]
    timestep_strings = [
        f"{TIMESTART + float(x) * RENDER_STEP:08.3f}" for x in frames_int
    ]

    visibility_mask, _ = get_visibility_mask(world_state_og)
    total_visible_objects = np.sum(visibility_mask, axis=0)

    final_timestep = timestep_strings[-1]
    initial_timestep = timestep_strings[0]

    # this is not the initial count, but the count at the timestep before disappearing
    count_objects_initial = total_visible_objects[frames_int[0]]

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
        ["OBJECT"], world_state_og, final_timestep
    )

    return fill_questions_cf(
        question,
        labels,
        correct_idx,
        world_state_og,
        world_state_mod,
        final_timestep,
        resolved_attributes,
        initial_timestep=initial_timestep,
    )


@with_resolved_attributes_cf
def CF_PERSISTENCE_OBJECT_TOTAL_COUNT_HIDDEN(
    world_state_og: WorldState,
    world_state_mod: WorldState,
    answer_list_original_data_cf: dict,
    question: QuestionPayload,
    attributes,
    **kwargs,
) -> Sequence[str]:
    """How many objects are present but not visible in the last frame?"""

    # note -> if the simulation timestep length is longer in the modified than in the og, then
    # we do not have corresponding data for the modified simulation to compare against
    if len(world_state_og["simulation"]) < len(world_state_mod["simulation"]):
        raise ImpossibleToAnswer(
            "Modified simulation has fewer timesteps than original; cannot compare."
        )

    assert len(attributes) == 0

    frames_images = answer_list_original_data_cf[0][3]
    frames_int = [int(x) for x in frames_images]
    timestep_strings = [
        f"{TIMESTART + float(x) * RENDER_STEP:08.3f}" for x in frames_int
    ]

    visibility_mask, _ = get_visibility_mask(world_state_og)
    total_visible_objects = np.sum(visibility_mask, axis=0)

    final_timestep = timestep_strings[-1]
    initial_timestep = timestep_strings[0]

    # this is not the initial count, but the count at the timestep before disappearing
    count_objects_initial = total_visible_objects[frames_int[0]]
    count_objects_final = total_visible_objects[frames_int[-1]]

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
        ["OBJECT"], world_state_og, final_timestep
    )

    return fill_questions_cf(
        question,
        labels,
        correct_idx,
        world_state_og,
        world_state_mod,
        final_timestep,
        resolved_attributes,
        initial_timestep=initial_timestep,
    )
