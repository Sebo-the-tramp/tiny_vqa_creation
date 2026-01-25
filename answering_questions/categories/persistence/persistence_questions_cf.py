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
from utils.helpers import (
    fill_questions_cf,
    get_timestep_from_idx,
    get_visibility_mask_soft,
    # resolve_attributes_visible_at_timestep,
)
from utils.decorators import with_resolved_attributes_cf
from utils.my_exception import ImpossibleToAnswer
from utils.bin_creation import create_mc_object_names_from_dataset

from categories.persistence.persistence_helpers import (
    # choose_best_window_object_id,
    has_min_consecutive_visibility,
    # get_maximum_windows_for_each_object,
)

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

def _count_objects_with_min_consecutive_visibility(
    visibility_mask_window: np.ndarray,
) -> int:
    return int(
        sum(
            has_min_consecutive_visibility(row, 0, len(row) - 1)
            for row in visibility_mask_window
        )
    )


def get_frames_visible(world_state: WorldState) -> int:

    # order for longest window
    visibility_mask, _ = get_visibility_mask_soft(world_state)

    # we clip visibility max from up to max 24 frames so that we will have also the first frame t=0
    visibility_mask = visibility_mask[:, : 24]

    final_timestep_index = visibility_mask.shape[1] - 1
    final_timestep = get_timestep_from_idx(final_timestep_index)
    initial_timestep_index = 0
    initial_timestep = get_timestep_from_idx(initial_timestep_index)

    curr_frame_interleave = ((final_timestep_index) - initial_timestep_index) // (
        CLIP_LENGTH - 1
    )  # there seems to be a problem here
    visibility_objects_window = visibility_mask[
        :, initial_timestep_index : final_timestep_index + 1
    ][:, ::curr_frame_interleave]    

    total_unique_objects_seen = _count_objects_with_min_consecutive_visibility(
        visibility_objects_window
    )

    return visibility_mask, visibility_objects_window, total_unique_objects_seen, initial_timestep, final_timestep, initial_timestep_index, final_timestep_index


@with_resolved_attributes_cf
def CF_PERSISTENCE_OBJECT_TOTAL_COUNT_HIDDEN(
    world_state_og: WorldState,
    world_state_mod: WorldState,    
    question: QuestionPayload,
    attributes,
    resolved_attributes,
    **kwargs,
) -> Sequence[str]:
    """How many objects are present but not visible in the last frame?"""

    # note -> if the simulation timestep length is longer in the modified than in the og, then
    # we do not have corresponding data for the modified simulation to compare against

    # I don't think this holds any longer as long as there are at least 16 frames
    # if len(world_state_og["simulation"]) < len(world_state_mod["simulation"]):
    #     raise ImpossibleToAnswer(
    #         "Modified simulation has fewer timesteps than original; cannot compare."
    #     )

    assert len(attributes) == 0

    counterfactual_object_id = kwargs['object_moved_id']

    visibility_mask_og, visibility_objects_window_og, total_unique_objects_seen, \
    initial_timestep, final_timestep, initial_timestep_index, final_timestep_index \
        = get_frames_visible(world_state_og)

    count_objects_final_og = visibility_objects_window_og[:, -1].sum()

    visibility_mask_mod, visibility_objects_window_mod, total_unique_objects_seen, \
    initial_timestep, final_timestep, initial_timestep_index, final_timestep_index \
        = get_frames_visible(world_state_mod)

    count_objects_final_mod = visibility_objects_window_mod[:, -1].sum()

    if count_objects_final_mod >= count_objects_final_og:
        raise ImpossibleToAnswer("8 No change in number of visible objects between original and modified world.")

    hidden = total_unique_objects_seen - count_objects_final_mod

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
