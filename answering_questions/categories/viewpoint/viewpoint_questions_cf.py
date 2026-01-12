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

from utils.config import get_config
from utils.decorators import with_resolved_attributes_cf
from utils.my_exception import ImpossibleToAnswer
from utils.bin_creation import create_mc_object_names_from_dataset
from utils.helpers import (
    fill_questions_cf,
    resolve_attributes_visible_at_timestep,
)

from categories.viewpoint.viewpoint_helpers import get_number_of_visible_objects


Number = Union[int, float]
Vector = Tuple[float, float, float]
WorldState = Mapping[str, Any]
QuestionPayload = Mapping[str, Any]
Answer = Union[str, float, Vector, Mapping[str, Any], Sequence[str]]

CLIP_LENGTH = get_config()["clip_length"]
FRAME_INTERLEAVE = get_config()["frame_interleave"]
MIN_VISIBLE_PIXELS = get_config()["min_pixels_visible"]
TIMESTART = get_config()["timestart"]
SAMPLING_RATE = get_config()["sampling_rate"]
RENDER_STEP = 1.0 / SAMPLING_RATE


@with_resolved_attributes_cf
def CF_VISIBILITY_OBJECT_COUNT(
    world_state_og: WorldState,
    world_state_mod: WorldState,
    answer_list_original_data_cf: dict,
    question: QuestionPayload,
    attributes,
    **kwargs,
) -> int:
    # we do not have corresponding data for the modified simulation to compare against
    if len(world_state_og["simulation"]) < len(world_state_mod["simulation"]):
        raise ImpossibleToAnswer(
            "Modified simulation has fewer timesteps than original; cannot compare."
        )
    
    assert len(attributes) == 0

    # Only take the first as the image is always that one
    answer_list_original_data_cf = answer_list_original_data_cf[0]
    timestep_index = int(
        answer_list_original_data_cf[3][0]
    )  # this has to be the image to get the question
    timestep = f"{TIMESTART + float(timestep_index) * RENDER_STEP:08.3f}"

    total_visible_objects = get_number_of_visible_objects(
        world_state_og, timestep
    )

    # balanced options around the initial count
    start = max(0, total_visible_objects - 2)
    shift = abs(total_visible_objects - 2) if total_visible_objects < 2 else 0
    balanced_bins = [
        str(i)
        for i in range(start, total_visible_objects + 2 + shift)
        if i != total_visible_objects
    ]

    labels, correct_idx = create_mc_object_names_from_dataset(
        str(total_visible_objects),
        [],
        balanced_bins,
        num_answers=4,
    )

    resolved_attributes = resolve_attributes_visible_at_timestep(
        ["OBJECT"], world_state_mod, timestep
    )

    return fill_questions_cf(
        question,
        labels,
        correct_idx,
        world_state_og,
        world_state_mod,
        timestep,
        resolved_attributes,
    )
