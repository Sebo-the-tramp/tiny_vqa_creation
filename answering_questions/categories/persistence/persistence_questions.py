"""
Mock temporal reasoning resolvers.

These helpers extract best-effort temporal answers from the provided world state.
They operate on lightweight metadata (positions, orientations, region tags, etc.)
and fall back to sensible defaults when information is missing.
"""

from __future__ import annotations

import random
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

from utils.helpers import fill_questions, resolve_attributes_visible_at_timestep, get_timestep_from_idx

from utils.bin_creation import create_mc_object_names_from_dataset
from categories.persistence.persistence_helpers import (
    get_visibility_mask,    
    get_optimal_timestep_interval,    
    get_maximum_windows_for_each_object
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

    object_proposed = get_maximum_windows_for_each_object(world_state)
    
    # check fo unique object that appears and then disappears for time interval
    chosen_object_id = None

    all_objects_ids = set([str(i) for i in range(1, len(world_state["objects"]) + 1)])

    for obj_id in object_proposed:
        initial_timestep_index = object_proposed[str(obj_id)][2]
        final_timestep_index = object_proposed[str(obj_id)][3]
        
        for obj_id_check in object_proposed:
            if str(obj_id) == str(obj_id_check):
                continue
            
            initial_timestep_index_check = object_proposed[str(obj_id_check)][2]
            final_timestep_index_check = object_proposed[str(obj_id_check)][3]

            if final_timestep_index_check == -1 or initial_timestep_index_check == -1:
                all_objects_ids.discard(str(obj_id_check))
                continue

            # check for overlap
            if not (final_timestep_index < initial_timestep_index_check or final_timestep_index_check < initial_timestep_index):
                all_objects_ids.discard(str(obj_id_check))
                all_objects_ids.discard(str(obj_id))

        if len(all_objects_ids) == 0:
            raise ImpossibleToAnswer(
                "More than one object found that appears and then disappears."
            )

    chosen_object_id = random.choice(list(all_objects_ids)) if len(all_objects_ids) == 1 else None

    if chosen_object_id is None:
        raise ImpossibleToAnswer(
            "No object found that appears and then disappears."
        )

    final_timestep_index = object_proposed[str(chosen_object_id)][3]
    initial_timestep_index = object_proposed[str(chosen_object_id)][2]

    if final_timestep_index == -1 or initial_timestep_index == -1:
        raise ImpossibleToAnswer(
            "No object found that appears and then disappears."
        )
    
    if final_timestep_index < initial_timestep_index:
        raise ImpossibleToAnswer(
            "No object found that appears and then disappears."
        )

    final_timestep = get_timestep_from_idx(final_timestep_index)    
    initial_timestep = get_timestep_from_idx(initial_timestep_index)

    obj_name = world_state["objects"][str(chosen_object_id)]["name"]

    labels, correct_idx = create_mc_object_names_from_dataset(
        obj_name,
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
    curr_frame_interleave = (final_timestep_index - initial_timestep_index) // (CLIP_LENGTH - 1)
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
    curr_frame_interleave = (final_timestep_index - initial_timestep_index) // (CLIP_LENGTH - 1)
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
