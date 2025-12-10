"""
Mock temporal reasoning resolvers.

These helpers extract best-effort temporal answers from the provided world state.
They operate on lightweight metadata (positions, orientations, region tags, etc.)
and fall back to sensible defaults when information is missing.
"""

from __future__ import annotations

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

from utils.helpers import (
    iter_objects,
    fill_questions,
    resolve_attributes_visible_at_timestep
)

from utils.bin_creation import create_mc_object_names_from_dataset
from categories.persistence.persistence_helpers import (
    check_visibility_sequence,
    compute_visibility_counts,
    find_first_visibility_drop
)


Number = Union[int, float]
Vector = Tuple[float, float, float]
WorldState = Mapping[str, Any]
QuestionPayload = Mapping[str, Any]
Answer = Union[str, float, Vector, Mapping[str, Any], Sequence[str]]

## --- Resolver functions -- ##

from utils.config import get_config
import itertools

SAMPLING_RATE = get_config()["sampling_rate"]
RENDER_STEP = 1.0 / SAMPLING_RATE
FRAME_INTERLEAVE = 4  # custom only for temporal questions (heuristic)
MIN_PIXELS_VISIBLE = get_config()["min_pixels_visible"]
CLIP_LENGTH = get_config()["clip_length"]




@with_resolved_attributes
def F_PERSISTENCE_OBJECT_PRESENT(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> Sequence[str]:
    
    assert len(attributes) == 0

    candidate_object = None
    for object in iter_objects(world_state):
        found_pattern, final_timestep = check_visibility_sequence(
            world_state, object["id"],
            list(world_state["simulation"].keys())
        )

        if found_pattern:
            answer = object["name"]
            # if there are 2 objects (unlikely)
            if candidate_object is not None:
                raise ImpossibleToAnswer("Multiple objects found with the required persistence pattern.")
            else:
                candidate_object = object
                print("Found object with persistence pattern:", answer)
                print("At timestep:", final_timestep)
                break

    if candidate_object is None:
        raise ImpossibleToAnswer("No object found with the required persistence pattern.")
    
    labels, correct_idx = create_mc_object_names_from_dataset(
        candidate_object["name"],
        [],
        get_all_objects_names(),
        num_answers=4,
    )

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, final_timestep
    )

    return fill_questions(
        question, labels, correct_idx, world_state, final_timestep, resolved_attributes
    )


@with_resolved_attributes
def F_PERSISTENCE_OBJECT_DISAPPEAR(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> Sequence[str]:
    
    return F_PERSISTENCE_OBJECT_PRESENT(
        world_state, question, kwargs["destination_simulation_id_path"]
    )
# this has to be really thought throughly cause as it is it can not be factually answered


@with_resolved_attributes
def F_PERSISTENCE_OBJECT_TOTAL_COUNT(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> Sequence[str]:

    assert len(attributes) == 0

    list_indexes = list(world_state["simulation"].keys())[::FRAME_INTERLEAVE] 

    # I need to start from timestep 4*FRAME_INTERLEAVE to have enough margin if the second frame is dropping a number of visibility
    timestep_counts = compute_visibility_counts(
        world_state, list_indexes[4:-4]
    )

    timestep_of_hidden = find_first_visibility_drop(timestep_counts)

    if timestep_of_hidden is None:
        raise ImpossibleToAnswer("No visibility drop detected in the simulation.")

    # checking the final timestep after the drop
    final_timestep_index = min(timestep_of_hidden + 4, len(list_indexes) - 1)
    initial_timestep_index = max(0, final_timestep_index - CLIP_LENGTH)

    if final_timestep_index - initial_timestep_index < CLIP_LENGTH:
        raise ImpossibleToAnswer("Not enough timesteps before visibility drop to answer the question.")

    final_timestep = list_indexes[final_timestep_index]
    count_objects_initial = timestep_counts[initial_timestep_index]

    labels, correct_idx = create_mc_object_names_from_dataset(
        str(count_objects_initial),
        [],
        [str(i) for i in range(0, 11)],
        num_answers=4,
    )

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, final_timestep
    )

    return fill_questions(
        question, labels, correct_idx, world_state, final_timestep, resolved_attributes 
    )


@with_resolved_attributes
def F_PERSISTENCE_OBJECT_TOTAL_COUNT_HIDDEN(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> Sequence[str]:

    assert len(attributes) == 0

    list_indexes = list(world_state["simulation"].keys())[::FRAME_INTERLEAVE] 

    # I need to start from timestep 4*FRAME_INTERLEAVE to have enough margin if the second frame is dropping a number of visibility
    timestep_counts = compute_visibility_counts(
        world_state, list_indexes[4:-4]
    )

    timestep_of_hidden = find_first_visibility_drop(timestep_counts)

    if timestep_of_hidden is None:
        raise ImpossibleToAnswer("No visibility drop detected in the simulation.")

    # checking the final timestep after the drop
    final_timestep_index = min(timestep_of_hidden + 4, len(timestep_counts) - 1)
    initial_timestep_index = max(0, final_timestep_index - CLIP_LENGTH)

    if final_timestep_index - initial_timestep_index < CLIP_LENGTH:
        raise ImpossibleToAnswer("Not enough timesteps before visibility drop to answer the question.")

    final_timestep = list_indexes[final_timestep_index]
    count_objects_initial = timestep_counts[initial_timestep_index]
    count_objects_final = timestep_counts[final_timestep_index]

    labels, correct_idx = create_mc_object_names_from_dataset(
        str(count_objects_initial - count_objects_final),
        [],
        [str(i) for i in range(0, 11)],
        num_answers=4,
    )

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, final_timestep
    )

    return fill_questions(
        question, labels, correct_idx, world_state, final_timestep, resolved_attributes 
    )