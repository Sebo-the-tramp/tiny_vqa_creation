"""
Mock temporal reasoning resolvers.

These helpers extract best-effort temporal answers from the provided world state.
They operate on lightweight metadata (positions, orientations, region tags, etc.)
and fall back to sensible defaults when information is missing.
"""

from __future__ import annotations

from utils.decorators import with_resolved_attributes
from utils.frames_selection import uniformly_sample_frames_start_end_delta

from typing import (
    Any,
    Mapping,
    Sequence,
    Tuple,
    Union,
)

from utils.helpers import (
    _get_random_integer,
)

import random

random.seed(41)

Number = Union[int, float]
Vector = Tuple[float, float, float]
WorldState = Mapping[str, Any]
QuestionPayload = Mapping[str, Any]
Answer = Union[str, float, Vector, Mapping[str, Any], Sequence[str]]

## --- Resolver functions -- ##
FPS = 100
DELTA_STEP = 0.01
VIDEO_DURATION_IN_SECONDS = 5
N_FRAMES = 4
DELTA_TIMESTEP = 25  # quarter of a second


@with_resolved_attributes
def F_TEMPORAL_SEQUENCE_IMAGES(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> Sequence[str]:
    """here we select a sequence of uniformly sampled images and return the next with random
    position in the simulations steps"""
    assert len(resolved_attributes) == 0

    total_steps = len(world_state["simulation"])
    min = 0
    max = total_steps - (N_FRAMES * DELTA_TIMESTEP) - 1
    start_timestep = _get_random_integer(min, max)

    imgs_idx = uniformly_sample_frames_start_end_delta(
        start_timestep, start_timestep + N_FRAMES * DELTA_TIMESTEP, DELTA_TIMESTEP
    )

    choices = ["A.", "B.", "C.", "D."]

    pair_choice_imgs_idx = [(imgs_idx[i], choices[i]) for i in range(len(choices))]

    correct_pair_choice_imgs_idx = random.sample(pair_choice_imgs_idx, len(choices))
    shuffled_imgs_idx = [pair[0] for pair in correct_pair_choice_imgs_idx]
    choices_correct_order = "".join([pair[1] for pair in correct_pair_choice_imgs_idx])
    # so here sequence will correspond to the order chose
    other_choices = ["".join(random.sample(choices, len(choices))) for _ in range(3)]

    correct_index = _get_random_integer(0, 3)
    labels = (
        other_choices[:correct_index]
        + [choices_correct_order]
        + other_choices[correct_index:]
    )

    return (question, labels, correct_index, shuffled_imgs_idx)


@with_resolved_attributes
def F_TEMPORAL_PREDICTION_NEXT_IMAGE(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> Sequence[str]:
    """here we select a sequence of uniformly sampled images and return the next with random
    position in the simulations steps"""
    assert len(resolved_attributes) == 0

    min = 0
    total_steps = len(world_state["simulation"])
    max = (total_steps // DELTA_TIMESTEP) - (N_FRAMES + 1)

    sequence = [
        f"{(i * DELTA_TIMESTEP)}".zfill(6) for i in range(total_steps // DELTA_TIMESTEP)
    ]

    start_timestep = _get_random_integer(min, max)

    given_sequence = sequence[
        start_timestep : start_timestep + 4
    ]  # this is what we need next
    next_image = sequence[start_timestep + 4]

    confounding_images = sequence[:start_timestep] + sequence[start_timestep + 5 :]

    random.shuffle(confounding_images)

    confounding_images = confounding_images[:3]

    correct_index = _get_random_integer(0, 3)
    labels = (
        confounding_images[:correct_index]
        + [next_image]
        + confounding_images[correct_index:]
    )

    return question, labels, correct_index, given_sequence


@with_resolved_attributes
def F_TEMPORAL_PREDICTION_PREVIOUS_IMAGE(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> Sequence[str]:
    """here we select a sequence of uniformly sampled images and return the next with random
    position in the simulations steps"""
    assert len(resolved_attributes) == 0

    total_steps = len(world_state["simulation"])
    min = 1
    max = (total_steps // DELTA_TIMESTEP) - (N_FRAMES)

    sequence = [
        f"{(i * DELTA_TIMESTEP)}".zfill(6) for i in range(total_steps // DELTA_TIMESTEP)
    ]

    start_timestep = _get_random_integer(min, max)

    given_sequence = sequence[
        start_timestep : start_timestep + 4
    ]  # this is what we need next
    previous_image = sequence[start_timestep - 1]

    confounding_images = sequence[: start_timestep - 1] + sequence[start_timestep + 4 :]

    random.shuffle(confounding_images)

    confounding_images = confounding_images[:3]

    correct_index = _get_random_integer(0, 3)
    labels = (
        confounding_images[:correct_index]
        + [previous_image]
        + confounding_images[correct_index:]
    )

    return question, labels, correct_index, given_sequence


@with_resolved_attributes
def F_TEMPORAL_PREDICTION_MISSING_IMAGE(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> Sequence[str]:
    """here we select a sequence of uniformly sampled images and return the next with random
    position in the simulations steps"""
    assert len(resolved_attributes) == 0

    total_steps = len(world_state["simulation"])
    min = 1
    max = (total_steps // DELTA_TIMESTEP) - (N_FRAMES)

    sequence = [
        f"{(i * DELTA_TIMESTEP)}".zfill(6) for i in range(total_steps // DELTA_TIMESTEP)
    ]

    start_timestep = _get_random_integer(min, max)

    given_sequence = sequence[start_timestep : start_timestep + 5]

    # assumption
    # we do this to ensure that the in the confounding images, we do not have by chance
    # another one that could complete the sequence as the first or last
    confounding_images = sequence[: start_timestep - 1] + sequence[start_timestep + 6 :]

    random.shuffle(confounding_images)

    confounding_images = confounding_images[:3]

    index_of_image_to_remove = _get_random_integer(0, 4)
    given_sequence.pop(index_of_image_to_remove)

    correct_index = _get_random_integer(0, 3)

    labels = (
        confounding_images[:correct_index]
        + [given_sequence[correct_index]]
        + confounding_images[correct_index:]
    )

    return question, labels, correct_index, given_sequence
