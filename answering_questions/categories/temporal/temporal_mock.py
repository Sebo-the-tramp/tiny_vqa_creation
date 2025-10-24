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
    get_random_integer,
)

import random

random.seed(41)

Number = Union[int, float]
Vector = Tuple[float, float, float]
WorldState = Mapping[str, Any]
QuestionPayload = Mapping[str, Any]
Answer = Union[str, float, Vector, Mapping[str, Any], Sequence[str]]

## --- Resolver functions -- ##

from utils.config import get_config

SAMPLING_RATE = get_config()["sampling_rate"]
RENDER_STEP = 1.0 / SAMPLING_RATE

# I want to sample every quarter of a second
FRAME_STRIDE = int(
    -(-0.25 // RENDER_STEP)
)  # same as math.ceil(0.25 / RENDER_STEP) but better quarter of a second


@with_resolved_attributes
def F_TEMPORAL_SEQUENCE_IMAGES(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> Sequence[str]:
    """here we select a sequence of uniformly sampled images and return the next with random
    position in the simulations steps"""
    assert len(resolved_attributes) == 0
    n_frames = 4

    total_frames = len(world_state["simulation"])
    min_frame = 0
    max_frame = total_frames - (n_frames * FRAME_STRIDE) - 1
    start_frame = get_random_integer(min_frame, max_frame)
    end_frame = start_frame + (n_frames * FRAME_STRIDE) - 1

    imgs_idx = uniformly_sample_frames_start_end_delta(
        start_frame, end_frame, FRAME_STRIDE
    )

    choices = ["A.", "B.", "C.", "D."]

    pair_choice_imgs_idx = [(imgs_idx[i], choices[i]) for i in range(len(choices))]

    correct_pair_choice_imgs_idx = random.sample(pair_choice_imgs_idx, len(choices))
    shuffled_imgs_idx = [pair[0] for pair in correct_pair_choice_imgs_idx]
    choices_correct_order = "".join([pair[1] for pair in correct_pair_choice_imgs_idx])
    # so here sequence will correspond to the order chose
    other_choices = ["".join(random.sample(choices, len(choices))) for _ in range(3)]

    correct_index = get_random_integer(0, 3)
    labels = (
        other_choices[:correct_index]
        + [choices_correct_order]
        + other_choices[correct_index:]
    )

    return [[question, labels, correct_index, shuffled_imgs_idx]]


@with_resolved_attributes
def F_TEMPORAL_PREDICTION_NEXT_IMAGE(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> Sequence[str]:
    """here we select a sequence of uniformly sampled images and return the next with random
    position in the simulations steps"""
    assert len(resolved_attributes) == 0
    n_frames = 5

    total_frames = len(world_state["simulation"])
    min_frame = 0
    max_frame = total_frames - (n_frames * FRAME_STRIDE) - 1
    start_frame = get_random_integer(min_frame, max_frame)
    end_frame = start_frame + (n_frames * FRAME_STRIDE) - 1

    all_frames_idx = uniformly_sample_frames_start_end_delta(
        0, total_frames, 1
    )

    sequence_idx = uniformly_sample_frames_start_end_delta(
        start_frame, end_frame, FRAME_STRIDE
    )

    given_sequence = sequence_idx[:4]
    next_image = sequence_idx[4]

    confounding_images = all_frames_idx[:start_frame] + all_frames_idx[end_frame:]

    random.shuffle(confounding_images)

    confounding_images = confounding_images[:3]

    correct_index = get_random_integer(0, 3)
    labels = (
        confounding_images[:correct_index]
        + [next_image]
        + confounding_images[correct_index:]
    )

    return [[question, labels, correct_index, given_sequence]]


@with_resolved_attributes
def F_TEMPORAL_PREDICTION_PREVIOUS_IMAGE(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> Sequence[str]:
    """here we select a sequence of uniformly sampled images and return the next with random
    position in the simulations steps"""
    assert len(resolved_attributes) == 0
    n_frames = 5

    total_frames = len(world_state["simulation"])
    min_frame = 0
    max_frame = total_frames - (n_frames * FRAME_STRIDE) - 1
    start_frame = get_random_integer(min_frame, max_frame)
    end_frame = start_frame + (n_frames * FRAME_STRIDE) - 1

    all_frames_idx = uniformly_sample_frames_start_end_delta(
        0, total_frames, 1
    )

    sequence_idx = uniformly_sample_frames_start_end_delta(
        start_frame, end_frame, FRAME_STRIDE
    )

    given_sequence = sequence_idx[1:]
    next_image = sequence_idx[0]

    confounding_images = all_frames_idx[:start_frame] + all_frames_idx[end_frame:]

    random.shuffle(confounding_images)

    confounding_images = confounding_images[:3]

    correct_index = get_random_integer(0, 3)
    labels = (
        confounding_images[:correct_index]
        + [next_image]
        + confounding_images[correct_index:]
    )

    return [[question, labels, correct_index, given_sequence]]


@with_resolved_attributes
def F_TEMPORAL_PREDICTION_MISSING_IMAGE(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> Sequence[str]:
    """here we select a sequence of uniformly sampled images and return the next with random
    position in the simulations steps"""
    assert len(resolved_attributes) == 0

    n_frames = 5

    total_frames = len(world_state["simulation"])
    min_frame = 0
    max_frame = total_frames - (n_frames * FRAME_STRIDE) - 1
    start_frame = get_random_integer(min_frame, max_frame)
    end_frame = start_frame + (n_frames * FRAME_STRIDE) - 1

    all_frames_idx = uniformly_sample_frames_start_end_delta(
        0, total_frames, 1
    )

    sequence_idx = uniformly_sample_frames_start_end_delta(
        start_frame, end_frame, FRAME_STRIDE
    )

    # 0 - start_frame - 2*FRAME_STRIDE ... start_frame ... end_frame ... end_frame + 2*FRAME_STRIDE - total_frames
    first_possible = max(0, start_frame - 2 * FRAME_STRIDE)
    last_possible = min(total_frames, end_frame + 2 * FRAME_STRIDE)

    confounding_images = (
        all_frames_idx[:first_possible] + all_frames_idx[last_possible:]
    )

    random.shuffle(confounding_images)

    confounding_images = confounding_images[:3]

    index_of_image_to_remove = get_random_integer(0, 4)
    given_sequence = sequence_idx.copy()
    given_sequence.pop(index_of_image_to_remove)

    correct_index = get_random_integer(0, 3)

    labels = (
        confounding_images[:correct_index]
        + [sequence_idx[index_of_image_to_remove]]
        + confounding_images[correct_index:]
    )

    return [[question, labels, correct_index, given_sequence]]
