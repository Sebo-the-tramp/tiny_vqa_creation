"""
Mock temporal reasoning resolvers.

These helpers extract best-effort temporal answers from the provided world state.
They operate on lightweight metadata (positions, orientations, region tags, etc.)
and fall back to sensible defaults when information is missing.
"""

from __future__ import annotations

from utils.decorators import with_resolved_attributes

from utils.bin_creation import (
    create_mc_options_around_gt,
    create_mc_object_names_from_dataset,
    uniform_labels,
)

from typing import (
    Any,
    Mapping,
    Sequence,
    Tuple,
    Union,
)

from utils.helpers import (
    _get_random_integer,
    _shuffle_array,
)

import random
random.seed(41)

Number = Union[int, float]
Vector = Tuple[float, float, float]
WorldState = Mapping[str, Any]
QuestionPayload = Mapping[str, Any]
Answer = Union[str, float, Vector, Mapping[str, Any], Sequence[str]]

## --- Resolver functions -- ##
FPS=100
DELTA_STEP=0.01
VIDEO_DURATION_IN_SECONDS = 5
TOTAL_STEPS = FPS * VIDEO_DURATION_IN_SECONDS

@with_resolved_attributes
def F_TEMPORAL_SEQUENCE_IMAGES(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> Sequence[str]:
    """here we select a sequence of uniformly sampled images and return the next with random
    position in the simulations steps"""
    assert len(resolved_attributes) == 0

    n_frames = 4
    delta_timestep = 25 # quarter of a second    
    min = 0
    max = TOTAL_STEPS - (n_frames*delta_timestep) -1
    start_timestep = _get_random_integer(min, max)

    sequence = []
    for i in range(n_frames):
        image_filename = f"{((start_timestep + (i * delta_timestep)) * DELTA_STEP):.2f}"
        sequence.append(image_filename)

    choices = ["A.", "B.", "C.", "D."] # This could be anything <image_1> <image_2> etc. BUT WE CANNOT GIVE IN ORDER

    choices_correct_order = "".join(random.sample(choices, len(choices)))
    # so here sequence will correspond to the order chose 
    other_choices = ["".join(random.sample(choices, len(choices))) for _ in range(3)]

    correct_index = _get_random_integer(0,3)
    labels = other_choices[:correct_index] + [choices_correct_order] + other_choices[correct_index:]

    return question, labels, correct_index


@with_resolved_attributes
def F_TEMPORAL_PREDICTION_NEXT_IMAGE(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> Sequence[str]:
    """here we select a sequence of uniformly sampled images and return the next with random
    position in the simulations steps"""
    assert len(resolved_attributes) == 0

    n_frames = 4
    delta_timestep = 25 # quarter of a second    
    min = 0
    max = (TOTAL_STEPS//delta_timestep) - (n_frames + 1)    

    sequence = [f"{(i*delta_timestep*DELTA_STEP):.2f}" for i in range(TOTAL_STEPS//delta_timestep)]    

    start_timestep = _get_random_integer(min, max)

    given_sequence = sequence[start_timestep:start_timestep+4] # this is what we need next
    next_image = sequence[start_timestep+4]

    confounding_images = sequence[:start_timestep] + sequence[start_timestep+5:]

    random.shuffle(confounding_images)

    confounding_images = confounding_images[:3] 

    correct_index = _get_random_integer(0,3)
    labels = confounding_images[:correct_index] + [next_image] + confounding_images[correct_index:]

    return question, labels, correct_index


@with_resolved_attributes
def F_TEMPORAL_PREDICTION_PREVIOUS_IMAGE(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> Sequence[str]:
    """here we select a sequence of uniformly sampled images and return the next with random
    position in the simulations steps"""
    assert len(resolved_attributes) == 0

    n_frames = 4
    delta_timestep = 25 # quarter of a second    
    min = 1
    max = (TOTAL_STEPS//delta_timestep) - (n_frames)

    sequence = [f"{(i*delta_timestep*DELTA_STEP):.2f}" for i in range(TOTAL_STEPS//delta_timestep)]    

    start_timestep = _get_random_integer(min, max)

    given_sequence = sequence[start_timestep:start_timestep+4] # this is what we need next
    previous_image = sequence[start_timestep-1]

    confounding_images = sequence[:start_timestep-1] + sequence[start_timestep+4:]

    random.shuffle(confounding_images)

    confounding_images = confounding_images[:3] 

    correct_index = _get_random_integer(0,3)
    labels = confounding_images[:correct_index] + [previous_image] + confounding_images[correct_index:]

    return question, labels, correct_index

