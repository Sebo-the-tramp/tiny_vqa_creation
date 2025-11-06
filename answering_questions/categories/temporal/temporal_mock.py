"""
Mock temporal reasoning resolvers.

These helpers extract best-effort temporal answers from the provided world state.
They operate on lightweight metadata (positions, orientations, region tags, etc.)
and fall back to sensible defaults when information is missing.
"""

from __future__ import annotations

from categories.temporal.temporal_helpers import calculate_most_dissimilar_confounding_images
from utils.decorators import with_resolved_attributes
from utils.frames_selection import uniformly_sample_frames_start_end_delta
from utils.my_exception import ImpossibleToAnswer

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
import numpy as np

import torch
from fused_ssim import fused_ssim # nofa

random.seed(41)

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
# FRAME_INTERLEAVE = get_config()["frame_interleave"]
FRAME_INTERLEAVE = 4 # custom only for temporal questions (heuristic)

MIN_PIXELS_VISIBLE = get_config()["min_pixels_visible"]

# I want to sample every quarter of a second
# FRAME_INTERLEAVE = int(
#     -(-0.25 // RENDER_STEP)
# )  # same as math.ceil(0.25 / RENDER_STEP) but better quarter of a second


@with_resolved_attributes
def F_TEMPORAL_SEQUENCE_IMAGES(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> Sequence[str]:
    """here we select a sequence of uniformly sampled images and return the next with random
    position in the simulations steps"""
    assert len(attributes) == 0
    n_frames = 4    

    total_frames = len(world_state["simulation"]) // 3
    min_frame = 0
    max_frame = total_frames - (n_frames * FRAME_INTERLEAVE) - 1
    start_frame = get_random_integer(min_frame, max_frame)
    end_frame = start_frame + (n_frames * FRAME_INTERLEAVE)

    imgs_idx = uniformly_sample_frames_start_end_delta(
        start_frame, end_frame, FRAME_INTERLEAVE
    )

    imgs_idx_shuffled = imgs_idx.copy()
    random.shuffle(imgs_idx_shuffled)

    choices = ["A", "B", "C", "D"]

    pair_choice_imgs_idx = [(imgs_idx_shuffled[i], choices[i]) for i in range(len(choices))]

    correct_pair_choice_imgs_idx = sorted(pair_choice_imgs_idx, key=lambda x: x[0])    
    choices_correct_order = "-".join([pair[1] for pair in correct_pair_choice_imgs_idx])
    # so here sequence will correspond to the order chose
    other_choices = ["-".join(random.sample(choices, len(choices))) for _ in range(3)]

    correct_index = get_random_integer(0, 3)
    labels = (
        other_choices[:correct_index]
        + [choices_correct_order]
        + other_choices[correct_index:]
    )

    return [[question, labels, correct_index, imgs_idx_shuffled]]


@with_resolved_attributes
def F_TEMPORAL_PREDICTION_NEXT_IMAGE(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> Sequence[str]:
    """here we select a sequence of uniformly sampled images and return the next with random
    position in the simulations steps"""
    assert len(attributes) == 0
    n_frames = 5    

    total_frames = len(world_state["simulation"]) // 3
    min_frame = 0
    max_frame = total_frames - (n_frames * FRAME_INTERLEAVE) - 1
    start_frame = get_random_integer(min_frame, max_frame)
    end_frame = start_frame + (n_frames * FRAME_INTERLEAVE)

    all_frames_idx = uniformly_sample_frames_start_end_delta(
        0, total_frames, 1
    )

    sequence_idx = uniformly_sample_frames_start_end_delta(
        start_frame, end_frame, FRAME_INTERLEAVE
    )

    given_sequence = sequence_idx[:4]
    next_image = sequence_idx[4]

    confounding_images_candidates = all_frames_idx[:start_frame] + all_frames_idx[end_frame:]
    confounding_images = calculate_most_dissimilar_confounding_images(
        confounding_images_candidates, next_image, **kwargs
    )

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
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> Sequence[str]:
    """here we select a sequence of uniformly sampled images and return the next with random
    position in the simulations steps"""
    assert len(attributes) == 0
    n_frames = 5    

    total_frames = len(world_state["simulation"]) // 3
    min_frame = 0
    max_frame = total_frames - (n_frames * FRAME_INTERLEAVE) - 1
    start_frame = get_random_integer(min_frame, max_frame)
    end_frame = start_frame + (n_frames * FRAME_INTERLEAVE)

    all_frames_idx = uniformly_sample_frames_start_end_delta(
        0, total_frames, 1
    )

    sequence_idx = uniformly_sample_frames_start_end_delta(
        start_frame, end_frame, FRAME_INTERLEAVE
    )

    given_sequence = sequence_idx[1:]
    previous_image = sequence_idx[0]

    confounding_images_candidates = all_frames_idx[:start_frame] + all_frames_idx[end_frame:]
    confounding_images = calculate_most_dissimilar_confounding_images(
        confounding_images_candidates, previous_image, **kwargs
    )

    random.shuffle(confounding_images)

    confounding_images = confounding_images[:3]

    correct_index = get_random_integer(0, 3)
    labels = (
        confounding_images[:correct_index]
        + [previous_image]
        + confounding_images[correct_index:]
    )

    return [[question, labels, correct_index, given_sequence]]


@with_resolved_attributes
def F_TEMPORAL_PREDICTION_MISSING_IMAGE(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> Sequence[str]:
    """here we select a sequence of uniformly sampled images and return the next with random
    position in the simulations steps"""
    assert len(attributes) == 0

    n_frames = 5

    total_frames = len(world_state["simulation"]) // 3
    min_frame = 0
    max_frame = total_frames - (n_frames * FRAME_INTERLEAVE) - 1
    start_frame = get_random_integer(min_frame, max_frame)
    end_frame = start_frame + (n_frames * FRAME_INTERLEAVE) - 1

    all_frames_idx = uniformly_sample_frames_start_end_delta(
        0, total_frames, 1
    )

    sequence_idx = uniformly_sample_frames_start_end_delta(
        start_frame, end_frame, FRAME_INTERLEAVE
    )

    # 0 - start_frame - 2*FRAME_INTERLEAVE ... start_frame ... end_frame ... end_frame + 2*FRAME_INTERLEAVE - total_frames
    first_possible = max(0, start_frame - 2 * FRAME_INTERLEAVE)
    last_possible = min(total_frames, end_frame + 2 * FRAME_INTERLEAVE)

    index_of_image_to_remove = get_random_integer(0, 4)
    given_sequence = sequence_idx.copy()
    given_sequence.pop(index_of_image_to_remove)

    confounding_images_candidates = all_frames_idx[:first_possible] + all_frames_idx[last_possible:]    
    confounding_images = calculate_most_dissimilar_confounding_images(
        confounding_images_candidates, sequence_idx[index_of_image_to_remove], **kwargs
    )

    random.shuffle(confounding_images)

    confounding_images = confounding_images[:3]

    correct_index = get_random_integer(0, 3)    

    labels = (
        confounding_images[:correct_index]
        + [sequence_idx[index_of_image_to_remove]]
        + confounding_images[correct_index:]
    )

    return [[question, labels, correct_index, given_sequence]]


@with_resolved_attributes
def F_CAMERA_MOTION_DIRECTION(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> Sequence[str]:
    
    assert len(attributes) == 0
    n_frames = 8

    all_timesteps = len(world_state["simulation"]) // 3
    last_frame_idx = get_random_integer(n_frames * FRAME_INTERLEAVE, all_timesteps)
    first_frame_idx = last_frame_idx - (n_frames * FRAME_INTERLEAVE)

    all_frames_idx = list(world_state["simulation"].keys())

    timestep_frame_0 = all_frames_idx[first_frame_idx]
    timestep_mid = all_frames_idx[first_frame_idx + (n_frames//2)* FRAME_INTERLEAVE]
    timestep_final = all_frames_idx[last_frame_idx - FRAME_INTERLEAVE]

    given_sequence = uniformly_sample_frames_start_end_delta(
        first_frame_idx,
        last_frame_idx - FRAME_INTERLEAVE,
        FRAME_INTERLEAVE,
    )

    initial_look_at = np.array(world_state["simulation"][timestep_frame_0]["camera"]["at"])        
    initial_eye = np.array(world_state["simulation"][timestep_frame_0]["camera"]["eye"])
    mid_eye = np.array(world_state["simulation"][timestep_mid]["camera"]["eye"])
    final_eye = np.array(world_state["simulation"][timestep_final]["camera"]["eye"])
    
    initial_direction = initial_look_at - initial_eye
    initial_direction = initial_direction / np.linalg.norm(initial_direction)
    right_vector = np.cross(np.array([0, 0, -1]), initial_direction)  # not sure about the sign here though
    up_vector = np.array([0, 0, -1])  # assuming z is up

    movement_initial_to_mid = mid_eye - initial_eye
    movement_mid_to_final = final_eye - mid_eye

    projection_initial_to_mid = [
        np.dot(movement_initial_to_mid, initial_direction),
        np.dot(movement_initial_to_mid, right_vector),
        np.dot(movement_initial_to_mid, up_vector),
    ]

    projection_mid_to_final = [
        np.dot(movement_mid_to_final, initial_direction),
        np.dot(movement_mid_to_final, right_vector),
        np.dot(movement_mid_to_final, up_vector),
    ]

    threshold = 0.05

    directions = ["forward", "backward", "right", "left", "up", "down"]
    all_possible_answers = []
    for perm in itertools.permutations(directions, 2):
        if perm[0] != perm[1]:
            all_possible_answers.append(f"{perm[0]} then {perm[1]}")
        else:
            all_possible_answers.append(f"{perm[0]}")

    # first movement
    if abs(projection_initial_to_mid[0]) > abs(projection_initial_to_mid[1]) and abs(projection_initial_to_mid[0]) > abs(projection_initial_to_mid[2]):
        # forward/backward
        if projection_initial_to_mid[0] > threshold:
            first_movement = "forward"
        elif projection_initial_to_mid[0] < -threshold:
            first_movement = "backward"
        else:
            first_movement = "no significant movement"
    elif abs(projection_initial_to_mid[1]) > abs(projection_initial_to_mid[0]) and abs(projection_initial_to_mid[1]) > abs(projection_initial_to_mid[2]):   
        # right/left
        if projection_initial_to_mid[1] > threshold:
            first_movement = "right"
        elif projection_initial_to_mid[1] < -threshold:
            first_movement = "left"
        else:
            first_movement = "no significant movement"
    else:   
        # up/down
        if projection_initial_to_mid[2] > threshold:
            first_movement = "up"
        elif projection_initial_to_mid[2] < -threshold:
            first_movement = "down"
        else:
            first_movement = "no significant movement"

    # second movement 
    if abs(projection_mid_to_final[0]) > abs(projection_mid_to_final[1]) and abs(projection_mid_to_final[0]) > abs(projection_mid_to_final[2]):
        # forward/backward
        if projection_mid_to_final[0] > threshold:
            second_movement = "forward"
        elif projection_mid_to_final[0] < -threshold:
            second_movement = "backward"
        else:
            second_movement = "no significant movement"
    elif abs(projection_mid_to_final[1]) > abs(projection_mid_to_final[0]) and abs(projection_mid_to_final[1]) > abs(projection_mid_to_final[2]):
        # right/left
        if projection_mid_to_final[1] > threshold:
            second_movement = "right"
        elif projection_mid_to_final[1] < -threshold:
            second_movement = "left"
        else:
            second_movement = "no significant movement"
    else:   
        # up/down exactly I modified the sign above because of that
        if projection_mid_to_final[2] < -threshold:
            second_movement = "up"
        elif projection_mid_to_final[2] > threshold:
            second_movement = "down"
        else:
            second_movement = "no significant movement"
    if first_movement == "no significant movement" and second_movement == "no significant movement":
        answer = "no significant movement"
    elif first_movement == "no significant movement":
        answer = second_movement
    elif second_movement == "no significant movement":
        answer = first_movement
    elif first_movement == second_movement:
        answer = first_movement
    else:
        answer = f"{first_movement} then {second_movement}"    

    other_answers = [ans for ans in all_possible_answers if ans != answer]
    random.shuffle(other_answers)
    other_answers = other_answers[:3]
    correct_index = get_random_integer(0, 3)
    labels = (  
        other_answers[:correct_index]
        + [answer]
        + other_answers[correct_index:]
    )
    return [[question, labels, correct_index, given_sequence]]

    
@with_resolved_attributes
def F_CAMERA_ZOOM_BEHAVIOR(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> Sequence[str]:

    assert len(attributes) == 0

    n_frames = 8

    all_timesteps = len(world_state["simulation"])
    last_frame_idx = get_random_integer(n_frames * FRAME_INTERLEAVE, all_timesteps) #
    first_frame_idx = last_frame_idx - (n_frames * FRAME_INTERLEAVE)

    all_frames_idx = list(world_state["simulation"].keys())

    timestep_frame_0 = all_frames_idx[first_frame_idx]
    timestep_mid = all_frames_idx[first_frame_idx + (n_frames//2)* FRAME_INTERLEAVE]
    timestep_final = all_frames_idx[last_frame_idx - FRAME_INTERLEAVE]

    given_sequence = uniformly_sample_frames_start_end_delta(
        first_frame_idx,
        last_frame_idx - FRAME_INTERLEAVE,
        FRAME_INTERLEAVE,
    )

    initial_fov = world_state["simulation"][timestep_frame_0]["camera"]["fov"]
    mid_fov = world_state["simulation"][timestep_mid]["camera"]["fov"]
    final_fov = world_state["simulation"][timestep_final]["camera"]["fov"]

    zoom_threshold = 5.0  # degrees

    if mid_fov < initial_fov - zoom_threshold:
        first_movement = "zoom in"
    elif mid_fov > initial_fov + zoom_threshold:
        first_movement = "zoom out"
    else:
        first_movement = "no zoom"

    if final_fov < mid_fov - zoom_threshold:
        second_movement = "zoom in"
    elif final_fov > mid_fov + zoom_threshold:
        second_movement = "zoom out"
    else:
        second_movement = "no zoom"
    if first_movement == "no zoom" and second_movement == "no zoom":
        answer = "no zoom"
    elif first_movement == "no zoom":
        answer = second_movement
    elif second_movement == "no zoom":
        answer = first_movement
    elif first_movement == second_movement:
        answer = first_movement
    else:
        answer = f"{first_movement} then {second_movement}"

    all_possible_answers = ["zoom in", "zoom out", "no zoom", "zoom in then out", "zoom out then in"]

    other_answers = [ans for ans in all_possible_answers if ans != answer]
    random.shuffle(other_answers)
    other_answers = other_answers[:3]
    correct_index = get_random_integer(0, 3)
    labels = (
        other_answers[:correct_index]
        + [answer]
        + other_answers[correct_index:]
    )   

    return [[question, labels, correct_index, given_sequence]]