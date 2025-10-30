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
import numpy as np

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

# I want to sample every quarter of a second
FRAME_STRIDE = int(
    -(-0.25 // RENDER_STEP)
)  # same as math.ceil(0.25 / RENDER_STEP) but better quarter of a second


@with_resolved_attributes
def F_TEMPORAL_SEQUENCE_IMAGES(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> Sequence[str]:
    """here we select a sequence of uniformly sampled images and return the next with random
    position in the simulations steps"""
    assert len(attributes) == 0
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
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> Sequence[str]:
    """here we select a sequence of uniformly sampled images and return the next with random
    position in the simulations steps"""
    assert len(attributes) == 0
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
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> Sequence[str]:
    """here we select a sequence of uniformly sampled images and return the next with random
    position in the simulations steps"""
    assert len(attributes) == 0
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
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> Sequence[str]:
    """here we select a sequence of uniformly sampled images and return the next with random
    position in the simulations steps"""
    assert len(attributes) == 0

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


@with_resolved_attributes
def F_CAMERA_MOTION_DIRECTION(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> Sequence[str]:

    assert len(attributes) == 0

    n_frames = 8

    all_timesteps = len(world_state["simulation"])
    last_frame_idx = get_random_integer(n_frames * FRAME_STRIDE, all_timesteps) #
    first_frame_idx = last_frame_idx - (n_frames * FRAME_STRIDE)

    all_frames_idx = list(world_state["simulation"].keys())

    timestep_frame_0 = all_frames_idx[first_frame_idx]
    timestep_mid = all_frames_idx[first_frame_idx + (n_frames//2)* FRAME_STRIDE]
    timestep_final = all_frames_idx[last_frame_idx - FRAME_STRIDE]

    given_sequence = uniformly_sample_frames_start_end_delta(
        first_frame_idx,
        last_frame_idx - FRAME_STRIDE,
        FRAME_STRIDE,
    )

    initial_look_at = np.array(world_state["simulation"][timestep_frame_0]["camera"]["at"])
    initial_position = np.array(world_state["simulation"][timestep_frame_0]["camera"]["eye"])
    mid_position = np.array(world_state["simulation"][timestep_mid]["camera"]["eye"])
    final_position = np.array(world_state["simulation"][timestep_final]["camera"]["eye"])

    movement_initial_to_mid = mid_position - initial_position
    movement_mid_to_final = final_position - mid_position

    # in the test should go backward
    # we treat initial look at as reference direction

    # okay in order to disambiguate the question I need to take the assumption that the camera is moving 
    # with respect to the scene center ( usually at [0,0,0] )
    # so the direction_vector is from initial position to scene center
    # e.g. the camera might look somewhere else but the movement is defined with respect to the scene center
    # TODO double check with Raoul if we should take as center the mean of objects centers

    # direction_vector = [0,0,0] - initial_position
    direction_vector = initial_look_at
    direction_vector = direction_vector / np.linalg.norm(direction_vector)
    right_vector = np.cross(np.array([0, 0, 1]), direction_vector)  # not sure about the sign here though
    up_vector = np.array([0, 0, 1])  # assuming z is up

    movement_initial_to_mid_norm = movement_initial_to_mid / np.linalg.norm(movement_initial_to_mid)
    movement_mid_to_final_norm = movement_mid_to_final / np.linalg.norm(movement_mid_to_final)

    fwd_initial = np.dot(direction_vector, movement_initial_to_mid_norm)
    fwd_final = np.dot(direction_vector, movement_mid_to_final_norm)
    right_initial = np.dot(right_vector, movement_initial_to_mid_norm)
    right_final = np.dot(right_vector, movement_mid_to_final_norm)
    up_initial = np.dot(up_vector, movement_initial_to_mid_norm)
    up_final = np.dot(up_vector, movement_mid_to_final_norm)
   
    threshold = 0.5

    # # what is the biggest vector component
    abs_initial_components = [abs(fwd_initial), abs(right_initial), abs(up_initial)]
    abs_final_components = [abs(fwd_final), abs(right_final), abs(up_final)]
    max_initial_idx = np.argmax(abs_initial_components)
    max_final_idx = np.argmax(abs_final_components)

    all_possible_answers = []
    directions = ["forward", "backward", "right", "left", "up", "down"]    
    for perm in itertools.permutations(directions, 2):
        if perm[0] != perm[1]:
            all_possible_answers.append(f"{perm[0]} then {perm[1]}")
        else:
            all_possible_answers.append(f"{perm[0]}")

    if max_initial_idx == 0:  # forward/backward
        if fwd_initial > threshold:
            first_movement = "forward"
        elif fwd_initial < -threshold:
            first_movement = "backward"
        else:
            first_movement = "no significant movement"
    elif max_initial_idx == 1:  # right/left
        if right_initial > threshold:
            first_movement = "right"
        elif right_initial < -threshold:
            first_movement = "left"
        else:
            first_movement = "no significant movement"
    else:  # up/down
        if up_initial > threshold:
            first_movement = "up"
        elif up_initial < -threshold:
            first_movement = "down"
        else:
            first_movement = "no significant movement"

    if max_final_idx == 0:  # forward/backward
        if fwd_final > threshold:
            second_movement = "forward"
        elif fwd_final < -threshold:
            second_movement = "backward"
        else:
            second_movement = "no significant movement"
    elif max_final_idx == 1:  # right/left
        if right_final > threshold:
            second_movement = "right"
        elif right_final < -threshold:
            second_movement = "left"
        else:
            second_movement = "no significant movement"
    else:  # up/down
        if up_final > threshold:
            second_movement = "up"
        elif up_final < -threshold:
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

    all_possible_answers = list(set(all_possible_answers)) + directions

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
    last_frame_idx = get_random_integer(n_frames * FRAME_STRIDE, all_timesteps) #
    first_frame_idx = last_frame_idx - (n_frames * FRAME_STRIDE)

    all_frames_idx = list(world_state["simulation"].keys())

    timestep_frame_0 = all_frames_idx[first_frame_idx]
    timestep_mid = all_frames_idx[first_frame_idx + (n_frames//2)* FRAME_STRIDE]
    timestep_final = all_frames_idx[last_frame_idx - FRAME_STRIDE]

    given_sequence = uniformly_sample_frames_start_end_delta(
        first_frame_idx,
        last_frame_idx - FRAME_STRIDE,
        FRAME_STRIDE,
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