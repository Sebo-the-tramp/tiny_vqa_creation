"""
Mock temporal reasoning resolvers.

These helpers extract best-effort temporal answers from the provided world state.
They operate on lightweight metadata (positions, orientations, region tags, etc.)
and fall back to sensible defaults when information is missing.
"""

from __future__ import annotations

from categories.temporal.temporal_helpers import (
    calculate_most_dissimilar_confounding_images,
)
from utils.decorators import with_resolved_attributes
from utils.frames_selection import uniformly_sample_frames_start_end_delta
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
    get_random_integer,
    resolve_attributes_visible_at_timestep
)

from utils.bin_creation import create_mc_object_names_from_dataset

import random
import numpy as np

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
FRAME_INTERLEAVE = get_config()["frame_interleave"]
MIN_PIXELS_VISIBLE = get_config()["min_pixels_visible"]
CLIP_LENGTH = get_config()["clip_length"]

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

    if max_frame <= min_frame:
        raise ImpossibleToAnswer("Not enough frames to sample the sequence with the given interleave.")

    start_frame = get_random_integer(min_frame, max_frame)
    end_frame = start_frame + (n_frames * FRAME_INTERLEAVE)

    imgs_idx = uniformly_sample_frames_start_end_delta(
        start_frame, end_frame, FRAME_INTERLEAVE
    )

    imgs_idx_shuffled = imgs_idx.copy()
    random.shuffle(imgs_idx_shuffled)

    choices = ["A", "B", "C", "D"]

    pair_choice_imgs_idx = [
        (imgs_idx_shuffled[i], choices[i]) for i in range(len(choices))
    ]

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

    return [[question, labels, correct_index, imgs_idx_shuffled, world_state, {}]]


def F_TEMPORAL_PREDICTION_NEXT_IMAGE(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> Sequence[str]:
    """here we select a sequence of uniformly sampled images and return the next with random
    position in the simulations steps"""
    assert len(attributes) == 0
    n_frames = 5

    _FRAME_INTERLEAVE = kwargs['frame_interleave']

    total_frames = len(world_state["simulation"]) // 3
    min_frame = 0
    max_frame = total_frames - (n_frames * _FRAME_INTERLEAVE) - 1

    if max_frame <= min_frame:
        raise ImpossibleToAnswer("Not enough frames to sample the sequence with the given interleave.")
    
    start_frame = get_random_integer(min_frame, max_frame)
    end_frame = start_frame + (n_frames * _FRAME_INTERLEAVE)

    all_frames_idx = uniformly_sample_frames_start_end_delta(0, total_frames, 1)

    sequence_idx = uniformly_sample_frames_start_end_delta(
        start_frame, end_frame, _FRAME_INTERLEAVE
    )

    given_sequence = sequence_idx[:4]
    next_image = sequence_idx[4]

    confounding_images_candidates = (
        all_frames_idx[:start_frame] + all_frames_idx[end_frame:]
    )
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

    return [[question, labels, correct_index, given_sequence, world_state, {}]]

@with_resolved_attributes
def F_TEMPORAL_PREDICTION_NEXT_IMAGE_GRANULARITY_1(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> Sequence[str]:
    kwargs["frame_interleave"] = 1
    return F_TEMPORAL_PREDICTION_NEXT_IMAGE(
        world_state, question, attributes, **kwargs
    )
    

@with_resolved_attributes
def F_TEMPORAL_PREDICTION_NEXT_IMAGE_GRANULARITY_2(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> Sequence[str]:
    kwargs["frame_interleave"] = 2
    return F_TEMPORAL_PREDICTION_NEXT_IMAGE(
        world_state, question, attributes, **kwargs
    )

@with_resolved_attributes
def F_TEMPORAL_PREDICTION_NEXT_IMAGE_GRANULARITY_5(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> Sequence[str]:
    kwargs["frame_interleave"] = 5
    return F_TEMPORAL_PREDICTION_NEXT_IMAGE(
        world_state, question, attributes, **kwargs
    )


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

    if max_frame <= min_frame:
        raise ImpossibleToAnswer("Not enough frames to sample the sequence with the given interleave.")

    start_frame = get_random_integer(min_frame, max_frame)
    end_frame = start_frame + (n_frames * FRAME_INTERLEAVE)

    all_frames_idx = uniformly_sample_frames_start_end_delta(0, total_frames, 1)

    sequence_idx = uniformly_sample_frames_start_end_delta(
        start_frame, end_frame, FRAME_INTERLEAVE
    )

    given_sequence = sequence_idx[1:]
    previous_image = sequence_idx[0]

    confounding_images_candidates = (
        all_frames_idx[:start_frame] + all_frames_idx[end_frame:]
    )
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

    return [[question, labels, correct_index, given_sequence, world_state, {}]]


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
    
    if max_frame <= min_frame:
        raise ImpossibleToAnswer("Not enough frames to sample the sequence with the given interleave.")
    
    start_frame = get_random_integer(min_frame, max_frame)
    end_frame = start_frame + (n_frames * FRAME_INTERLEAVE) - 1

    all_frames_idx = uniformly_sample_frames_start_end_delta(0, total_frames, 1)

    sequence_idx = uniformly_sample_frames_start_end_delta(
        start_frame, end_frame, FRAME_INTERLEAVE
    )

    # 0 - start_frame - 2*FRAME_INTERLEAVE ... start_frame ... end_frame ... end_frame + 2*FRAME_INTERLEAVE - total_frames
    first_possible = max(0, start_frame - 2 * FRAME_INTERLEAVE)
    last_possible = min(total_frames, end_frame + 2 * FRAME_INTERLEAVE)

    index_of_image_to_remove = get_random_integer(0, 4)
    given_sequence = sequence_idx.copy()
    given_sequence.pop(index_of_image_to_remove)

    confounding_images_candidates = (
        all_frames_idx[:first_possible] + all_frames_idx[last_possible:]
    )
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

    return [[question, labels, correct_index, given_sequence, world_state, {}]]


@with_resolved_attributes
def F_CAMERA_MOTION_DIRECTION(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> Sequence[str]:
    assert len(attributes) == 0
    n_frames = 8
    all_timesteps = len(world_state["simulation"]) // 3
    if(n_frames * FRAME_INTERLEAVE >= all_timesteps):
        raise ImpossibleToAnswer("Not enough frames to determine camera motion direction.")
    
    last_frame_idx = get_random_integer(n_frames * FRAME_INTERLEAVE, all_timesteps)
    first_frame_idx = last_frame_idx - (n_frames * FRAME_INTERLEAVE)

    all_frames_idx = list(world_state["simulation"].keys())

    timestep_frame_0 = all_frames_idx[first_frame_idx]
    timestep_mid = all_frames_idx[first_frame_idx + (n_frames // 2) * FRAME_INTERLEAVE]
    timestep_final = all_frames_idx[last_frame_idx - FRAME_INTERLEAVE]

    given_sequence = uniformly_sample_frames_start_end_delta(
        first_frame_idx,
        last_frame_idx - FRAME_INTERLEAVE,
        FRAME_INTERLEAVE,
    )

    initial_look_at = np.array(
        world_state["simulation"][timestep_frame_0]["camera"]["at"]
    )
    initial_eye = np.array(world_state["simulation"][timestep_frame_0]["camera"]["eye"])
    mid_eye = np.array(world_state["simulation"][timestep_mid]["camera"]["eye"])
    final_eye = np.array(world_state["simulation"][timestep_final]["camera"]["eye"])

    initial_direction = initial_look_at - initial_eye
    initial_direction = initial_direction / np.linalg.norm(initial_direction)
    right_vector = np.cross(
        np.array([0, 0, -1]), initial_direction
    )  # not sure about the sign here though
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
    if abs(projection_initial_to_mid[0]) > abs(projection_initial_to_mid[1]) and abs(
        projection_initial_to_mid[0]
    ) > abs(projection_initial_to_mid[2]):
        # forward/backward
        if projection_initial_to_mid[0] > threshold:
            first_movement = "forward"
        elif projection_initial_to_mid[0] < -threshold:
            first_movement = "backward"
        else:
            first_movement = "no significant movement"
    elif abs(projection_initial_to_mid[1]) > abs(projection_initial_to_mid[0]) and abs(
        projection_initial_to_mid[1]
    ) > abs(projection_initial_to_mid[2]):
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
    if abs(projection_mid_to_final[0]) > abs(projection_mid_to_final[1]) and abs(
        projection_mid_to_final[0]
    ) > abs(projection_mid_to_final[2]):
        # forward/backward
        if projection_mid_to_final[0] > threshold:
            second_movement = "forward"
        elif projection_mid_to_final[0] < -threshold:
            second_movement = "backward"
        else:
            second_movement = "no significant movement"
    elif abs(projection_mid_to_final[1]) > abs(projection_mid_to_final[0]) and abs(
        projection_mid_to_final[1]
    ) > abs(projection_mid_to_final[2]):
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
    if (
        first_movement == "no significant movement"
        and second_movement == "no significant movement"
    ):
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
    labels = other_answers[:correct_index] + [answer] + other_answers[correct_index:]
    return [[question, labels, correct_index, given_sequence, world_state, {}]]


@with_resolved_attributes
def F_CAMERA_ZOOM_BEHAVIOR(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> Sequence[str]:
    assert len(attributes) == 0

    n_frames = 8

    all_timesteps = len(world_state["simulation"])
    if(n_frames * FRAME_INTERLEAVE >= all_timesteps):
        raise ImpossibleToAnswer("Not enough frames to determine camera motion direction.")
    
    last_frame_idx = get_random_integer(n_frames * FRAME_INTERLEAVE, all_timesteps)  #
    first_frame_idx = last_frame_idx - (n_frames * FRAME_INTERLEAVE)

    all_frames_idx = list(world_state["simulation"].keys())

    timestep_frame_0 = all_frames_idx[first_frame_idx]
    timestep_mid = all_frames_idx[first_frame_idx + (n_frames // 2) * FRAME_INTERLEAVE]
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

    all_possible_answers = [
        "zoom in",
        "zoom out",
        "no zoom",
        "zoom in then out",
        "zoom out then in",
    ]

    other_answers = [ans for ans in all_possible_answers if ans != answer]
    random.shuffle(other_answers)
    other_answers = other_answers[:3]
    correct_index = get_random_integer(0, 3)
    labels = other_answers[:correct_index] + [answer] + other_answers[correct_index:]

    return [[question, labels, correct_index, given_sequence, world_state, {}]]


# I think this is the most important one

def check_visibility_sequence(
    world_state: WorldState, obj_id: str, all_timesteps: Sequence[str],
    min_ones: int = 4, min_zeros: int = 4, gap_limit: int = 1
) -> Sequence[int]:

    phase = "ones"  # expecting ones-block first

    ones_count = zeros_count = 0
    ones_gaps = zeros_gaps = 0
    final_timestep = None
    found = False

    for t in all_timesteps[::FRAME_INTERLEAVE]:
        obj_states = world_state["simulation"][t]["objects"]
        visible = (
            obj_id in obj_states
            and obj_states[obj_id]["infov_pixels"] >= MIN_PIXELS_VISIBLE
        )
        bit = 1 if visible else 0

        if phase == "ones":
            if bit == 1:
                ones_count += 1
            else:
                ones_gaps += 1

            # If gap tolerance exceeded → reset
            if ones_gaps > gap_limit:
                ones_count = ones_gaps = 0

            # If ones-block satisfied → move to zeros phase
            if ones_count >= min_ones:
                phase = "zeros"

        elif phase == "zeros":
            if bit == 0:
                zeros_count += 1
            else:
                zeros_gaps += 1

            # If gap tolerance exceeded → reset zeros-phase
            if zeros_gaps > gap_limit:
                zeros_count = zeros_gaps = 0

            # Full pattern detected
            if zeros_count >= min_zeros:
                found = True
                final_timestep = t
                break

    return found, final_timestep

def compute_visibility_counts(world_state, timesteps):
    counts = []

    for t in timesteps:
        visible_count = 0
        objects = world_state["simulation"][t]["objects"]

        for obj in objects.values():
            if obj["infov_pixels"] >= MIN_PIXELS_VISIBLE:
                visible_count += 1

        counts.append(visible_count)

    return counts

def find_first_visibility_drop(counts):
    for i in range(1, len(counts)):
        if counts[i] < counts[i-1]:       # drop ≥ 1
            return i                      # index of timestep where drop starts
    return None


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