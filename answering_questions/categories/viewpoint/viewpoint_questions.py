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

import math
import random

from utils.config import get_config
from utils.all_objects import get_all_objects_names
from utils.decorators import with_resolved_attributes
from utils.bin_creation import create_mc_object_names_from_dataset
from utils.my_exception import ImpossibleToAnswer

from utils.helpers import (
    get_random_timestep_from_list,
    iter_objects,
    fill_questions,
    resolve_attributes,
    get_visibility_mask,
    get_timestep_from_idx,
    get_camera_at_timestep,
    resolve_attributes_visible_at_timestep,
    resolve_attributes_most_visible_at_timestep,
    get_visible_timesteps_for_attributes_min_objects,
)

from categories.viewpoint.viewpoint_helpers import (
    infer_world_up,
    forward,
    pitch_deg,
    classify_camera_angle_index,
    horizontal_fov_rad,
    classify_focal_length_index,
    get_number_of_visible_objects,
)

Number = Union[int, float]
Vector = Tuple[float, float, float]
WorldState = Mapping[str, Any]
QuestionPayload = Mapping[str, Any]
Answer = Union[str, float, Vector, Mapping[str, Any], Sequence[str]]

CLIP_LENGTH = get_config()["clip_length"]
FRAME_INTERLEAVE = get_config()["frame_interleave"]
MIN_VISIBLE_PIXELS = get_config()["min_pixels_visible"]


## --- Resolver functions -- ##
@with_resolved_attributes
def F_VISIBILITY_OBJECT(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    assert len(attributes) == 0

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        ["OBJECT"],
        world_state,
        min_objects=min(kwargs["current_world_number_of_objects"], 2),
    )

    final_timestep = get_random_timestep_from_list(visible_timesteps, question)
    final_timestep_index = world_state["simulation"][final_timestep]["frame_idx"]

    candidates = [
        k for k in (1, 2, 3, 4) if final_timestep_index - (k * (CLIP_LENGTH - 1)) >= 0
    ]
    if len(candidates) == 0:
        raise ImpossibleToAnswer("Not enough previous frames to determine visibility.")

    max_k = max(candidates)
    initial_timestep_index = final_timestep_index - (max_k * (CLIP_LENGTH - 1))
    initial_timestep = get_timestep_from_idx(initial_timestep_index)

    # resolve attributes MOST visible
    resolved_attributes = resolve_attributes_most_visible_at_timestep(
        ["OBJECT"], world_state, final_timestep
    )

    object = resolved_attributes["OBJECT"]["choice"]

    presents = [obj["name"] for obj in iter_objects(world_state)]
    all_objects = get_all_objects_names()

    all_objects_minus_present = [obj for obj in all_objects if obj not in presents]

    labels, correct_idx = create_mc_object_names_from_dataset(
        object["name"], all_objects_minus_present, [], num_answers=4
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
def F_VISIBILITY_OBJECT_COUNT(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    assert len(attributes) == 0

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        ["OBJECT"], world_state, min_objects=2
    )

    final_timestep = get_random_timestep_from_list(visible_timesteps, question)
    final_timestep_index = world_state["simulation"][final_timestep]["frame_idx"]

    candidates = [
        k for k in (1, 2, 3, 4) if final_timestep_index - (k * (CLIP_LENGTH - 1)) >= 0
    ]
    if len(candidates) == 0:
        raise ImpossibleToAnswer("Not enough previous frames to determine visibility.")

    max_k = max(candidates)
    initial_timestep_index = final_timestep_index - (max_k * (CLIP_LENGTH - 1))
    initial_timestep = get_timestep_from_idx(initial_timestep_index)

    total_visible_objects = get_number_of_visible_objects(world_state, final_timestep)

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
        ["OBJECT"], world_state, final_timestep
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
def F_OCCLUSION_PERCENTAGE_OBJECT(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    assert len(attributes) == 1 and "OBJECT" in attributes

    # I mean it doesn't have to be visible at all it can just be any timestep
    final_timestep = get_random_timestep_from_list(
        list(world_state["simulation"].keys())[CLIP_LENGTH:], question
    )
    final_timestep_index = world_state["simulation"][final_timestep]["frame_idx"]

    _, visibility_percentage_matrix = get_visibility_mask(
        world_state, max_timestep=final_timestep
    )

    candidates = [
        k for k in (1, 2, 3, 4) if final_timestep_index - (k * (CLIP_LENGTH - 1)) >= 0
    ]
    if len(candidates) == 0:
        raise ImpossibleToAnswer("Not enough previous frames to determine visibility.")

    max_k = max(candidates)
    initial_timestep_index = final_timestep_index - (max_k * (CLIP_LENGTH - 1))
    initial_timestep = get_timestep_from_idx(initial_timestep_index)

    # First we find the pairs of objects visible
    # also the object can be any object in the scene not necessarily visible
    resolved_attributes = resolve_attributes(["OBJECT-RANDOM"], world_state)

    resolved_attributes["OBJECT"] = resolved_attributes.pop("OBJECT-RANDOM")
    object_id = resolved_attributes["OBJECT"]["choice"]["id"]

    visibility_object = (
        visibility_percentage_matrix[int(object_id) - 1, final_timestep_index] / 100.0
    )

    if visibility_object < 0.25:
        correct_idx = 0
    elif visibility_object < 0.65:
        correct_idx = 1
    elif visibility_object < 0.95:
        correct_idx = 2
    else:
        correct_idx = 3

    labels = [
        "Severely Occluded (0-25% visible)",  # Hard: Requires context/guessing
        "Partially Occluded (25-65% visible)",  # Medium: Major parts missing
        "Slightly Occluded (65-95% visible)",  # Easy: Minor obstructions
        "Fully Visible (>95% visible)",  # Control: Clean object
    ]

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
def F_VIEWPOINT_CAMERA_ANGLE(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    """
    Maps camera pose to one of:
    ["low angle","eye level","high angle","bird's-eye","worm's-eye"]
    """
    assert len(attributes) == 0

    resolved_attributes = resolve_attributes([], world_state)

    all_timesteps = list(world_state["simulation"].keys())

    if len(all_timesteps) <= CLIP_LENGTH * FRAME_INTERLEAVE - FRAME_INTERLEAVE:
        raise ImpossibleToAnswer("Not enough timesteps in the simulation.")

    if "multi" in question.get("task_splits", ""):
        timestep = random.choice(
            all_timesteps[CLIP_LENGTH * FRAME_INTERLEAVE - FRAME_INTERLEAVE :]
        )
    else:
        timestep = random.choice(all_timesteps)

    cam = get_camera_at_timestep(world_state, timestep)

    eye = cam["eye"]
    at = cam["at"]
    up_cam = cam["up"]

    fwd = forward(eye, at)
    world_up = infer_world_up(world_state, up_cam)
    pitch = pitch_deg(fwd, world_up)

    labels, correct_idx = classify_camera_angle_index(pitch)
    return fill_questions(
        question, labels, correct_idx, world_state, timestep, resolved_attributes
    )


@with_resolved_attributes
def F_FOCAL_LENGTH_CLASS(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    assert len(attributes) == 0
    """
    Maps FOV to one of:
    ["ultra-wide","wide","normal","short telephoto","telephoto"]
    Uses horizontal FOV for stable categorization across aspect ratios.
    """
    resolved_attributes = resolve_attributes([], world_state)

    all_timesteps = list(world_state["simulation"].keys())

    if len(all_timesteps) <= CLIP_LENGTH * FRAME_INTERLEAVE - FRAME_INTERLEAVE:
        raise ImpossibleToAnswer("Not enough timesteps in the simulation.")

    if "multi" in question.get("task_splits", ""):
        timestep = random.choice(
            all_timesteps[CLIP_LENGTH * FRAME_INTERLEAVE - FRAME_INTERLEAVE :]
        )
    else:
        timestep = random.choice(all_timesteps)

    cam = get_camera_at_timestep(world_state, timestep)

    fov = cam["fov"]
    width = cam["width"]
    height = cam["height"]

    # If your camera dict exposes an axis flag, honor it; else assume vertical FOV.
    fov_axis = cam.get("fov_axis", "vertical")
    hfov_rad = horizontal_fov_rad(fov, width, height, fov_axis=fov_axis)
    hfov_deg = math.degrees(hfov_rad)

    labels, correct_idx = classify_focal_length_index(hfov_deg)
    return fill_questions(
        question, labels, correct_idx, world_state, timestep, resolved_attributes
    )
