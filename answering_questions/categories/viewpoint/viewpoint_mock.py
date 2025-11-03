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

from utils.my_exception import ImpossibleToAnswer

import math
import random

Number = Union[int, float]
Vector = Tuple[float, float, float]
WorldState = Mapping[str, Any]
QuestionPayload = Mapping[str, Any]
Answer = Union[str, float, Vector, Mapping[str, Any], Sequence[str]]

from utils.decorators import with_resolved_attributes

from utils.bin_creation import (
    create_mc_options_around_gt,
    uniform_labels,
    create_mc_object_names_from_dataset
)

from utils.helpers import (
    get_random_timestep_from_list,
    iter_objects,
    fill_questions,    
    distance_between,
    resolve_attributes,
    get_camera_at_timestep,
    get_object_state_at_timestep,
    resolve_attributes_visible_at_timestep,
    get_visible_timesteps_for_attributes_min_objects,
)

from categories.viewpoint.viewpoint_helpers import (
    infer_world_up,
    forward,
    pitch_deg,
    classify_camera_angle_index,
    horizontal_fov_rad,
    classify_focal_length_index,
)

from utils.all_objects import get_all_objects_names

from utils.config import get_config

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
        ["OBJECT"], world_state, min_objects=kwargs["current_world_number_of_objects"]
    )

    timestep = get_random_timestep_from_list(
        visible_timesteps, question
    ) 

    resolved_attributes = resolve_attributes_visible_at_timestep(
        ["OBJECT"], world_state, timestep
    )

    object = resolved_attributes["OBJECT"]['choice']

    presents = [obj["name"] for obj in iter_objects(world_state)]
    all_objects = get_all_objects_names()

    all_objects_minus_present = [obj for obj in all_objects if obj not in presents]

    labels, correct_idx = create_mc_object_names_from_dataset(
        object["name"], all_objects_minus_present, [], num_answers=4
    )

    return fill_questions(
        question, labels, correct_idx, world_state, timestep, resolved_attributes
    )    


@with_resolved_attributes
def F_VISIBILITY_PERCENTAGE_OBJECT(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:    
    assert len(attributes) == 1 and "OBJECT" in attributes

    # First we find the pairs of objects visible
    resolved_attributes = resolve_attributes(
        ["OBJECT"], world_state
    )

    all_timesteps = list(world_state["simulation"].keys())

    if "multi" in question.get("task_splits", ""):
        timestep = random.choice(all_timesteps[CLIP_LENGTH*FRAME_INTERLEAVE - FRAME_INTERLEAVE:])
    else:
        timestep = random.choice(all_timesteps)

    object = resolved_attributes["OBJECT"]['choice']
    visibility_object = get_object_state_at_timestep(
        world_state, object["id"], timestep
    )["fov_visibility"]

    if not world_state['simulation'][timestep]['objects'][object["id"]]['infov_pixels'] > MIN_VISIBLE_PIXELS:
        visibility_object = 0.0

    if visibility_object < 0.25:
        correct_idx = 0
    elif visibility_object < 0.5:
        correct_idx = 1
    elif visibility_object < 0.75:
        correct_idx = 2
    else:
        correct_idx = 3

    labels = ["0-25%", "26-50%", "51-75%", "76-100%"]

    return fill_questions(
        question, labels, correct_idx, world_state, timestep, resolved_attributes
    )


## --- Camera characteristics resolvers --- ##

@with_resolved_attributes
def F_VIEWPOINT_CAMERA_ANGLE(world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    """
    Maps camera pose to one of:
    ["low angle","eye level","high angle","bird's-eye","worm's-eye"]
    """
    assert len(attributes) == 0
    
    resolved_attributes = resolve_attributes([], world_state)

    all_timesteps = list(world_state["simulation"].keys())

    if "multi" in question.get("task_splits", ""):
        timestep = random.choice(all_timesteps[CLIP_LENGTH*FRAME_INTERLEAVE - FRAME_INTERLEAVE:])
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
def F_FOCAL_LENGTH_CLASS(world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    assert len(attributes) == 0
    """
    Maps FOV to one of:
    ["ultra-wide","wide","normal","short telephoto","telephoto"]
    Uses horizontal FOV for stable categorization across aspect ratios.
    """
    resolved_attributes = resolve_attributes([], world_state)

    all_timesteps = list(world_state["simulation"].keys())

    if "multi" in question.get("task_splits", ""):
        timestep = random.choice(all_timesteps[CLIP_LENGTH*FRAME_INTERLEAVE - FRAME_INTERLEAVE:])
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