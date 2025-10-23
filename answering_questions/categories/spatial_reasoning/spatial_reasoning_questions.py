"""
Mock spatial reasoning resolvers.

These helpers extract best-effort spatial answers from the provided world state.
They operate on lightweight metadata (positions, orientations, region tags, etc.)
and fall back to sensible defaults when information is missing.
"""

from __future__ import annotations

from utils.decorators import with_resolved_attributes
from utils.all_objects import get_all_objects_names
from utils.frames_selection import uniformly_sample_frames, sample_frames_at_timesteps, sample_frames_before_timestep

from typing import (
    Any,
    Mapping,
    Sequence,
    Tuple,
    Union,
)

import random
random.seed(42)

from utils.my_exception import ImpossibleToAnswer

from utils.helpers import (
    _distance_between,
    _iter_objects,
    get_object_state_at_timestep,
    get_random_timestep,
    _resolve_attributes,
    _resolve_attributes_visible_at_timestep,
    _fill_template,
    _get_visible_timesteps_for_attributes_min_objects,
)

from .spatial_reasoning_helpers import (
    _get_position,
    point_to_plane_distance,
    _get_position_camera,
    get_max_height_from_obb,
)

from utils.bin_creation import (
    create_mc_options_around_gt,
    create_mc_object_names_from_dataset,
    uniform_labels,
)

Number = Union[int, float]
Vector = Tuple[float, float, float]
WorldState = Mapping[str, Any]
QuestionPayload = Mapping[str, Any]
Answer = Union[str, float, Vector, Mapping[str, Any], Sequence[str]]


## --- Resolver functions -- ##

@with_resolved_attributes
def F_DISTANCE_OBJECT_OBJECT(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    assert (
        len(attributes) == 2
        and "OBJECT_1" in attributes
        and "OBJECT_2" in attributes        
    )

    # First we find the pairs of objects visible

    visible_timesteps = _get_visible_timesteps_for_attributes_min_objects(
        attributes, world_state, min_objects=2
    )
    # if we are in a multi-image setting, we need to ensure there are enough frames
    if len(visible_timesteps) == 0:
        raise ImpossibleToAnswer("No timestep with both objects visible.")
    if(question["task_splits"] == "multi"):
        visible_timesteps = visible_timesteps[7:]  # need at least 8 frames before
    
    timestep = random.choice(visible_timesteps)

    resolved_attributes = _resolve_attributes_visible_at_timestep(
        attributes, world_state, timestep
    )
    obj1_id = resolved_attributes["OBJECT_1"]["choice"]['id']
    obj2_id = resolved_attributes["OBJECT_2"]["choice"]['id']

    obj1_pos = _get_position(
        world_state, obj1_id, timestep      
    )
    obj2_pos = _get_position(
        world_state, obj2_id, timestep
    )

    distance = _distance_between(obj1_pos, obj2_pos)

    options, correct_idx = create_mc_options_around_gt(
        distance, num_answers=4, display_decimals=1, lo=0.0
    )
    labels = uniform_labels(options, integer=False, decimals=1)
    labels = [str(label) + " meters" for label in labels]

    _fill_template(question, resolved_attributes)

    questions = []
    if("single" in question["task_splits"]):
        questions.append(
            question,
            labels,
            correct_idx,
            sample_frames_before_timestep(world_state, timestep, num_frames=8, frame_interleave=1),
        )
    if("multi" in question["task_splits"]):
        questions.append(
            question,
            labels,
            correct_idx,
            sample_frames_before_timestep(world_state, timestep, num_frames=8, frame_interleave=1),
        )

    return questions