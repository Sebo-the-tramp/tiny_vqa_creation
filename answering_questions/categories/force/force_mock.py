"""
Mock implementations for force-related question resolvers.

Each function below mirrors the semantics of a question template and produces
an answer using the provided ``world_state`` and ``question`` payloads.
"""

from __future__ import annotations

import math

from typing import Any, Mapping, Union

from utils.decorators import with_resolved_attributes
from utils.helpers import get_object_state_at_timestep, _iter_objects
from utils.all_objects import get_all_objects_names
from utils.frames_selection import uniformly_sample_frames

from utils.bin_creation import (
    create_mc_options_around_gt,
    create_mc_object_names_from_dataset,
    uniform_labels,
)

Number = Union[int, float]
WorldState = Mapping[str, Any]
QuestionPayload = Mapping[str, Any]
Answer = Union[int, float, str]

MOVEMENT_TOLERANCE = 1e-3
DEFAULT_DISPLACEMENT_THRESHOLD = 2.0

## --- Resolver functions -- ##


@with_resolved_attributes
def F_FORCE_OF_GRAVITY_OBJECT_TIME(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> int:
    assert len(resolved_attributes) == 2 and "OBJECT" in resolved_attributes and "TIME"

    # In general this question is composite, because we need to estimate mass, the force of gravity is mass * gravity_acceleration
    # where gravity_acceleration is given by the scene configuration or assumed to be 9.81 m/s^2 downwards if not given

    timestep = resolved_attributes["TIME"]["choice"]
    object = resolved_attributes["OBJECT"]["choice"]

    gravity_vector = world_state["config"]["scene"]["gravity"]

    force_of_gravity = -gravity_vector[2] * object["mass"]
    options, correct_idx = create_mc_options_around_gt(
        force_of_gravity, num_answers=4, display_decimals=1, lo=0.0, min_rel_gap=0.2
    )
    labels = uniform_labels(options, integer=False, decimals=1)
    labels = [str(label) + " Newtons" for label in labels]

    # TODO maybe this could also be changed to create a series of images around the time step
    imgs_idx = uniformly_sample_frames(world_state)

    return question, labels, correct_idx, imgs_idx


@with_resolved_attributes
def F_FORCES_NET_FORCE_OBJECT_TIME(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> int:
    assert len(resolved_attributes) == 2 and "OBJECT" in resolved_attributes and "TIME"

    timestep = resolved_attributes["TIME"]["choice"]
    object = resolved_attributes["OBJECT"]["choice"]

    object_state = get_object_state_at_timestep(world_state, object["id"], timestep)
    # TODO check about the availability of this variable
    net_force = object_state["net_force"]
    net_force_magnitude = math.sqrt(sum(f**2 for f in net_force))
    options, correct_idx = create_mc_options_around_gt(
        net_force_magnitude, num_answers=4, display_decimals=1, lo=0.0
    )
    labels = uniform_labels(options, integer=False, decimals=1)
    labels = [str(label) + " Newtons" for label in labels]

    return question, labels, correct_idx


@with_resolved_attributes
def F_FORCES_TORQUE_OBJECT_TIME(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> int:
    assert len(resolved_attributes) == 2 and "OBJECT" in resolved_attributes and "TIME"

    timestep = resolved_attributes["TIME"]["choice"]
    object = resolved_attributes["OBJECT"]["choice"]

    object_state = get_object_state_at_timestep(world_state, object["id"], timestep)
    # TODO check about the availability of this variable
    torque = object_state["torque"]
    torque_magnitude = math.sqrt(sum(t**2 for t in torque))
    options, correct_idx = create_mc_options_around_gt(
        torque_magnitude, num_answers=4, display_decimals=1, lo=0.0
    )
    labels = uniform_labels(options, integer=False, decimals=1)
    labels = [str(label) + " Newtons*meters" for label in labels]

    return question, labels, correct_idx


@with_resolved_attributes
def F_FORCES_NORMAL_FORCE_OBJECT_TIME(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> int:
    assert len(resolved_attributes) == 2 and "OBJECT" in resolved_attributes and "TIME"

    timestep = resolved_attributes["TIME"]["choice"]
    object = resolved_attributes["OBJECT"]["choice"]

    object_state = get_object_state_at_timestep(world_state, object["id"], timestep)
    # TODO check about the availability of this variable
    normal_force = object_state["normal_force"]
    normal_force_magnitude = math.sqrt(sum(nf**2 for nf in normal_force))
    options, correct_idx = create_mc_options_around_gt(
        normal_force_magnitude, num_answers=4, display_decimals=1, lo=0.0
    )
    labels = uniform_labels(options, integer=False, decimals=1)
    labels = [str(label) + " Newtons" for label in labels]

    return question, labels, correct_idx


@with_resolved_attributes
def F_FORCES_HIGHEST_FORCE_OBJECT(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> int:
    assert len(resolved_attributes) == 1 and "TIME" in resolved_attributes

    timestep = resolved_attributes["TIME"]["choice"]

    max_force = -1.0
    object_with_max_force = None

    for obj in _iter_objects(world_state):
        obj_state = get_object_state_at_timestep(world_state, obj["id"], timestep)
        # TODO check about the availability of this variable
        net_force = obj_state["net_force"]
        net_force_magnitude = math.sqrt(sum(f**2 for f in net_force))
        if "max_force" not in locals() or net_force_magnitude > max_force:
            max_force = net_force_magnitude
            object_with_max_force = obj

    present = [
        obj["name"]
        for obj in list(_iter_objects(world_state))
        if obj["id"] != object_with_max_force["id"]
    ]

    labels, idx = create_mc_object_names_from_dataset(
        object_with_max_force["name"], present, get_all_objects_names()
    )

    return question, labels, idx
