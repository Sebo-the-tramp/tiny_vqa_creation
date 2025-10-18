"""
Mock kinematics reasoning resolvers.

These helpers extract best-effort kinematics answers from the provided world state.
They operate on lightweight metadata (positions, orientations, region tags, etc.)
and fall back to sensible defaults when information is missing.
"""

from __future__ import annotations

import math
import numpy as np

from typing import Any, Mapping, Union

from utils.decorators import with_resolved_attributes
from utils.all_objects import get_all_objects_names
from utils.frames_selection import uniformly_sample_frames

from utils.bin_creation import (
    create_mc_options_around_gt,
    create_mc_object_names_from_dataset,
    uniform_labels,
)

from utils.helpers import (
    _get_displacement,
    _get_speed,
    _get_acceleration,
    get_angular_velocity_vector,
    _is_moving,
    _iter_objects,
    _objects_of_type,
    _distance_between,
)

from utils.helpers import get_object_state_at_timestep

Number = Union[int, float]
WorldState = Mapping[str, Any]
QuestionPayload = Mapping[str, Any]
Answer = Union[int, float, str]

MOVEMENT_TOLERANCE = 1e-3
DEFAULT_DISPLACEMENT_THRESHOLD = 2.0


@with_resolved_attributes
def F_KINEMATICS_VELOCITY_OBJECT_AT_TIMESTEP(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> int:
    """Return the velocity of the object referenced in the question."""

    assert (
        len(resolved_attributes) == 2
        and "OBJECT" in resolved_attributes
        and "TIME" in resolved_attributes
    )
    object_id = resolved_attributes["OBJECT"]["choice"]["id"]
    timestep = resolved_attributes["TIME"]["choice"]

    velocity_object_at_timestep = _get_speed(object_id, timestep, world_state)

    # create the answer options
    options, correct_idx = create_mc_options_around_gt(
        velocity_object_at_timestep, num_answers=4, min_rel_gap=0.3
    )
    # add units to options
    options_with_units = [f"{opt} m/s" for opt in options]
    return (
        question,
        options_with_units,
        correct_idx,
        uniformly_sample_frames(world_state),
    )


@with_resolved_attributes
def F_KINEMATICS_FASTEST_OBJECT_AT_TIMESTEP(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> float:
    """Return the object with highest velocity at timestep."""
    assert len(resolved_attributes) == 1 and "TIME" in resolved_attributes

    timestep = resolved_attributes["TIME"]["choice"]

    velocities = []
    for obj in _iter_objects(world_state):
        obj_state = get_object_state_at_timestep(world_state, obj["id"], timestep)
        velocities.append((obj, round(obj_state["kinematics"]["speed"], 2)))

    # get the fastest object
    # check if they have the same speed
    same_speed = all((velocities[0][1] - v[1] < 1e-6) for v in velocities)
    if same_speed:
        present = [obj["name"] for obj in list(_iter_objects(world_state))]
        correct_answer = "They all have the same speed"
    else:
        fastest_object, _ = max(velocities, key=lambda x: x[1])
        present = [
            obj["name"]
            for obj in list(_iter_objects(world_state))
            if obj["id"] != fastest_object["id"]
        ]
        correct_answer = fastest_object["name"]

    labels, idx = create_mc_object_names_from_dataset(
        correct_answer, present, get_all_objects_names()
    )

    return question, labels, idx, uniformly_sample_frames(world_state)


@with_resolved_attributes
def F_KINEMATICS_ESTIMATION_FASTEST_VELOCITY(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> float:
    """Return the velocity of the fastest moving object."""
    assert len(resolved_attributes) == 0

    max_velocity = 0.0
    for timestep in world_state["simulation"]:
        velocitys = [
            _get_speed(obj["id"], timestep, world_state)
            for obj in _iter_objects(world_state)
        ]
        max_velocity_at_timestep = max(velocitys, default=0.0)
        if max_velocity_at_timestep > max_velocity:
            max_velocity = max_velocity_at_timestep

    options, correct_idx = create_mc_options_around_gt(
        max_velocity, num_answers=4, lo=0.0
    )
    options_with_units = [f"{opt} m/s" for opt in options]
    return (
        question,
        options_with_units,
        correct_idx,
        uniformly_sample_frames(world_state),
    )


@with_resolved_attributes
def F_KINEMATICS_ACCEL_OBJECT_TIMESTEP(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> float:
    """Return the acceleration of the object referenced in the question."""
    assert (
        len(resolved_attributes) == 2
        and "TIME" in resolved_attributes
        and "OBJECT" in resolved_attributes
    )
    timestep = resolved_attributes["TIME"]["choice"]
    object_id = resolved_attributes["OBJECT"]["choice"]["id"]

    acceleration_object_at_timestep = _get_acceleration(
        object_id, timestep, world_state
    )

    options, correct_idx = create_mc_options_around_gt(
        acceleration_object_at_timestep, num_answers=4, lo=0.0, display_decimals=2
    )
    labels = uniform_labels(options, integer=False, decimals=2)
    labels = [f"{opt} m/s^2" for opt in labels]
    return question, labels, correct_idx, uniformly_sample_frames(world_state)


@with_resolved_attributes
def F_KINEMATICS_ANGULAR_VELOCITY_OBJECT_TIMESTEP(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> float:
    """Return the angular velocity of the object referenced in the question."""
    assert (
        len(resolved_attributes) == 2
        and "TIME" in resolved_attributes
        and "OBJECT" in resolved_attributes
    )
    timestep = resolved_attributes["TIME"]["choice"]
    object_id = resolved_attributes["OBJECT"]["choice"]["id"]

    angular_velocity_object_at_timestep = get_angular_velocity_vector(
        object_id, timestep, world_state
    )

    # convert from 3_vector to scalar (magnitude)
    magnitude = math.sqrt(sum(x**2 for x in angular_velocity_object_at_timestep))

    options, correct_idx = create_mc_options_around_gt(
        magnitude, num_answers=4, min_rel_gap=0.3
    )
    options_with_units = [
        f"{opt} deg/s" for opt in options
    ]  # TODO also need to agree on units
    return (
        question,
        options_with_units,
        correct_idx,
        uniformly_sample_frames(world_state),
    )


@with_resolved_attributes
def F_KINEMATICS_PEAK_VELOCITY_TIMESTEP(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> float:
    assert len(resolved_attributes) == 1 and "OBJECT" in resolved_attributes

    object = resolved_attributes["OBJECT"]["choice"]

    max_velocity = 0.0

    for timestep, value in world_state["simulation"].items():
        object_state = get_object_state_at_timestep(world_state, object["id"], timestep)
        if object_state["kinematics"]["speed"] > max_velocity:
            max_velocity = object_state["kinematics"]["speed"]

    options, correct_idx = create_mc_options_around_gt(
        max_velocity, num_answers=4, display_decimals=1, lo=0.0, min_rel_gap=0.3
    )
    labels = uniform_labels(options, integer=False, decimals=1)
    labels = [f"{opt} m/s^2" for opt in labels]

    return question, labels, correct_idx, uniformly_sample_frames(world_state)


@with_resolved_attributes
def F_KINEMATICS_DISTANCE_TRAVELED_INTERVAL(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> int:
    """Count objects of a specific type that moved more than a given metric distance."""
    assert (
        len(resolved_attributes) == 3
        and "OBJECT" in resolved_attributes
        and "TIME_1" in resolved_attributes
        and "TIME_2" in resolved_attributes
    )

    object = resolved_attributes["OBJECT"]["choice"]

    timestep_start = resolved_attributes["TIME_1"]["choice"]
    timestep_end = resolved_attributes["TIME_2"]["choice"]

    obj_state_timestep_start = get_object_state_at_timestep(
        world_state, object["id"], timestep_start
    )
    obj_state_timestep_end = get_object_state_at_timestep(
        world_state, object["id"], timestep_end
    )

    center_obj_state_timestep_start = obj_state_timestep_start["obb"]["center"]
    center_obj_state_timestep_end = obj_state_timestep_end["obb"]["center"]

    distance = _distance_between(
        center_obj_state_timestep_start, center_obj_state_timestep_end
    )

    options, correct_idx = create_mc_options_around_gt(
        distance, num_answers=4, display_decimals=1, lo=0.0, min_rel_gap=0.2
    )
    labels = uniform_labels(options, integer=False, decimals=1)
    labels = [f"{opt} meters" for opt in labels]

    # here we could sample from start_timestep to end_timestep
    return question, labels, correct_idx, uniformly_sample_frames(world_state)


@with_resolved_attributes
def F_KINEMATICS_DIRECTION_CHANGE_COUNT(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> int:
    assert len(resolved_attributes) == 1 and "OBJECT" in resolved_attributes

    object = resolved_attributes["OBJECT"]["choice"]

    previous_state_transform = get_object_state_at_timestep(
        world_state, object["id"], "0000.010"
    )["kinematics"]["linear_velocity_world"][:3]

    count_change_dir = 0
    for timestep, value in world_state["simulation"].items():
        current_state_transform = get_object_state_at_timestep(
            world_state, object["id"], timestep
        )["kinematics"]["linear_velocity_world"][:3]

        current_sign = np.array(current_state_transform) > 0.05
        previous_sign = np.array(previous_state_transform) > 0.05

        # XOR detects where the sign flipped between timesteps
        if np.any(current_sign ^ previous_sign):
            print(
                "timestep:",
                timestep,
                "sign changed",
                current_state_transform,
                previous_state_transform,
            )
            count_change_dir += 1

        previous_state_transform = current_state_transform

    options, correct_idx = create_mc_options_around_gt(
        count_change_dir, num_answers=4, display_decimals=0, lo=0.0
    )
    labels = uniform_labels(options, integer=True, decimals=0)

    return question, labels, correct_idx, uniformly_sample_frames(world_state)


@with_resolved_attributes
def F_KINEMATICS_COUNTING_MOVEMENT_OBJECT_CATEGORY(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> int:
    """Count objects of a specific type that are currently moving."""

    assert len(resolved_attributes) == 1 and "OBJECT-CATEGORY" in resolved_attributes

    # core logic
    object_category = resolved_attributes["OBJECT-CATEGORY"]["choice"]
    object_id_has_moved = {
        obj["id"]: False for obj in _objects_of_type(world_state, object_category)
    }
    for timestep in world_state["simulation"]:
        for obj in _objects_of_type(world_state, object_category):
            if not object_id_has_moved[obj["id"]] and _is_moving(
                obj["id"], timestep, world_state
            ):
                object_id_has_moved[obj["id"]] = True

    total_moved_objects = sum(1 for moved in object_id_has_moved.values() if moved)

    options, correct_idx = create_mc_options_around_gt(
        total_moved_objects, num_answers=4, display_decimals=0, lo=0.0
    )
    labels = uniform_labels(options, integer=False, decimals=0)

    return question, labels, correct_idx, uniformly_sample_frames(world_state)


@with_resolved_attributes
def F_KINEMATICS_COUNTING_MOVEMENT_OBJECT_CATEGORY_TIME(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> int:
    """Count objects of a specific type that are currently moving."""

    assert (
        len(resolved_attributes) == 2
        and "OBJECT-CATEGORY" in resolved_attributes
        and "TIME" in resolved_attributes
    )

    # core logic
    object_category = resolved_attributes["OBJECT-CATEGORY"]["choice"]
    timestep = resolved_attributes["TIME"]["choice"]
    count_moving = sum(
        1
        for obj in _objects_of_type(world_state, object_category)
        if _is_moving(obj["id"], timestep, world_state)
    )

    options, correct_idx = create_mc_options_around_gt(
        count_moving, num_answers=4, display_decimals=0, lo=0.0
    )
    labels = uniform_labels(options, integer=True, decimals=0)

    return question, labels, correct_idx, uniformly_sample_frames(world_state)


@with_resolved_attributes
def F_KINEMATICS_COUNTING_STILL(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> int:
    """Count objects of a specific type that are effectively still."""
    assert len(resolved_attributes) == 0

    still_objects = 0
    for obj in _iter_objects(world_state):
        obj_displacement = _get_displacement(
            obj["id"], kwargs["timestep_start"], kwargs["timestep_end"], world_state
        )
        if obj_displacement is None:
            raise ValueError(
                f"Cannot compute displacement for object {obj['id']} due to missing position data."
            )
        if obj_displacement < MOVEMENT_TOLERANCE:
            still_objects += 1

    options, correct_idx = create_mc_options_around_gt(
        still_objects, num_answers=4, display_decimals=0, lo=0.0
    )
    labels = uniform_labels(options, integer=True, decimals=0)
    return question, labels, correct_idx, uniformly_sample_frames(world_state)


@with_resolved_attributes
def F_KINEMATICS_COUNTING_STILL_OBJECT_CATEGORY(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> int:
    """Count objects of a specific type that are effectively still."""

    assert len(resolved_attributes) == 1 and "OBJECT-CATEGORY" in resolved_attributes
    OBJECT_CATEGORY = resolved_attributes["OBJECT-CATEGORY"]["choice"]["category_GSO"]

    still_objects = 0
    for obj in _objects_of_type(world_state, OBJECT_CATEGORY):
        obj_displacement = _get_displacement(
            obj["id"], kwargs["timestep_start"], kwargs["timestep_end"], world_state
        )
        if obj_displacement is None:
            raise ValueError(
                f"Cannot compute displacement for object {obj['id']} due to missing position data."
            )
        if obj_displacement < MOVEMENT_TOLERANCE:
            still_objects += 1

    options, correct_idx = create_mc_options_around_gt(
        still_objects, num_answers=4, display_decimals=0, lo=0.0
    )
    labels = uniform_labels(options, integer=True, decimals=0)
    return question, labels, correct_idx, uniformly_sample_frames(world_state)
