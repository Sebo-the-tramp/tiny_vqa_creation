"""
Mock implementations for force-related question resolvers.

Each function below mirrors the semantics of a question template and produces
an answer using the provided ``world_state`` and ``question`` payloads.
"""

from __future__ import annotations

from typing import Any, Mapping, Union

from utils.bin_creation import create_mc_options_around_gt, uniform_labels
from utils.decorators import with_resolved_attributes
from utils.helpers import (
    _coerce_to_float,
    _get_displacement,
    _get_speed,
    _get_acceleration,
    get_angular_velocity,
    _is_moving,
    _iter_objects,
    _objects_of_type,
    _fill_template,
)

Number = Union[int, float]
WorldState = Mapping[str, Any]
QuestionPayload = Mapping[str, Any]
Answer = Union[int, float, str]

MOVEMENT_TOLERANCE = 1e-3
DEFAULT_DISPLACEMENT_THRESHOLD = 2.0

## --- Resolver functions -- ##


@with_resolved_attributes
def F_FORCES_COUNTING_MOVEMENT_OBJECT_TYPE(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> int:
    """Count objects of a specific type that are currently moving."""

    assert len(resolved_attributes) == 1 and "OBJECT_TYPE" in resolved_attributes

    # core logic
    object_data = resolved_attributes["OBJECT_TYPE"]["choice"]
    object_id_has_moved = {
        obj["id"]: False for obj in _objects_of_type(world_state, object_data["type"])
    }
    for timestep in world_state["simulation_steps"]:
        for obj in _objects_of_type(world_state, object_data["type"]):
            if not object_id_has_moved[obj["id"]] and _is_moving(
                obj["id"], timestep, world_state
            ):
                object_id_has_moved[obj["id"]] = True

    total_moved_objects = sum(1 for moved in object_id_has_moved.values() if moved)
    _fill_template(question, resolved_attributes)

    options, correct_idx = create_mc_options_around_gt(
        total_moved_objects, num_answers=4, display_decimals=0, lo=0.0
    )
    labels = uniform_labels(options, integer=False, decimals=0)

    return question, labels, correct_idx


@with_resolved_attributes
def F_FORCES_COUNTING_MOVEMENT_OBJECT_TYPE_TIME(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> int:
    """Count objects of a specific type that are currently moving."""

    assert (
        len(resolved_attributes) == 2
        and "OBJECT_TYPE" in resolved_attributes
        and "TIME" in resolved_attributes
    )

    # core logic
    object_data = resolved_attributes["OBJECT_TYPE"]["choice"]
    timestep = resolved_attributes["TIME"]["choice"]
    count_moving = sum(
        1
        for obj in _objects_of_type(world_state, object_data["type"])
        if _is_moving(obj["id"], timestep, world_state)
    )
    _fill_template(question, resolved_attributes)

    options, correct_idx = create_mc_options_around_gt(
        count_moving, num_answers=4, display_decimals=0, lo=0.0
    )
    labels = uniform_labels(options, integer=True, decimals=0)

    return question, labels, correct_idx


# do we have the vector of the velocity? if not we cannot do this
def F_FORCES_COUNTING_MOVEMENT_UPWARD(
    world_state: WorldState, question: QuestionPayload
) -> int:
    """Count objects of a specific type that have a positive vertical velocity."""
    return {}, ["not_implemented"], 0  # Placeholder for actual implementation.


@with_resolved_attributes
def F_FORCES_COUNTING_MOVEMENT_METRIC(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> int:
    """Count objects of a specific type that moved more than a given metric distance."""
    assert (
        len(resolved_attributes) == 2
        and "OBJECT_TYPE" in resolved_attributes
        and "DISTANCE" in resolved_attributes
    )

    object_type = resolved_attributes["OBJECT_TYPE"]["choice"]["type"]
    distance = resolved_attributes["DISTANCE"]["choice"]

    count = 0
    for obj in _objects_of_type(world_state, object_type):
        displacement = _get_displacement(
            obj["id"], kwargs["timestep_start"], kwargs["timestep_end"], world_state
        )
        if displacement is None:
            raise ValueError(
                f"Cannot compute displacement for object {obj['id']} due to missing position data."
            )
        if displacement > distance:
            count += 1

    _fill_template(question, resolved_attributes)
    options, correct_idx = create_mc_options_around_gt(
        count, num_answers=4, display_decimals=0, lo=0.0
    )
    labels = uniform_labels(options, integer=True, decimals=0)

    return question, labels, correct_idx


@with_resolved_attributes
def F_FORCES_COUNTING_STILL(
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

    _fill_template(question, resolved_attributes)
    options, correct_idx = create_mc_options_around_gt(
        still_objects, num_answers=4, display_decimals=0, lo=0.0
    )
    labels = uniform_labels(options, integer=True, decimals=0)
    return question, labels, correct_idx


@with_resolved_attributes
def F_FORCES_COUNTING_STILL_OBJECT_TYPE(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> int:
    """Count objects of a specific type that are effectively still."""

    assert len(resolved_attributes) == 1 and "OBJECT_TYPE" in resolved_attributes
    object_type = resolved_attributes["OBJECT_TYPE"]["choice"]["type"]

    still_objects = 0
    for obj in _objects_of_type(world_state, object_type):
        obj_displacement = _get_displacement(
            obj["id"], kwargs["timestep_start"], kwargs["timestep_end"], world_state
        )
        if obj_displacement is None:
            raise ValueError(
                f"Cannot compute displacement for object {obj['id']} due to missing position data."
            )
        if obj_displacement < MOVEMENT_TOLERANCE:
            still_objects += 1

    _fill_template(question, resolved_attributes)
    options, correct_idx = create_mc_options_around_gt(
        still_objects, num_answers=4, display_decimals=0, lo=0.0
    )
    labels = uniform_labels(options, integer=True, decimals=0)
    return question, labels, correct_idx


@with_resolved_attributes
def F_FORCES_ESTIMATION_FASTEST_SPEED(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> float:
    """Return the speed of the fastest moving object."""
    assert len(resolved_attributes) == 0

    max_speed = 0.0
    for timestep in world_state["simulation_steps"]:
        speeds = [
            _get_speed(obj["id"], timestep, world_state)
            for obj in _iter_objects(world_state)
        ]
        max_speed_at_timestep = max(speeds, default=0.0)
        if max_speed_at_timestep > max_speed:
            max_speed = max_speed_at_timestep

    _fill_template(question, resolved_attributes)
    options, correct_idx = create_mc_options_around_gt(max_speed, num_answers=4, lo=0.0)
    options_with_units = [f"{opt} m/s" for opt in options]
    return question, options_with_units, correct_idx


@with_resolved_attributes
def F_FORCES_ESTIMATION_SPEED_OBJECT(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> float:
    """Return the speed of the object referenced in the question."""

    assert (
        len(resolved_attributes) == 2
        and "TIME" in resolved_attributes
        and "OBJECT" in resolved_attributes
    )
    timestep = resolved_attributes["TIME"]["choice"]
    object_id = resolved_attributes["OBJECT"]["choice"]["id"]

    velocity_object_at_timestep = _get_speed(object_id, timestep, world_state)

    _fill_template(question, resolved_attributes)

    # create the answer options
    options, correct_idx = create_mc_options_around_gt(
        velocity_object_at_timestep, num_answers=4
    )
    # add units to options
    options_with_units = [f"{opt} m/s" for opt in options]
    return question, options_with_units, correct_idx


@with_resolved_attributes
def F_FORCES_ESTIMATION_ACCELERATION_OBJECT(
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
    _fill_template(question, resolved_attributes)

    options, correct_idx = create_mc_options_around_gt(
        acceleration_object_at_timestep, num_answers=4, lo=0.0, display_decimals=2
    )
    labels = uniform_labels(options, integer=False, decimals=2)
    labels = [f"{opt} m/s^2" for opt in labels]
    return question, labels, correct_idx


@with_resolved_attributes
def F_FORCES_ESTIMATION_ANGULAR_VELOCITY_OBJECT(
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

    angular_velocity_object_at_timestep = get_angular_velocity(
        object_id, timestep, world_state
    )

    _fill_template(question, resolved_attributes)

    options, correct_idx = create_mc_options_around_gt(
        angular_velocity_object_at_timestep, num_answers=4
    )
    options_with_units = [f"{opt} deg/s" for opt in options]
    return question, options_with_units, correct_idx


def F_FORCES_GRAVITY_ESTIMATION(
    world_state: WorldState, question: QuestionPayload
) -> float:
    """Return the gravitational acceleration specified by the world state."""
    del question  # Unused for this resolver.
    gravity = world_state.get("gravity")
    return _coerce_to_float(gravity) or 0.0
