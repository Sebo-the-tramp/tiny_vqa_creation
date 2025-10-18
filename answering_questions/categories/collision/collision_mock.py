"""
Mock implementations for contact/collision related questions.

The routines below synthesise answers using heuristic parsing of the world state.
They are resilient to partial information and gracefully degrade when specific
collision metadata is unavailable.
"""

from __future__ import annotations

from typing import Any, Mapping

from utils.all_objects import get_all_objects_names
from utils.decorators import with_resolved_attributes

from utils.frames_selection import uniformly_sample_frames

from utils.helpers import (
    _coerce_to_float,
    _iter_objects,
    _fill_template,
)

from utils.bin_creation import (
    create_mc_options_around_gt,
    create_mc_object_names_from_dataset,
    uniform_labels,
)

WorldState = Mapping[str, Any]
QuestionPayload = Mapping[str, Any]


@with_resolved_attributes
def F_COLLISIONS_OBJ_OBJ_FIRST(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> int:
    assert len(resolved_attributes) == 1 and "OBJECT" in resolved_attributes

    object = resolved_attributes["OBJECT"]["choice"]

    first_collided_object = None

    for _, value in world_state["simulation"].items():
        collisions_at_ts = value["collisions"]
        for collision in collisions_at_ts:
            obj_a = collision[0]
            obj_b = collision[1]
            if obj_a == 0 or obj_b == 0:
                continue  # we are just colliding with the ground
            if obj_a == object["id"] or obj_b == object["id"]:
                if obj_a == object["id"]:
                    first_collided_object = world_state[["objects"]][str(obj_b)]
                else:
                    first_collided_object = world_state[["objects"]][str(obj_b)]
                break

    _fill_template(question, resolved_attributes)

    DATASET = get_all_objects_names()
    _iter_objects_list = list(_iter_objects(world_state))
    present = [obj["name"] for obj in _iter_objects_list if obj["id"] != object["id"]]

    if first_collided_object is not None:
        labels, idx = create_mc_object_names_from_dataset(
            first_collided_object["name"], present, DATASET
        )
    else:
        labels, idx = create_mc_object_names_from_dataset("No Object", present, DATASET)

    # choosing frames to show collision
    imgs_idx = uniformly_sample_frames(world_state)

    return question, labels, idx, imgs_idx


@with_resolved_attributes
def F_COLLISIONS_OBJ_OBJ_COUNT_GENERAL(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> int:
    assert len(resolved_attributes) == 1 and "OBJECT" in resolved_attributes

    object = resolved_attributes["OBJECT"]["choice"]

    count = 0
    has_collided_in_previous_10_steps = False  # 0.1 second at 0.01 per step

    for ts, value in world_state["simulation"].items():
        collisions_at_ts = value["collisions"]
        for collision in collisions_at_ts:
            obj_a = str(collision[0])
            obj_b = str(collision[1])
            if (
                not has_collided_in_previous_10_steps
                and obj_a != "0"
                and obj_b != 0
                and (obj_a == object["id"] or obj_b == object["id"])
            ):
                count += 1
                has_collided_in_previous_10_steps = True
        # reset the flag after 10 steps
        if has_collided_in_previous_10_steps:
            step_float = float(ts)
            if step_float % 0.1 < 0.01:
                has_collided_in_previous_10_steps = False

    _fill_template(question, resolved_attributes)
    options, correct_idx = create_mc_options_around_gt(
        count, num_answers=4, display_decimals=0, lo=0.0
    )
    labels = uniform_labels(options, integer=True, decimals=0)

    # choosing frames to show collision
    imgs_idx = uniformly_sample_frames(world_state)

    return question, labels, correct_idx, imgs_idx


@with_resolved_attributes
def F_COLLISIONS_OBJ_OBJ_COUNT_TYPE(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> int:
    assert (
        len(resolved_attributes) == 2
        and "OBJECT_1" in resolved_attributes
        and "OBJECT_2" in resolved_attributes
    )

    object_1 = resolved_attributes["OBJECT_1"]["choice"]
    object_2 = resolved_attributes["OBJECT_2"]["choice"]

    count = 0
    has_collided_in_previous_10_steps = False  # 0.1 second at 0.01 per step
    for ts, value in world_state["simulation"].items():
        collisions_at_ts = value["collisions"]
        for collision in collisions_at_ts:
            obj_a = str(collision[0])
            obj_b = str(collision[1])
            if not has_collided_in_previous_10_steps and (
                (obj_a == object_1["id"] and obj_b == object_2["id"])
                or (obj_a == object_2["id"] and obj_b == object_1["id"])
            ):
                count += 1
                has_collided_in_previous_10_steps = True
        # reset the flag after 10 steps
        if has_collided_in_previous_10_steps:
            step_float = float(ts)
            if step_float % 0.1 < 0.01:
                has_collided_in_previous_10_steps = False

    _fill_template(question, resolved_attributes)
    options, correct_idx = create_mc_options_around_gt(
        count, num_answers=4, display_decimals=0, lo=0.0
    )
    labels = uniform_labels(options, integer=True, decimals=0)

    # choosing frames to show collision
    imgs_idx = uniformly_sample_frames(world_state)

    return question, labels, correct_idx, imgs_idx


@with_resolved_attributes
def F_COLLISIONS_OBJ_OBJ_TIME_FIRST(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> int:
    assert len(resolved_attributes) == 1 and "OBJECT" in resolved_attributes

    object = resolved_attributes["OBJECT"]["choice"]

    first_collision_time = None

    # supposing 8 frames we give to the LLM to answer the question
    # I mean we can't expect perfect time, but it should interpolate anyway, knowing the number of frames
    # and the FPS of the simulation
    for ts, value in world_state["simulation"].items():
        collisions_at_ts = value["collisions"]
        for collision in collisions_at_ts:
            obj_a = str(collision[0])
            obj_b = str(collision[1])
            if obj_a == object["id"] or obj_b == object["id"]:
                first_collision_time = ts
                break
        if first_collision_time is not None:
            break

    _fill_template(question, resolved_attributes)

    # choosing frames to show collision
    imgs_idx = uniformly_sample_frames(world_state)

    # create a random option between 0 and max time (wanted do do something)
    # cleverer but lets's see what chatgpt comes up with
    if first_collision_time is None:
        max_time = 0.01 * len(world_state["simulation"]) - 0.01
        options, correct_idx = create_mc_options_around_gt(
            max_time / 2,
            num_answers=4,
            display_decimals=2,
            lo=0.0,
            hi=len(world_state["simulation"]) * 0.01 - 0.01,
        )
        labels = uniform_labels(options, integer=False, decimals=2)
        labels = [str(label) + " seconds" for label in labels]

        labels[correct_idx] = "No Collision"
        return question, labels, correct_idx

    else:
        first_collision_time = float(first_collision_time)
        options, correct_idx = create_mc_options_around_gt(
            first_collision_time,
            num_answers=4,
            display_decimals=2,
            lo=0.0,
            min_rel_gap=(len(world_state["simulation"]) // 7) * 0.01,
            hi=len(world_state["simulation"]) * 0.01 - 0.01,
        )
        labels = uniform_labels(options, integer=False, decimals=2)
        labels = [str(label) + " seconds" for label in labels]

        return question, labels, correct_idx, imgs_idx


@with_resolved_attributes
def F_COLLISIONS_OBJ_OBJ_TIME_LAST(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> int:
    assert len(resolved_attributes) == 1 and "OBJECT" in resolved_attributes

    object = resolved_attributes["OBJECT"]["choice"]

    first_collision_time = None
    # first collision in a reversed list is last :)
    for ts, value in dict(reversed(world_state["simulation"].items())).items():
        collisions_at_ts = value["collisions"]
        for collision in collisions_at_ts:
            obj_a = str(collision[0])
            obj_b = str(collision[1])
            if obj_a == object["id"] or obj_b == object["id"]:
                first_collision_time = ts
                break
        if first_collision_time is not None:
            break

    _fill_template(question, resolved_attributes)

    # choosing frames to show collision
    imgs_idx = uniformly_sample_frames(world_state)

    # create a random option between 0 and max time (wanted do do something)
    # cleverer but lets's see what chatgpt comes up with
    if first_collision_time is None:
        max_time = 0.01 * len(world_state["simulation"]) - 0.01
        options, correct_idx = create_mc_options_around_gt(
            max_time / 2,
            num_answers=4,
            display_decimals=2,
            lo=0.0,
            hi=len(world_state["simulation"]) * 0.01 - 0.01,
        )
        labels = uniform_labels(options, integer=False, decimals=2)
        labels = [str(label) + " seconds" for label in labels]

        labels[correct_idx] = "No Collision"
        return question, labels, correct_idx, imgs_idx

    else:
        first_collision_time = float(first_collision_time)
        options, correct_idx = create_mc_options_around_gt(
            first_collision_time,
            num_answers=4,
            display_decimals=2,
            lo=0.0,
            hi=len(world_state["simulation"]) * 0.01 - 0.01,
        )
        labels = uniform_labels(options, integer=False, decimals=2)
        labels = [str(label) + " seconds" for label in labels]

        return question, labels, correct_idx, imgs_idx


@with_resolved_attributes
def F_COLLISIONS_OBJ_OBJ_FORCE(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> int:
    assert (
        len(resolved_attributes) == 2
        and "OBJECT_1" in resolved_attributes
        and "OBJECT_2" in resolved_attributes
    )

    object_1 = resolved_attributes["OBJECT_1"]["choice"]
    object_2 = resolved_attributes["OBJECT_2"]["choice"]

    first_collision_force = None
    for ts, value in world_state["simulation"].items():
        for object_id_in_ts in value["objects"]:
            if object_id_in_ts != object_1["id"]:
                collisions = value["objects"][object_id_in_ts].get("collide", [])
                if collisions != []:
                    for collision in collisions:
                        if object_1["id"] in list(collision.keys()):
                            first_collision_force = collision[object_1["id"]].get(
                                "force", None
                            )
                            break

    _fill_template(question, resolved_attributes)

    # choosing frames to show collision
    imgs_idx = uniformly_sample_frames(world_state)

    # create a random option between 0 and max force (wanted do do something)
    # cleverer but lets's see what chatgpt comes up with
    if first_collision_force is None:
        max_force = 10.0
        options, correct_idx = create_mc_options_around_gt(
            max_force / 2, num_answers=4, display_decimals=1, lo=0.0
        )
        labels = uniform_labels(options, integer=False, decimals=1)
        labels = [str(label) + " Newtons" for label in labels]

        labels[correct_idx] = "No Collision"
        return question, labels, correct_idx, imgs_idx

    else:
        first_collision_force = _coerce_to_float(first_collision_force)
        options, correct_idx = create_mc_options_around_gt(
            first_collision_force, num_answers=4, display_decimals=1, lo=0.0
        )
        labels = uniform_labels(options, integer=False, decimals=1)
        labels = [str(label) + " Newtons" for label in labels]

        return question, labels, correct_idx, imgs_idx
