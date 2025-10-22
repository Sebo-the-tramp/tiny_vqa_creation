"""
Mock implementations for contact/collision related questions.

The routines below synthesise answers using heuristic parsing of the world state.
They are resilient to partial information and gracefully degrade when specific
collision metadata is unavailable.
"""

from __future__ import annotations

from typing import Any, Mapping

from utils.all_objects import get_all_objects_names, get_all_scenes_segments
from utils.decorators import with_resolved_attributes

from utils.frames_selection import uniformly_sample_frames

from utils.helpers import (
    _coerce_to_float,
    _iter_objects,
)

from utils.bin_creation import (
    create_mc_options_around_gt,
    create_mc_object_names_from_dataset,
    uniform_labels,
)

WorldState = Mapping[str, Any]
QuestionPayload = Mapping[str, Any]

from utils.config import get_config

SAMPLING_RATE = get_config()["sampling_rate"]
DELTA_TIME_BETWEEN_STEPS = 1.0 / SAMPLING_RATE
COLLISION_BUFFER_TIME = 0.1  # seconds
COLLISION_BUFFER_STEPS = -int(-COLLISION_BUFFER_TIME // DELTA_TIME_BETWEEN_STEPS)


@with_resolved_attributes
def F_COLLISIONS_OBJ_OBJ_FIRST(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> int:
    assert len(resolved_attributes) == 1 and "OBJECT" in resolved_attributes

    object = resolved_attributes["OBJECT"]["choice"]

    first_collided_object = None

    for _, value in world_state["simulation"].items():
        collisions_at_sim_step = value["collisions"]
        for collision in collisions_at_sim_step:
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

    DATASET = get_all_objects_names()
    _iter_objects_list = list(_iter_objects(world_state))
    present = [obj["name"] for obj in _iter_objects_list if obj["id"] != object["id"]]

    if first_collided_object is not None:
        labels, idx = create_mc_object_names_from_dataset(
            first_collided_object["name"], present, DATASET
        )
    else:
        labels, idx = create_mc_object_names_from_dataset("No Object", present, DATASET)

    return question, labels, idx, uniformly_sample_frames(world_state)


@with_resolved_attributes
def F_COLLISIONS_OBJ_OBJ_COUNT_GENERAL(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> int:
    assert len(resolved_attributes) == 1 and "OBJECT" in resolved_attributes

    object = resolved_attributes["OBJECT"]["choice"]

    count = 0
    is_colliding = False  # 0.1 second at 0.01 per step
    last_collision_sim_step = -1
    for value in world_state["simulation"].values():
        sim_step = value["simstep"]
        collisions_at_sim_step = value["collisions"]
        for collision in collisions_at_sim_step:
            obj_a = str(collision[0])
            obj_b = str(collision[1])
            if (
                not is_colliding
                and obj_a != 0
                and obj_b != 0
                and (obj_a == object["id"] or obj_b == object["id"])
            ):
                last_collision_sim_step = sim_step
                is_colliding = True
                count += 1

            # if object is not in the collision, reset the flag
            if (
                obj_a != object["id"]
                and obj_b != object["id"]
                and (
                    sim_step - last_collision_sim_step
                    >= COLLISION_BUFFER_STEPS  # since now steps are not necessarily 0.01s
                )
            ):
                is_colliding = False

    options, correct_idx = create_mc_options_around_gt(
        count, num_answers=4, display_decimals=0, lo=0.0
    )
    labels = uniform_labels(options, integer=True, decimals=0)

    return question, labels, correct_idx, uniformly_sample_frames(world_state)


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
    is_colliding = False
    last_collision_sim_step = -1
    for value in world_state["simulation"].values():
        sim_step = value["simstep"]
        collisions_at_sim_step = value["collisions"]
        for collision in collisions_at_sim_step:
            obj_a = str(collision[0])
            obj_b = str(collision[1])
            if not is_colliding and (
                (obj_a == object_1["id"] and obj_b == object_2["id"])
                or (obj_a == object_2["id"] and obj_b == object_1["id"])
            ):
                last_collision_sim_step = sim_step
                is_colliding = True
                count += 1

            # if object is not in the collision, reset the flag
            if (
                is_colliding
                and not (
                    (obj_a == object_1["id"] and obj_b == object_2["id"])
                    or (obj_a == object_2["id"] and obj_b == object_1["id"])
                )
                and (
                    sim_step - last_collision_sim_step
                    >= COLLISION_BUFFER_STEPS  # since now steps are not necessarily 0.01s
                )
            ):
                is_colliding = False

    options, correct_idx = create_mc_options_around_gt(
        count, num_answers=4, display_decimals=0, lo=0.0
    )
    labels = uniform_labels(options, integer=True, decimals=0)

    return question, labels, correct_idx, uniformly_sample_frames(world_state)


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
    for value in world_state["simulation"].values():
        sim_step = value["simstep"]
        collisions_at_sim_step = value["collisions"]
        for collision in collisions_at_sim_step:
            obj_a = str(collision[0])
            obj_b = str(collision[1])
            if obj_a == object["id"] or obj_b == object["id"]:
                first_collision_time = sim_step * DELTA_TIME_BETWEEN_STEPS
                break
        if first_collision_time is not None:
            break

    # create a random option between 0 and max time (wanted do do something)
    # cleverer but lesim_step's see what chatgpt comes up with
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
        return question, labels, correct_idx, uniformly_sample_frames(world_state)

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

        return question, labels, correct_idx, uniformly_sample_frames(world_state)


@with_resolved_attributes
def F_COLLISIONS_OBJ_OBJ_TIME_LAST(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> int:
    assert len(resolved_attributes) == 1 and "OBJECT" in resolved_attributes

    object = resolved_attributes["OBJECT"]["choice"]

    first_collision_time = None
    # first collision in a reversed list is last :)
    for sim_step, value in dict(reversed(world_state["simulation"].items())).items():
        collisions_at_sim_step = value["collisions"]
        for collision in collisions_at_sim_step:
            obj_a = str(collision[0])
            obj_b = str(collision[1])
            if obj_a == object["id"] or obj_b == object["id"]:
                first_collision_time = sim_step
                break
        if first_collision_time is not None:
            break

    # create a random option between 0 and max time (wanted do do something)
    # cleverer but lesim_step's see what chatgpt comes up with
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
        return question, labels, correct_idx, uniformly_sample_frames(world_state)

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

        return question, labels, correct_idx, uniformly_sample_frames(world_state)


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
    for sim_step, value in world_state["simulation"].items():
        for object_id_in_sim_step in value["objects"]:
            if object_id_in_sim_step != object_1["id"]:
                collisions = value["objects"][object_id_in_sim_step].get("collide", [])
                if collisions != []:
                    for collision in collisions:
                        obj_a = str(collision[0])
                        obj_b = str(collision[1])
                        if (obj_a == object_1["id"] and obj_b == object_2["id"]) or (
                            obj_a == object_2["id"] and obj_b == object_1["id"]
                        ):
                            first_collision_force = collision[object_1["id"]].get(
                                "force", None
                            )
                            break

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
        return question, labels, correct_idx, uniformly_sample_frames(world_state)

    else:
        first_collision_force = _coerce_to_float(first_collision_force)
        options, correct_idx = create_mc_options_around_gt(
            first_collision_force, num_answers=4, display_decimals=1, lo=0.0
        )
        labels = uniform_labels(options, integer=False, decimals=1)
        labels = [str(label) + " Newtons" for label in labels]

        return question, labels, correct_idx, uniformly_sample_frames(world_state)


@with_resolved_attributes
def F_COLLISIONS_OBJ_SCENE(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> int:
    object = resolved_attributes["OBJECT"]["choice"]

    count = 0
    is_colliding = False  # 0.1 second at 0.01 per step
    last_collision_sim_step = -1
    for value in world_state["simulation"].values():
        sim_step = value["simstep"]
        collisions_at_sim_step = value["collisions"]
        for collision in collisions_at_sim_step:
            obj_a = str(collision[0])
            obj_b = str(collision[1])
            if (
                not is_colliding
                and (obj_a == 0 or obj_b == 0)
                and (obj_a == object["id"] or obj_b == object["id"])
            ):
                last_collision_sim_step = sim_step
                is_colliding = True
                count += 1

            # if object is not in the collision, reset the flag
            if (obj_a != 0 and obj_b != 0) and (
                sim_step - last_collision_sim_step >= COLLISION_BUFFER_STEPS
            ):
                is_colliding = False

    options, correct_idx = create_mc_options_around_gt(
        count, num_answers=4, display_decimals=0, lo=0.0
    )
    labels = uniform_labels(options, integer=True, decimals=0)

    return question, labels, correct_idx, uniformly_sample_frames(world_state)


# TODO this needs to be done
@with_resolved_attributes
def F_COLLISIONS_OBJ_SCENE_SEGMENT(
    world_state: WorldState, question: QuestionPayload, resolved_attributes, **kwargs
) -> int:
    object = resolved_attributes["OBJECT"]["choice"]

    segment_name = ""
    for sim_step, value in world_state["simulation"].items():
        collisions_at_sim_step = value["collisions"]
        for collision in collisions_at_sim_step:
            obj_a = str(collision[0])
            obj_b = str(collision[1])
            if (obj_a == 0 or obj_b == 0) and (
                obj_a == object["id"] or obj_b == object["id"]
            ):
                # check which part of the scene it is colliding with
                # TODO

                segment_name = "Tree"  # dummy
                break

    correct_answer = segment_name if segment_name != "" else "No Collision"

    labels, correct_idx = create_mc_object_names_from_dataset(
        correct_answer, present, get_all_scenes_segments()
    )

    return question, labels, correct_idx, uniformly_sample_frames(world_state)
