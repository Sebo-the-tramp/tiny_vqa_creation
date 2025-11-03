"""
Mock spatial reasoning resolvers.

These helpers extract best-effort spatial answers from the provided world state.
They operate on lightweight metadata (positions, orientations, region tags, etc.)
and fall back to sensible defaults when information is missing.
"""

from __future__ import annotations

from utils.decorators import with_resolved_attributes

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

from utils.all_objects import get_all_objects_names

from utils.helpers import (
    iter_objects,
    distance_between,
    resolve_attributes_visible_at_timestep,
    get_visible_timesteps_for_attributes_min_objects,
    get_continuous_subsequences_min_length,
    is_object_visible_at_timestep,
    get_random_timestep_from_list,
    fill_template,
)

from utils.frames_selection import ( 
    sample_frames_before_timestep,
)

from .mechanics_helpers import get_speed, fill_questions, get_acceleration, get_position, get_rotation

from utils.config import get_config

from utils.bin_creation import create_mc_options_around_gt, create_mc_object_names_from_dataset, uniform_labels

Number = Union[int, float]
Vector = Tuple[float, float, float]
WorldState = Mapping[str, Any]
QuestionPayload = Mapping[str, Any]
Answer = Union[str, float, Vector, Mapping[str, Any], Sequence[str]]

CLIP_LENGTH = get_config()["clip_length"]
FRAME_INTERLEAVE = get_config()["frame_interleave"]

MOVEMENT_TOLERANCE = get_config()["movement_tolerance"]
ROTATION_TOLERANCE = get_config()["rotation_tolerance"]

## --- Resolver functions -- ##


@with_resolved_attributes
def F_KINEMATICS_SPEED_OBJECT(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    """Return the velocity of the object referenced in the question."""

    assert len(attributes) == 1 and "OBJECT" in attributes

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        attributes, world_state, min_objects=kwargs['current_world_number_of_objects']
    )
    
    timestep = get_random_timestep_from_list(
        visible_timesteps, question
    )

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, timestep
    )

    object_id = resolved_attributes["OBJECT"]["choice"]["id"]

    velocity_object_at_timestep = get_speed(object_id, timestep, world_state)

    labels, correct_idx = create_mc_options_around_gt(
        velocity_object_at_timestep, num_answers=4
    )
    labels = [f"{label} m/s" for label in labels]

    return fill_questions(
        question, labels, correct_idx, world_state, timestep, resolved_attributes
    )

# @with_resolved_attributes
# def F_KINEMATICS_FASTEST_OBJECT(
#     world_state: WorldState, question: QuestionPayload, attributes, **kwargs
# ) -> int:
#     """Return the velocity of the object referenced in the question."""

#     assert len(attributes) == 0

#     # First we find the pairs of objects visible
#     visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
#         ["OBJECT"], world_state, min_objects=kwargs['current_world_number_of_objects']
#     )
#     # if we are in a multi-image setting, we need to ensure there are enough frames
#     timestep = get_random_timestep_from_list(
#         visible_timesteps, question
#     )

#     resolved_attributes = resolve_attributes_visible_at_timestep(
#         attributes, world_state, timestep
#     )

#     highest_speed = -1.0
#     fastest_object = None
#     for object in iter_objects(world_state):
#         object_id = object["id"]
#         speed = get_speed(object_id, timestep, world_state)
#         if speed > highest_speed:
#             highest_speed = speed
#             fastest_object = object
    
#     present = [obj["name"] for obj in list(iter_objects(world_state)) if obj["id"] != object["id"]]

#     labels, correct_idx = create_mc_object_names_from_dataset(
#             fastest_object["name"], present, get_all_objects_names()
#         )

#     return fill_questions(
#         question, labels, correct_idx, world_state, timestep, resolved_attributes
#     )

# @with_resolved_attributes
# def F_KINEMATICS_FASTEST_OBJECT_SPEED(
#     world_state: WorldState, question: QuestionPayload, attributes, **kwargs
# ) -> int:
#     """Return the velocity of the object referenced in the question."""

#     assert len(attributes) == 0

#     # First we find the pairs of objects visible
#     visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
#         ["OBJECT"], world_state, min_objects=kwargs['current_world_number_of_objects']
#     )
#     # if we are in a multi-image setting, we need to ensure there are enough frames
#     timestep = get_random_timestep_from_list(
#         visible_timesteps, question
#     )

#     resolved_attributes = resolve_attributes_visible_at_timestep(
#         attributes, world_state, timestep
#     )

#     highest_speed = -1.0
#     for object in iter_objects(world_state):
#         object_id = object["id"]
#         speed = get_speed(object_id, timestep, world_state)
#         if speed > highest_speed:
#             highest_speed = speed

#     labels, correct_idx = create_mc_options_around_gt(highest_speed, num_answers=4)
#     labels = [f"{label} m/s" for label in labels]

#     return fill_questions(
#         question, labels, correct_idx, world_state, timestep, resolved_attributes
#     )


@with_resolved_attributes
def F_KINEMATICS_ACCEL_OBJECT(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    """Return the velocity of the object referenced in the question."""

    assert len(attributes) == 1 and "OBJECT" in attributes

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        attributes, world_state, min_objects=kwargs['current_world_number_of_objects']
    )
    # if we are in a multi-image setting, we need to ensure there are enough frames
    timestep = get_random_timestep_from_list(
        visible_timesteps, question
    )

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, timestep
    )

    object_id = resolved_attributes["OBJECT"]["choice"]["id"]

    acceleration_object = get_acceleration(object_id, timestep, world_state)

    labels, correct_idx = create_mc_options_around_gt(
        acceleration_object, num_answers=4, display_decimals=2, lo=-100.0, hi=100.0
    )
    labels = [f"{label} m/s^2" for label in labels]

    return fill_questions(
        question, labels, correct_idx, world_state, timestep, resolved_attributes
    )


@with_resolved_attributes
def F_KINEMATICS_DISTANCE_TRAVELED_INTERVAL(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    """Count objects of a specific type that moved more than a given metric distance."""
    assert len(attributes) == 1 and "OBJECT" in attributes

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        attributes, world_state, min_objects=kwargs['current_world_number_of_objects']
    )

    continuous_subsequences = get_continuous_subsequences_min_length(
        visible_timesteps, min_length=CLIP_LENGTH*FRAME_INTERLEAVE
    )

    visible_timesteps = random.choice(continuous_subsequences)

    timestep_end = get_random_timestep_from_list(
        visible_timesteps, question
    )
    timestep_start = visible_timesteps[
        visible_timesteps.index(timestep_end) - ((CLIP_LENGTH * FRAME_INTERLEAVE) - FRAME_INTERLEAVE)
    ]

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, timestep_start
    )

    object_id = resolved_attributes["OBJECT"]["choice"]["id"]

    position_obj_state_timestep_start = get_position(
        world_state, object_id, timestep_start
    )
    position_obj_state_timestep_end = get_position(world_state, object_id, timestep_end)
    distance = distance_between(
        position_obj_state_timestep_start, position_obj_state_timestep_end
    )

    options, correct_idx = create_mc_options_around_gt(
        distance, num_answers=4, display_decimals=1, lo=0.0
    )
    labels = uniform_labels(options, integer=False, decimals=1)
    labels = [f"{opt} meters" for opt in labels]

    return fill_questions(
        question, labels, correct_idx, world_state, timestep_end, resolved_attributes
    )


@with_resolved_attributes
def F_KINEMATICS_MOVING_OBJECT(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    """Return the velocity of the object referenced in the question."""

    assert len(attributes) == 1 and "OBJECT" in attributes

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        attributes, world_state, min_objects=kwargs['current_world_number_of_objects']
    )

    continuous_subsequences = get_continuous_subsequences_min_length(
        visible_timesteps, min_length=CLIP_LENGTH*FRAME_INTERLEAVE
    )

    visible_timesteps = random.choice(continuous_subsequences)

    timestep = get_random_timestep_from_list(
        visible_timesteps, question
    )

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, timestep
    )

    object_id = resolved_attributes["OBJECT"]["choice"]["id"]

    index_timestep = visible_timesteps.index(timestep)
    list_of_position = []
    list_of_rotation = []
    for i in range((CLIP_LENGTH - 1), -1, -1):        
        current_timestep = visible_timesteps[index_timestep - i]        
        position = get_position(world_state, object_id, current_timestep)
        rotation = get_rotation(world_state, object_id, current_timestep)
        list_of_position.append(position)
        list_of_rotation.append(rotation)

    is_moving = False
    for i in range(1, len(list_of_position)):
        dist = distance_between(list_of_position[i - 1], list_of_position[i])
        if dist > MOVEMENT_TOLERANCE:
            is_moving = True
            break
        rot_diff = sum(
            abs(list_of_rotation[i - 1][j] - list_of_rotation[i][j]) for j in range(3)
        )
        if rot_diff > ROTATION_TOLERANCE:
            is_moving = True
            break

    options = ["yes", "no"]
    correct_idx = 0 if is_moving else 1
    labels = options

    return fill_questions(
        question, labels, correct_idx, world_state, timestep, resolved_attributes
    )


@with_resolved_attributes
def F_KINEMATICS_STILL_OBJECT(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    """Return the velocity of the object referenced in the question."""

    assert len(attributes) == 1 and "OBJECT" in attributes

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        attributes, world_state, min_objects=kwargs['current_world_number_of_objects']
    )

    continuous_subsequences = get_continuous_subsequences_min_length(
        visible_timesteps, min_length=CLIP_LENGTH*FRAME_INTERLEAVE
    )

    visible_timesteps = random.choice(continuous_subsequences)    

    timestep = get_random_timestep_from_list(
        visible_timesteps, question
    )

    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, timestep
    )

    object_id = resolved_attributes["OBJECT"]["choice"]["id"]

    index_timestep = visible_timesteps.index(timestep)
    list_of_position = []
    for i in range((CLIP_LENGTH - 1), -1, -1):
        current_timestep = visible_timesteps[index_timestep - i]
        speed = get_position(world_state, object_id, current_timestep)
        list_of_position.append(speed)

    is_still = True
    for i in range(1, len(list_of_position)):
        dist = distance_between(list_of_position[i - 1], list_of_position[i])
        if dist > MOVEMENT_TOLERANCE:
            is_still = False
            break

    options = ["yes", "no"]
    correct_idx = 0 if is_still else 1
    labels = options

    return fill_questions(
        question, labels, correct_idx, world_state, timestep, resolved_attributes
    )


## --- COLLISION RESOLVERS --- ##

@with_resolved_attributes
def F_COLLISIONS_OBJ_OBJ_FIRST(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    assert len(attributes) == 1 and "OBJECT" in attributes

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        attributes, world_state, min_objects=kwargs['current_world_number_of_objects']
    )

    continuous_subsequences = get_continuous_subsequences_min_length(
        visible_timesteps, min_length=CLIP_LENGTH
    )

    visible_timesteps = random.choice(continuous_subsequences)[(CLIP_LENGTH - 1) :]
    
    resolved_attributes = resolve_attributes_visible_at_timestep(
        attributes, world_state, visible_timesteps[0]
    )

    object = resolved_attributes["OBJECT"]["choice"]

    first_collided_object = None
    for timestep in visible_timesteps:
        value = world_state["simulation"][str(timestep)]
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
    present = [obj["name"] for obj in list(iter_objects(world_state)) if obj["id"] != object["id"]]

    if first_collided_object is not None:
        labels, idx = create_mc_object_names_from_dataset(
            first_collided_object["name"], present, DATASET
        )
    else:
        labels, idx = create_mc_object_names_from_dataset("No Object", present, DATASET)

    return fill_questions(
        question, labels, idx, world_state, visible_timesteps[0], resolved_attributes
    )
 

@with_resolved_attributes
def F_COLLISION_OBJECT_OBJECT_FRAME_SINGLE(
        world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    assert len(attributes) == 1 and "OBJECT" in attributes

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        attributes, world_state, min_objects=kwargs['current_world_number_of_objects']
    )

    choice_collision = random.choice([0, 1]) # 0 for no, 1 for yes
    # choice_collision = 0  # forcing to NOT look for a collision

    collision_timestep = None
    non_collision_timestep = []

    # I just want to catch a collision here    
    for timestep in visible_timesteps:
        step_state = world_state["simulation"][str(timestep)]
        collisions_at_sim_step = step_state["collisions"]

        collisions_at_sim_step_ground = [
            collision for collision in collisions_at_sim_step if (collision[0] == 0 or collision[1] == 0)                
        ]

        # if we are looking for no collision, and there is none, we are done
        # this though is sampling the first timestep with no collision, not a random one
        if choice_collision == 0 and len(collisions_at_sim_step) == 0:
            non_collision_timestep.append(timestep)
            continue

        collisions_at_sim_visible_object = []
        for collision in collisions_at_sim_step_ground:
            obj_a = collision[0]
            obj_b = collision[1]
            if obj_a != 0 and obj_b != 0:
                if is_object_visible_at_timestep(str(obj_b), str(timestep), world_state) and \
                   is_object_visible_at_timestep(str(obj_a), str(timestep), world_state):
                    collisions_at_sim_visible_object.append(collision)

        if len(collisions_at_sim_visible_object) > 0: 
            collision_timestep = timestep
            collision_objects = collisions_at_sim_visible_object
            break

    if choice_collision == 1:
        if collision_timestep is None:
            raise ImpossibleToAnswer("No collision found in the visible timesteps.")        

        collision_between_obj_a_b = random.choice(collision_objects)
        collision_object_id = (
            collision_between_obj_a_b[0]
            if collision_between_obj_a_b[0] != 0
            else collision_between_obj_a_b[1]
        )   

        """ How the object looks like:
        {"OBJECT": {'choice': {'model': 'Olive_Kids_Game_On_Pack_n_Snack', 'sim': 'rho-medium_yms-medium_prs-medium', 'props': {...}, 'volume': 0.02960631065070629, 'mass': 1.6283470392227173, 'description': {...}, 'spawning_region': 'above_ground', 
        'initial_condition': {...}, 'scale': 1.2468836307525635, 'obb_size': None, 'id': '2', 'name': 'Olive_Kids_Game_On_Pack_n_Snack'}, 'category': 'OBJECT'}
        """
        #technically the resolved object should be the one colliding
        resolved_attributes = {"OBJECT": {"choice": world_state["objects"][str(collision_object_id)], "category": "OBJECT"}}    

    else:
        if len(non_collision_timestep) == 0:
            raise ImpossibleToAnswer("No non-collision found in the visible timesteps.")        
        collision_timestep = random.choice(non_collision_timestep)

        resolved_attributes = resolve_attributes_visible_at_timestep(
            attributes, world_state, collision_timestep
        )
    
    options = ["yes", "no"]
    correct_idx = abs(choice_collision - 1)
    labels = options

    return fill_questions(
        question, labels, correct_idx, world_state, collision_timestep, resolved_attributes
    )


@with_resolved_attributes
def F_COLLISION_OBJECT_OBJECT_FRAME_MULTI(
        world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    
    # print("ATTRIBUTES", attributes)

    assert len(attributes) == 1 and "OBJECT" in attributes

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        attributes, world_state, min_objects=kwargs['current_world_number_of_objects']
    )

    choice_collision = random.choice([0, 1]) # 0 for no, 1 for yes
    # choice_collision = 0  # forcing to look for a collision

    collision_timestep = None
    non_collision_timestep = []

    # I just want to catch a collision here    
    for timestep in visible_timesteps:
        step_state = world_state["simulation"][str(timestep)]
        collisions_at_sim_step = step_state["collisions"]

        collisions_at_sim_step_ground = [
            collision for collision in collisions_at_sim_step if (collision[0] == 0 or collision[1] == 0)                
        ]

        # if we are looking for no collision, and there is none, we are done
        # this though is sampling the first timestep with no collision, not a random one
        if choice_collision == 0 and len(collisions_at_sim_step) == 0:
            non_collision_timestep.append(timestep)
            continue

        collisions_at_sim_visible_object = []
        for collision in collisions_at_sim_step_ground:
            obj_a = collision[0]
            obj_b = collision[1]
            if obj_a != 0 and obj_b != 0:
                if is_object_visible_at_timestep(str(obj_b), str(timestep), world_state) and \
                   is_object_visible_at_timestep(str(obj_a), str(timestep), world_state):
                    collisions_at_sim_visible_object.append(collision)

        if len(collisions_at_sim_visible_object) > 0: 
            collision_timestep = timestep
            collision_objects = collisions_at_sim_visible_object
            break

    if choice_collision == 1:
        if collision_timestep is None:
            raise ImpossibleToAnswer("No collision found in the visible timesteps.")        

        collision_between_obj_a_b = random.choice(collision_objects)
        collision_object_id = (
            collision_between_obj_a_b[0]
            if collision_between_obj_a_b[0] != 0
            else collision_between_obj_a_b[1]
        )   

        """ How the object looks like:
        {"OBJECT": {'choice': {'model': 'Olive_Kids_Game_On_Pack_n_Snack', 'sim': 'rho-medium_yms-medium_prs-medium', 'props': {...}, 'volume': 0.02960631065070629, 'mass': 1.6283470392227173, 'description': {...}, 'spawning_region': 'above_ground', 
        'initial_condition': {...}, 'scale': 1.2468836307525635, 'obb_size': None, 'id': '2', 'name': 'Olive_Kids_Game_On_Pack_n_Snack'}, 'category': 'OBJECT'}
        """
        #technically the resolved object should be the one colliding
        resolved_attributes = {"OBJECT": {"choice": world_state["objects"][str(collision_object_id)], "category": "OBJECT"}}    

    else:
        if len(non_collision_timestep) == 0:
            raise ImpossibleToAnswer("No non-collision found in the visible timesteps.")        
        collision_timestep = random.choice(non_collision_timestep)

        resolved_attributes = resolve_attributes_visible_at_timestep(
            attributes, world_state, collision_timestep
        )

    frames_og = sample_frames_before_timestep(
        world_state, collision_timestep, num_frames=4, frame_interleave=2
    )   

    # Generate one shared permutation
    indices = list(range(len(frames_og)))
    random.shuffle(indices)

    labels = frames_og.copy()

    # Apply same shuffle to both
    frames = [frames_og[i] for i in indices]
    labels = [labels[i] for i in indices]

    correct_idx = labels.index(frames[3])

    fill_template(question, resolved_attributes)

    # only for this time because of the multi-frame choice nature of the question
    question["question"] = question["question"].replace("Consider all frames, but answer only based on the last frame. ", "")

    return [[
        question,
        labels,
        correct_idx,
        frames
    ]]


@with_resolved_attributes
def F_COLLISION_OBJECT_SCENE_FRAME_SINGLE(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:
    assert len(attributes) == 1 and "OBJECT" in attributes

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        attributes, world_state, min_objects=kwargs['current_world_number_of_objects']
    )

    choice_collision = random.choice([0, 1]) # 0 for no, 1 for yes
    # choice_collision = 0  # forcing to not look for a collision

    collision_timestep = None
    non_collision_timestep = []

    # I just want to catch a collision here    
    for timestep in visible_timesteps:
        step_state = world_state["simulation"][str(timestep)]
        collisions_at_sim_step = step_state["collisions"]

        collisions_at_sim_step_ground = [
            collision for collision in collisions_at_sim_step if (collision[0] == 0 or collision[1] == 0)                
        ]

        # if we are looking for no collision, and there is none, we are done
        # this though is sampling the first timestep with no collision, not a random one
        if choice_collision == 0 and len(collisions_at_sim_step) == 0:
            non_collision_timestep.append(timestep)
            continue

        collsions_at_sim_visible_object = []
        for collision in collisions_at_sim_step_ground:
            obj_a = collision[0]
            obj_b = collision[1]
            if obj_a == 0:                    
                if is_object_visible_at_timestep(str(obj_b), str(timestep), world_state):
                    collsions_at_sim_visible_object.append(collision)
            else:
                if is_object_visible_at_timestep(str(obj_a), str(timestep), world_state):
                    collsions_at_sim_visible_object.append(collision)

        if len(collsions_at_sim_visible_object) > 0: 
            collision_timestep = timestep
            collision_objects = collsions_at_sim_visible_object
            break

    if choice_collision == 1:
        if collision_timestep is None:
            raise ImpossibleToAnswer("No collision found in the visible timesteps.")        

        collision_between_obj_a_b = random.choice(collision_objects)
        collision_object_id = (
            collision_between_obj_a_b[0]
            if collision_between_obj_a_b[0] != 0
            else collision_between_obj_a_b[1]
        )   

        """ How the object looks like:
        {"OBJECT": {'choice': {'model': 'Olive_Kids_Game_On_Pack_n_Snack', 'sim': 'rho-medium_yms-medium_prs-medium', 'props': {...}, 'volume': 0.02960631065070629, 'mass': 1.6283470392227173, 'description': {...}, 'spawning_region': 'above_ground', 
        'initial_condition': {...}, 'scale': 1.2468836307525635, 'obb_size': None, 'id': '2', 'name': 'Olive_Kids_Game_On_Pack_n_Snack'}, 'category': 'OBJECT'}
        """
        #technically the resolved object should be the one colliding
        resolved_attributes = {"OBJECT": {"choice": world_state["objects"][str(collision_object_id)], "category": "OBJECT"}}    

    else:
        if len(non_collision_timestep) == 0:
            raise ImpossibleToAnswer("No non-collision found in the visible timesteps.")        
        collision_timestep = random.choice(non_collision_timestep)

        resolved_attributes = resolve_attributes_visible_at_timestep(
            attributes, world_state, collision_timestep
        )

    options = ["yes", "no"]
    correct_idx = abs(choice_collision - 1)
    labels = options

    return fill_questions(
        question, labels, correct_idx, world_state, collision_timestep, resolved_attributes
    )

# assumption that the object is not colliding at the start, falling and the colliding with the scene
@with_resolved_attributes
def F_COLLISION_OBJECT_SCENE_FRAME_MULTI(
    world_state: WorldState, question: QuestionPayload, attributes, **kwargs
) -> int:    
    assert len(attributes) == 1 and "OBJECT" in attributes

    # First we find the pairs of objects visible
    visible_timesteps = get_visible_timesteps_for_attributes_min_objects(
        attributes, world_state, min_objects=kwargs['current_world_number_of_objects']
    )

    collision_timestep = None
        
    for timestep in visible_timesteps:
        step_state = world_state["simulation"][str(timestep)]
        collisions_at_sim_step = step_state["collisions"]
        collisions_at_sim_step_ground = [
            collision for collision in collisions_at_sim_step if (collision[0] == 0 or collision[1] == 0)                
        ]
        collsions_at_sim_visible_object = []
        for collision in collisions_at_sim_step_ground:
            obj_a = collision[0]
            obj_b = collision[1]
            if obj_a == 0:                    
                if is_object_visible_at_timestep(str(obj_b), str(timestep), world_state):
                    collsions_at_sim_visible_object.append(collision)
            else:
                if is_object_visible_at_timestep(str(obj_a), str(timestep), world_state):
                    collsions_at_sim_visible_object.append(collision)

        if len(collsions_at_sim_visible_object) > 0: 
            collision_timestep = timestep
            collision_objects = collsions_at_sim_visible_object
            break

    if  collision_timestep is None:
        raise ImpossibleToAnswer("No collision found in the visible timesteps.")

    collision_between_obj_a_b = random.choice(collision_objects)
    collision_object_id = (
        collision_between_obj_a_b[0]
        if collision_between_obj_a_b[0] != 0
        else collision_between_obj_a_b[1]
    )   

    frames_og = sample_frames_before_timestep(
        world_state, timestep, num_frames=4, frame_interleave=2
    )

    #technically the resolved object should be the one colliding
    resolved_attributes = {"OBJECT": {"choice": world_state["objects"][str(collision_object_id)], "category": "OBJECT"}}
    
    # Create the labels copy
    labels = frames_og.copy()

    # Generate one shared permutation
    indices = list(range(len(frames_og)))
    random.shuffle(indices)

    # Apply same shuffle to both    
    labels = [labels[i] for i in indices]

    correct_idx = labels.index(frames_og[3])

    fill_template(question, resolved_attributes)

    # only for this time because of the multi-frame choice nature of the question
    question["question"] = question["question"].replace("Consider all frames, but answer only based on the last frame. ", "")

    return [[
        question,
        labels,
        correct_idx,
        [],
    ]]