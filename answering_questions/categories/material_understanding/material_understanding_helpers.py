from __future__ import annotations

import json

from typing import Any, Mapping, Union

# from utils.config import get_config

Number = Union[int, float]
WorldState = Mapping[str, Any]
QuestionPayload = Mapping[str, Any]
Answer = Union[int, float, str]


# MOVEMENT_TOLERANCE = 1e-3
# FRAME_INTERLEAVE = get_config()["frame_interleave"]


## --- Helper functions --- ##


# def fill_questions(
#     question, labels, correct_idx, world_state, timestep, resolved_attributes
# ) -> List:
#     questions = []
#     if "single" in question["task_splits"]:
#         question_copy = question.copy()
#         question_copy["task_splits"] = "single"  # ensure the question knows it's
#         fill_template(question_copy, resolved_attributes)
#         questions.append(
#             [
#                 question_copy,
#                 labels,
#                 correct_idx,
#                 sample_frames_at_timesteps(world_state, [timestep]),
#                 world_state,
#                 resolved_attributes,
#             ]
#         )
#     if "multi" in question["task_splits"]:
#         question_copy = question.copy()
#         question_copy["task_splits"] = "multi"  # ensure the question knows it's
#         fill_template(question_copy, resolved_attributes)
#         questions.append(
#             [
#                 question_copy,
#                 labels,
#                 correct_idx,
#                 sample_frames_before_timestep(
#                     world_state,
#                     timestep,
#                     num_frames=8,
#                     frame_interleave=FRAME_INTERLEAVE,
#                 ),
#                 world_state,
#                 resolved_attributes,
#             ]
#         )

#     return questions


# def get_position(
#     world_state: Mapping[str, Any], object_id: str, timestep: str
# ) -> Optional[Tuple[float, ...]]:
#     timestep_world = world_state["simulation"][timestep]
#     current_timestep_involved_object = timestep_world["objects"][object_id]["obb"][
#         "center"
#     ]
#     return as_vector(current_timestep_involved_object)


# def is_moving(object_id: str, timestep: str, world_state: Mapping[str, Any]) -> bool:
#     return get_speed(object_id, timestep, world_state) > MOVEMENT_TOLERANCE


# def get_speed(object_id: str, timestep: str, world_state: Mapping[str, Any]) -> float:
#     timestep_world = world_state["simulation"][timestep]
#     current_timestep_involved_object = timestep_world["objects"][object_id][
#         "kinematics"
#     ]["speed"]
#     return current_timestep_involved_object


# def get_acceleration(
#     object_id: str, timestep: str, world_state: Mapping[str, Any]
# ) -> float:
#     timestep_world = world_state["simulation"][timestep]
#     current_timestep_involved_object_velocity = timestep_world["objects"][object_id][
#         "kinematics"
#     ]["linear_accel_world"]

#     acceleration_magnitude = (
#         current_timestep_involved_object_velocity[0] ** 2
#         + current_timestep_involved_object_velocity[1] ** 2
#         + current_timestep_involved_object_velocity[2] ** 2
#     ) ** 0.5

#     return acceleration_magnitude

material_taxonomy = {}
with open("./categories/material_understanding/material_taxonomy.json", "r") as f:
    material_taxonomy = json.load(f)["material_taxonomy"]


def get_material_dataset_different_from_target(
    target_material: str, target_level: int = 2
) -> str:
    confounders_level_1 = set()
    confounders_level_2 = set()
    confounders_level_3 = set()

    target_material_level_1 = ""
    target_material_level_2 = ""
    target_material_level_3 = ""

    # chose any set of materials that is different from the target material, at the same category level
    for material_level1 in material_taxonomy:
        for material_level2 in material_level1["level_2_categories"]:
            for material in material_level2["level_3_items"]:
                if material["role"] == "CONFOUNDER":
                    confounders_level_1.add(material_level1["level_1_name"].lower())
                    confounders_level_2.add(material_level2["level_2_name"].lower())
                    confounders_level_3.add(material["name"].lower())
                elif material["name"] == target_material:
                    target_material_level_1 = material_level1["level_1_name"].lower()
                    target_material_level_2 = material_level2["level_2_name"].lower()
                    target_material_level_3 = material["name"].lower()

    confounders_level_3.discard(target_material)
    confounders_level_2.discard(target_material_level_2)
    confounders_level_1.discard(target_material_level_1)

    # returning confounders based on target level
    if target_level == 1:
        if target_material_level_1 in confounders_level_1:
            raise ValueError("Target material level 1 found in confounders!")
        return list(confounders_level_1), target_material_level_1
    elif target_level == 2:
        if target_material_level_2 in confounders_level_2:
            raise ValueError("Target material level 2 found in confounders!")
        return list(confounders_level_2), target_material_level_2
    else:
        if target_material_level_3 in confounders_level_3:
            raise ValueError("Target material level 3 found in confounders!")
        return list(confounders_level_3), target_material_level_3
