from __future__ import annotations

import json

from typing import Any, Mapping, Union

# from utils.config import get_config

Number = Union[int, float]
WorldState = Mapping[str, Any]
QuestionPayload = Mapping[str, Any]
Answer = Union[int, float, str]


## --- Helper functions --- ##

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
