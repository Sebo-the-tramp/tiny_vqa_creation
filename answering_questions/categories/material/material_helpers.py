from typing import List

from utils.all_objects import get_all_materials as get_all_materials_all_objects


def get_all_materials() -> List[str]:
    return list(get_all_materials_all_objects())


def get_all_materials_in_scene(world_state) -> List[str]:
    materials_in_scene = set()
    for _, obj in world_state.get("objects", []).items():
        materials_in_scene.add(obj["description"]["material_group"])
    return list(materials_in_scene)
