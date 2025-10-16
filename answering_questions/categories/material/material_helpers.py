from typing import List


def get_all_materials() -> List[str]:
    # TODO this is mocked for now, we should return the materials from the obejcts in the scene
    return [
        "metal",
        "rubber",
        "plastic",
        "wood",
        "glass",
        "fabric",
        "stone",
        "ceramic",
        "paper",
        "leather",
    ]


def get_all_materials_in_scene(world_state) -> List[str]:
    materials_in_scene = set()
    for _, obj in world_state.get("objects", []).items():
        if "material" in obj:
            materials_in_scene.add(obj["material"])
    return list(materials_in_scene)
