import json

with open("./utils/json/all_objects_data.json", "r") as f:
    all_objects = json.load(f)

all_materials = set()
all_objects_names = set()

gso_mapping = {}

with open("/data0/sebastian.cavada/datasets/gso/gso_mapping.json", "r") as f:
    gso_mapping = json.load(f)

def get_gso_mapping():
    return gso_mapping


def get_all_scenes_segments():
    # TODO dummy for now
    scenes_segments = [
        "Statue",
        "Wood Bench",
        "Marble Bench",
        "Fountain",
        "Tree",
        "Lamp Post",
        "Trash Can",
        "Bushes",
        "Flower Bed",
        "Clock Tower",
        "Bridge",
    ]

    return list(scenes_segments)


def get_all_objects_names():
    # Using cached version
    if len(all_objects_names) > 0:
        return list(all_objects_names)

    for obj in gso_mapping.values():
        all_objects_names.add(obj['name'])

    return list(all_objects_names)


def get_all_objects_categoories():
    categories = set()
    for obj in all_objects.values():
        categories.add(obj["categories_gso"])
    return list(categories)


def get_all_objects_ID():
    ids = set()
    for obj_id in all_objects.keys():
        ids.add(obj_id)
    return list(ids)


def get_all_materials():
    # Using cached version
    if len(all_materials) > 0:
        return list(all_materials)
    for obj in all_objects.values():
        material = obj["material_group"]
        all_materials.add(material)
    return list(all_materials)
