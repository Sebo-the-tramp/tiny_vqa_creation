import json

with open("./utils/all_objects_data.json", "r") as f:
    all_objects = json.load(f)

all_materials = set()
all_objects_names = set()


def get_all_objects_names():
    # Using cached version
    if len(all_objects_names) > 0:
        return list(all_objects_names)

    for model in all_objects.keys():
        all_objects_names.add(model)
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
