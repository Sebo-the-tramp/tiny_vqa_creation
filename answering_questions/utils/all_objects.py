import json

with open("./utils/full_list_objects.json", "r") as f:
    all_objects = json.load(f)


def get_all_objects_names():
    names = set()
    for obj in all_objects:
        names.add(obj["name"])
    return list(names)


def get_all_objects_categoories():
    categories = set()
    for obj in all_objects:
        categories.add(obj["category"])
    return list(categories)


def get_all_objects_ID():
    ids = set()
    for obj in all_objects:
        ids.add(obj["id"])
    return list(ids)
