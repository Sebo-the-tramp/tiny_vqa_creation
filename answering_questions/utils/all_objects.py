import json

with open("./utils/json/all_objects_data.json", "r") as f:
    all_objects = json.load(f)

all_materials = set()
all_objects_names = set()

gso_mapping = {}

with open("./utils/json/gso_mapping.json", "r") as f:
    gso_mapping = json.load(f)


def get_gso_mapping():
    return gso_mapping


def get_all_objects_names():
    # Using cached version
    if len(all_objects_names) > 0:
        return list(all_objects_names)

    for obj in gso_mapping.values():
        all_objects_names.add(obj["name"])

    return list(all_objects_names)
