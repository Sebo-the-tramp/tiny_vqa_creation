import json
import glob
import os


def read_simulation_json_files():
    # Get all JSON files ending with 'simulation.json' from ./data directory
    return glob.glob("**/simulation.json", recursive=True)


def read_simulation_kinematics_json_files():
    # Get all JSON files ending with 'simulation_kinematics.json' from ./data directory
    return glob.glob("**/simulation_kinematics.json", recursive=True)


def read_materials_group_json_files():
    path = "./Physics-gso/materials_for_objects.json"
    try:
        with open(path, "r", encoding="utf-8") as f:
            materials_group = json.load(f)
        return materials_group
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error reading {path}: {e}")
        return {}


if __name__ == "__main__":
    simulation_files = read_simulation_json_files()
    simulation_kinematics = read_simulation_kinematics_json_files()

    materials_group = read_materials_group_json_files()

    # invalidate any previsous generated data
    if len(simulation_kinematics) > 0:
        # remove files from kinematics
        for file in simulation_kinematics:
            print(f"Removing kinematics file: {file}")
            # os.remove(file)

    # merge exactly what we need
    for file in simulation_files:
        file_data = json.load(open(file, "r", encoding="utf-8"))

        for obj_id, obj in file_data["objects"].items():
            object_id_name = obj["model"]
            print(f"Processing object ID: {obj_id} with model name: {object_id_name}")

            new_file_path = os.path.join(
                "Physics-gso", "gso_batch_0", object_id_name, "materials_2.json"
            )
            new_data_object = json.load(open(new_file_path, "r", encoding="utf-8"))

            # Adding exactly and only the thing we need to the main file
            obj["description"]["material_group"] = materials_group[
                object_id_name
            ]  # grouped material
            obj["description"]["object_name_short"] = new_data_object[
                "object_name_short"
            ]
            obj["description"]["vlm_category"] = new_data_object[
                "vlm_category"
            ]  # this doesn't belong to any set of categories
            obj["description"]["visible_material_name"] = new_data_object[
                "visible_material_name"
            ]

            if obj["description"].get("category", None) is not None:
                obj["description"]["category_gso"] = obj["description"]["category"]
                obj["description"].pop("category", None)

        # print("####" * 10)
        # print(file_data['objects'])

        # Save the modified data to a new file
        json.dump(file_data, open(file, "w", encoding="utf-8"), indent=4)
        print(f"Updated file saved: {file}")
