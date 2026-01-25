from utils.all_objects import get_gso_mapping
from utils.helpers import extract_attributes

from utils.my_exception import ImpossibleToAnswer

import numpy as np

gso_mapping = get_gso_mapping()

def with_resolved_attributes(func):
    def wrapper(world_state, question, destination_simulation_id_path, *args, **kwargs):
        attributes = extract_attributes(question)
        current_world_number_of_objects = len(world_state["objects"])

        # Useful attributes without need of recomputation every time in each function
        list_timesteps = list(world_state["simulation"].keys())
        timestep_start = list_timesteps[0]
        timestep_end = list_timesteps[-1]

        is_counterfactual = "dl3dv-counterfact" in destination_simulation_id_path
        
        kwargs.update(
            {
                "timestep_start": timestep_start,
                "timestep_end": timestep_end,
                "current_world_number_of_objects": current_world_number_of_objects,
                "destination_simulation_id_path": destination_simulation_id_path,  # to add /render and get the images directly
                "counter_factual": is_counterfactual,
            }
        )

        # adaptor part to original names format
        for obj_id, object in world_state["objects"].items():
            object["id"] = obj_id
            object["name"] = gso_mapping[object["model"]]["name"]

        # Pass them along so the wrapped function can use them
        return func(world_state, question, attributes["attributes"], *args, **kwargs)

    return wrapper


def with_resolved_attributes_cf(func):
    def wrapper(
        world_state_og,
        world_state_modified,        
        question,
        destination_simulation_id_path,
        *args,
        **kwargs,
    ):
        attributes = extract_attributes(question)
        current_world_number_of_objects = len(world_state_modified["objects"])

        # Useful attributes without need of recomputation every time in each function
        list_timesteps = list(world_state_modified["simulation"].keys())
        timestep_start = list_timesteps[0]
        timestep_end = list_timesteps[-1]

        # adaptor part to original names format
        for obj_id, object in world_state_modified["objects"].items():
            object["id"] = obj_id
            object["name"] = gso_mapping[object["model"]]["name"]

        # adaptor part to original names format -> also for original even though st should be just for original
        for obj_id, object in world_state_og["objects"].items():
            object["id"] = obj_id
            object["name"] = gso_mapping[object["model"]]["name"]            

        # object_moved_id
        transform_per_object_mod = {
            k: {"translation": v["initial_condition"]["translation"], "rotation": v["initial_condition"]["rotation"], "scale": v["scale"]}
            for k, v in world_state_modified["objects"].items()
        }

        transform_per_object_og = {
            k: {"translation": v["initial_condition"]["translation"], "rotation": v["initial_condition"]["rotation"], "scale": v["scale"]}
            for k, v in world_state_og["objects"].items()
        }

        object_moved_id = -1
        scale_ratio = -1.0
        for object_id in transform_per_object_mod.keys():            
            if not np.allclose(transform_per_object_mod[object_id]['scale'], transform_per_object_og[object_id]['scale']):
                # print("SCALE CHANGED")
                # print("scale_difference:", transform_per_object_mod[object_id]['scale'], transform_per_object_og[object_id]['scale'])
                scale_ratio = np.array(transform_per_object_mod[object_id]['scale']) / np.array(transform_per_object_og[object_id]['scale'])
                # print("scale_ratio:", scale_ratio)
                # print("path_id", transform_per_object_mod[object_id]['scale'], transform_per_object_og[object_id]['scale'],  question['_simulation_id'])
                if scale_ratio < 1.4 or scale_ratio > 10.0:
                    raise ImpossibleToAnswer("4 - Scale change too little or to big to be considered a valid counterfactual change.")

            if not np.allclose(transform_per_object_mod[object_id]['translation'], transform_per_object_og[object_id]['translation']) or \
               not np.allclose(transform_per_object_mod[object_id]['scale'], transform_per_object_og[object_id]['scale']):
                object_moved_id = object_id
                break

        # Also here we need to do a big check before accepting the counterfactual        
        # 1) there should not be any other object with the same name as the moved one that could create ambiguity
        # 2) Visibility check at start and end timesteps        
        if object_moved_id != -1:            
            # Sanity check
            moved_object_name = world_state_modified["objects"][object_moved_id]["name"]
            count_same_name = 0
            for obj_id, obj in world_state_modified["objects"].items():
                if obj["name"] == moved_object_name:
                    count_same_name += 1
            if count_same_name > 1:
                raise ImpossibleToAnswer("4 - Multiple objects with the same name as the moved object exist, creating ambiguity.")
            
        if object_moved_id == -1 and 'low-gravity' in question['_simulation_id']:
            object_moved_id = list(world_state_modified["objects"].keys())[-1]  # default to first object if none found if we are in low-gravity cf

        kwargs.update(
            {
                "timestep_start": timestep_start,
                "timestep_end": timestep_end,
                "current_world_number_of_objects": current_world_number_of_objects,
                "destination_simulation_id_path": destination_simulation_id_path,  # to add /render and get the images directly
                "object_moved_id": object_moved_id,
            }
        )

        chosen_object = world_state_modified["objects"][object_moved_id]        
        resolved_attributes = {
            "OBJECT-CF": {"choice": chosen_object, "category": "OBJECT"},
            "OBJECT": {"choice": chosen_object, "category": "OBJECT"}
            }

        if scale_ratio != -1.0:
            resolved_attributes["SCALE"] = {"choice": round(float(scale_ratio),2), "category": "SCALE"}
        
        return func(
            world_state_og,
            world_state_modified,
            question,
            attributes["attributes"],
            resolved_attributes,
            *args,
            **kwargs,
        )

    return wrapper
