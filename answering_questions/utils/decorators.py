from utils.config import get_config
from utils.all_objects import get_gso_mapping
from utils.helpers import extract_attributes, minimum_n_visible_objects

from utils.my_exception import ImpossibleToAnswer

gso_mapping = get_gso_mapping()

MIN_PIXELS_VISIBLE = get_config()['min_pixels_visible']

def with_resolved_attributes(func):
    def wrapper(world_state, question, destination_simulation_id_path, *args, **kwargs):        
        attributes = extract_attributes(question)
        current_world_number_of_objects = len(world_state["objects"])

        if not minimum_n_visible_objects(
            world_state, n_objects=1, min_pixels=MIN_PIXELS_VISIBLE
        ): 
            print("Not enough objects with minimum pixels visible")
            raise ImpossibleToAnswer("Not enough objects with minimum pixels visible")

        # Useful attributes without need of recomputation every time in each function
        list_timesteps = list(world_state["simulation"].keys())
        timestep_start = list_timesteps[0]
        timestep_end = list_timesteps[-1]
                
        kwargs.update(
            {
                "timestep_start": timestep_start,
                "timestep_end": timestep_end,
                "current_world_number_of_objects": current_world_number_of_objects,
                "destination_simulation_id_path": destination_simulation_id_path, # to add /render and get the images directly
            }
        )

        # adaptor part to original names format
        for obj_id, object in world_state["objects"].items():
            object["id"] = obj_id
            object["name"] = gso_mapping[object["model"]]['name']
            # object.pop('model', None)

        # Pass them along so the wrapped function can use them
        return func(world_state, question, attributes["attributes"], *args, **kwargs)

    return wrapper
