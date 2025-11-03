from utils.config import get_config
from utils.all_objects import get_gso_mapping
from utils.helpers import extract_attributes, minimum_n_visible_objects

from utils.my_exception import ImpossibleToAnswer

gso_mapping = get_gso_mapping()

MIN_PIXELS_VISIBLE = get_config()['min_pixels_visible']


# this is a patch to solve for materials not being in all objects

# all material for reference
# ['plastic', 'metal', 'leather', 'wood', 'paper/cardboard', 'plush/fiberfill', 'ceramic', 'foam', 'fabric/textile', 'mixed (paper + plastic)']

material_patch = {
    "Chefmate_8_Frypan": "metal",
    "Dog": "plush/fiberfill",
    "Jansport_School_Backpack_Blue_Streak": "fabric/textile",
    "KS_Chocolate_Cube_Box_Assortment_By_Neuhaus_2010_Ounces": "paper/cardboard",
    "Marvel_Avengers_Titan_Hero_Series_Doctor_Doom": "plastic",
    "Nickelodeon_Teenage_Mutant_Ninja_Turtles_Raphael": "plastic",
    "Nintendo_Yoshi_Action_Figure": "plastic",
    "Ortho_Forward_Facing": "plush/fiberfill",
    "Ortho_Forward_Facing_CkAW6rL25xH": "metal",
    "Ortho_Forward_Facing_QCaor9ImJ2G": "plush/fiberfill",
    "Playmates_nickelodeon_teenage_mutant_ninja_turtles_shredder": "plastic",
    "Racoon": "plush/fiberfill",
    "Retail_Leadership_Summit_eCT3zqHYIkX": "fabric/textile",
    "Retail_Leadership_Summit_tQFCizMt6g0": "fabric/textile",
    "Rexy_Glove_Heavy_Duty_Large": "plastic",
    "Shark": "plastic",
    "Squirrel": "plush/fiberfill",
    "Vtech_Roll_Learn_Turtle": "plastic",
    "Weisshai_Great_White_Shark": "plastic",
    "Whey_Protein_Vanilla": "plastic",
}

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
            if object['description'].get('material_group', None) is None:
                object["material_group"] = material_patch[object['model']]            

        # Pass them along so the wrapped function can use them
        return func(world_state, question, attributes["attributes"], *args, **kwargs)

    return wrapper
