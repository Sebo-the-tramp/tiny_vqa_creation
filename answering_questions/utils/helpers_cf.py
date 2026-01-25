import random
import numpy as np

from utils.helpers import (
    is_object_visible,    
    get_visibility_mask,
    get_timestep_from_idx,
)

from utils.my_exception import ImpossibleToAnswer

CLIP_LENGTH = 8

def get_start_end_timesteps_visible_both(world_state_og, world_state_mod, object_counterfactual_id):
    
    # now we use the object that has been moved and we always need to consider the start and end timestep to 
    # be either 8-16-24 or max 32, because we always need to have the first timestep visible
    #
    timestep_start_index = 0  # always start at t=0
    timestep_start = get_timestep_from_idx(timestep_start_index)

    min_len_world = min(
        len(world_state_og["simulation"]),
        len(world_state_mod["simulation"]),
    )

    candidates_t_end = [
        k for k in (1, 2, 3, 4) if (k * (CLIP_LENGTH - 1)) <= min_len_world - 1
    ]

    # we only need to check that is visible in the original sim at the end timestep
    candidates_t_end_filtered = [
        k for k in candidates_t_end 
        if is_object_visible(
            world_state_og, object_counterfactual_id, get_timestep_from_idx(k * (CLIP_LENGTH - 1))
        )
    ]

    if is_object_visible(
        world_state_og, object_counterfactual_id, timestep_start
    ) is False:
        raise ImpossibleToAnswer("1 - The object in either sim is hidden at the start timestep.")

    if len(candidates_t_end_filtered) == 0:
        raise ImpossibleToAnswer("1 - The object in either sim is hidden at some of the possible end timesteps.")

    # choose randomly one of the candidates timesteps
    timestep_end_index = random.choice(candidates_t_end_filtered) * (CLIP_LENGTH - 1)
    timestep_end = get_timestep_from_idx(timestep_end_index)

    return timestep_start, timestep_end

def get_start_end_timesteps_visible_end(world_state_og, world_state_mod, object_counterfactual_id, steps=[1,2,3,4]):
    
    # now we use the object that has been moved and we always need to consider the start and end timestep to 
    # be either 8-16-24 or max 32, because we always need to have the first timestep visible
    timestep_start_index = 0  # always start at t=0
    timestep_start = get_timestep_from_idx(timestep_start_index)

    min_len_world = min(
        len(world_state_og["simulation"]),
        len(world_state_mod["simulation"]),
    )

    candidates_t_end = [
        k for k in steps if (k * (CLIP_LENGTH - 1)) <= min_len_world - 1
    ]

    candidates_t_end_filtered = [
        k for k in candidates_t_end 
        if is_object_visible(
            world_state_og, object_counterfactual_id, get_timestep_from_idx(k * (CLIP_LENGTH - 1))
        )
    ]

    if len(candidates_t_end_filtered) == 0:
        raise ImpossibleToAnswer("1 - The object in either sim is hidden at some of the possible end timesteps.")

    # choose randomly one of the candidates timesteps
    # we should chose k such that it maximizes the chance that the object is visible in the modified sim at the end timestep
    max_visible_objects = get_visibility_mask(
        world_state_mod, 
        max_timestep=get_timestep_from_idx(candidates_t_end_filtered[-1] * (CLIP_LENGTH - 1))
    )[0][:, ::CLIP_LENGTH-1][:,candidates_t_end_filtered[0]:].sum(axis=0) # -> here we need to start from the first of the candidates_t_end_filtered

    # I need to double check skipping first there

    # I changed here to always choose the best timestep instead of random
    # but also move one to ther front to avoid zero index and avoid having last as first thing.
    # okay there is sketch
    best_timestep_end_index = (np.argmax(max_visible_objects) + candidates_t_end_filtered[0]) * (CLIP_LENGTH - 1) # plus one else we would start from zero and that is kinda of invalid
    timestep_end = get_timestep_from_idx(best_timestep_end_index)

    return timestep_start, timestep_end


def get_random_timestep_end():
    return get_timestep_from_idx(random.choice([1, 2, 3, 4]) * (CLIP_LENGTH - 1))