from utils.my_exception import ImpossibleToAnswer

## --- Resolver functions -- ##

from typing import (
    Any,
    Mapping,
    Sequence,
    Tuple,
    Union,
)
WorldState = Mapping[str, Any]

from utils.config import get_config

SAMPLING_RATE = get_config()["sampling_rate"]
RENDER_STEP = 1.0 / SAMPLING_RATE
FRAME_INTERLEAVE = 4  # custom only for temporal questions (heuristic)
MIN_PIXELS_VISIBLE = get_config()["min_pixels_visible"]
CLIP_LENGTH = get_config()["clip_length"]


# I think this is the most important one
def check_visibility_sequence(
    world_state: WorldState, obj_id: str, all_timesteps: Sequence[str],
    min_ones: int = 4, min_zeros: int = 4, gap_limit: int = 1
) -> Sequence[int]:

    phase = "ones"  # expecting ones-block first

    ones_count = zeros_count = 0
    ones_gaps = zeros_gaps = 0
    final_timestep = None
    found = False

    for t in all_timesteps[::FRAME_INTERLEAVE]:
        obj_states = world_state["simulation"][t]["objects"]
        visible = (
            obj_id in obj_states
            and obj_states[obj_id]["infov_pixels"] >= MIN_PIXELS_VISIBLE
        )
        bit = 1 if visible else 0

        if phase == "ones":
            if bit == 1:
                ones_count += 1
            else:
                ones_gaps += 1

            # If gap tolerance exceeded → reset
            if ones_gaps > gap_limit:
                ones_count = ones_gaps = 0

            # If ones-block satisfied → move to zeros phase
            if ones_count >= min_ones:
                phase = "zeros"

        elif phase == "zeros":
            if bit == 0:
                zeros_count += 1
            else:
                zeros_gaps += 1

            # If gap tolerance exceeded → reset zeros-phase
            if zeros_gaps > gap_limit:
                zeros_count = zeros_gaps = 0

            # Full pattern detected
            if zeros_count >= min_zeros:
                found = True
                final_timestep = t
                break

    return found, final_timestep

def compute_visibility_counts(world_state, timesteps):
    counts = []

    for t in timesteps:
        visible_count = 0
        objects = world_state["simulation"][t]["objects"]

        for obj in objects.values():
            if obj["infov_pixels"] >= MIN_PIXELS_VISIBLE:
                visible_count += 1

        counts.append(visible_count)

    return counts

def find_first_visibility_drop(counts):
    for i in range(1, len(counts)):
        if counts[i] < counts[i-1]:       # drop ≥ 1
            return i                      # index of timestep where drop starts
    return None