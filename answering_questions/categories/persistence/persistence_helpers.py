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
VISIBILITY_THRESHOLD = 0.05
CLIP_LENGTH = get_config()["clip_length"]


# I think this is the most important one
def check_visibility_sequence(
    world_state: WorldState, obj_id: str, all_timesteps: Sequence[str],
    min_ones: int = 4, min_zeros: int = 4, gap_limit: int = 1
) -> Sequence[int]:

    phase = "ones"  # expecting ones-block first

    ones_count = zeros_count = 0
    ones_gaps = zeros_gaps = 0
    first_zero_timestep = None
    found = False

    for t in all_timesteps[::FRAME_INTERLEAVE]:
        obj_states = world_state["simulation"][t]["objects"]
        visible = (
            obj_id in obj_states
            and obj_states[obj_id]["fov_visibility"] >= VISIBILITY_THRESHOLD
        )
        bit = 1 if visible else 0

        if phase == "ones":
            if bit == 1:
                ones_count += 1
                ones_gaps = 0
            else:
                ones_gaps += 1

                # Block satisfied and first zero detected → start zeros phase
                if ones_count >= min_ones:
                    phase = "zeros"
                    zeros_count = 1
                    zeros_gaps = 0
                    first_zero_timestep = t
                    continue

            # If gap tolerance exceeded → reset
            if ones_gaps > gap_limit:
                ones_count = 0
                ones_gaps = 0

        elif phase == "zeros":
            if bit == 0:
                zeros_count += 1
                zeros_gaps = 0
                # record first moment of invisibility
                if first_zero_timestep is None:
                    first_zero_timestep = t
            else:
                zeros_gaps += 1

            # If gap tolerance exceeded → reset zeros-phase
            if zeros_gaps > gap_limit:
                consecutive_visible = zeros_gaps
                zeros_count = 0
                zeros_gaps = 0
                first_zero_timestep = None  # must restart detection
                phase = "ones"
                ones_count = consecutive_visible  # reuse the visible streak we just observed
                ones_gaps = 0
                continue

            # Full pattern detected
            if zeros_count >= min_zeros:
                found = True
                print("RETURNING at timestep:", found, first_zero_timestep)
                return found, first_zero_timestep

    return found, None

def compute_visibility_counts(world_state, timesteps):
    counts = []

    for t in timesteps:
        visible_count = 0
        objects = world_state["simulation"][t]["objects"]

        for obj in objects.values():
            if obj["fov_visibility"] >= VISIBILITY_THRESHOLD:
                visible_count += 1

        counts.append(visible_count)

    return counts

def find_first_visibility_drop(counts):
    for i in range(1, len(counts)):
        if counts[i] < counts[i-1]:       # drop ≥ 1
            return i                      # index of timestep where drop starts
    return None
