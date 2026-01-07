import numpy as np
from scipy.signal import convolve2d

from typing import Any, Mapping, Sequence

from utils.config import get_config
from utils.helpers import iter_objects

WorldState = Mapping[str, Any]

SAMPLING_RATE = get_config()["sampling_rate"]
RENDER_STEP = 1.0 / SAMPLING_RATE
FRAME_INTERLEAVE = 4  # custom only for temporal questions (heuristic)
MIN_PIXELS_VISIBLE = 300
VISIBILITY_THRESHOLD = 0.3
CLIP_LENGTH = get_config()["clip_length"]
HIGH_PIXEL_COUNT_THRESHOLD = 2000  # heuristic for occluded but distinct objects


def get_optimal_timestep_interval(world_state: WorldState) -> Sequence[str]:
    visibility_mask, visibility_percentage_mask, visibility_pixels_mask = (
        get_visibility_mask(world_state)
    )

    # Aggregate visibility metrics across all objects for each timestep
    visible_object_count = np.sum(visibility_mask, axis=0)
    total_visibility_percentage = np.sum(visibility_percentage_mask, axis=0)
    total_visibility_pixels = np.sum(visibility_pixels_mask, axis=0)

    # Composite score: objects are weighted most heavily, then percentage, then pixels
    visibility_score = (
        visible_object_count
        + (total_visibility_percentage / 1000)
        + (total_visibility_pixels / 10000)
    )

    # Find timesteps with maximum object visibility
    max_object_count = visible_object_count.max()
    has_max_objects = visible_object_count == max_object_count

    # Among max-object timesteps, find the one with highest visibility score
    masked_scores = np.where(has_max_objects, visibility_score, 0)
    initial_timestep_index = np.argmax(masked_scores)
    initial_timestep = list(world_state["simulation"].keys())[initial_timestep_index]

    # Create masks for further processing
    before_optimal = has_max_objects.copy()
    before_optimal[:initial_timestep_index] = True

    # this is to avoid choosing a timestep too far, where things might have changed too much    
    first_drop = np.argmax(~before_optimal)
    window_space = 10  # frames
    if (first_drop + window_space) < len(before_optimal):
        before_optimal[first_drop + window_space :] = True

    without_max_objects = ~before_optimal
    alternative_scores = np.where(without_max_objects, visibility_score, 0)

    # Find timestep with highest visibility score excluding max-object timesteps
    final_timestep_index = np.argmax(alternative_scores)
    final_timestep = list(world_state["simulation"].keys())[final_timestep_index]

    previous_phase = initial_timestep_index % CLIP_LENGTH
    next_phase = (
        (CLIP_LENGTH - previous_phase)
        if previous_phase != 0
        else initial_timestep_index
    )

    previous_phase_index = initial_timestep_index - previous_phase
    next_phase_index = initial_timestep_index + next_phase

    if previous_phase_index < 0:
        previous_phase_index = None
    if next_phase_index >= len(masked_scores):
        next_phase_index = None

    if previous_phase_index is not None and next_phase_index is not None:
        if (
            masked_scores[previous_phase_index] > masked_scores[next_phase_index]
            and masked_scores[next_phase_index] > 0
        ):
            initial_timestep_index = previous_phase_index
        elif (
            masked_scores[previous_phase_index] < masked_scores[next_phase_index]
            and masked_scores[previous_phase_index] > 0
        ):
            initial_timestep_index = next_phase_index
        else:
            pass  # keep original

    return (
        initial_timestep_index,
        initial_timestep,
        final_timestep_index,
        final_timestep,
    )


def get_visibility_mask(world_state: WorldState) -> Mapping[str, Sequence[int]]:
    all_timesteps = list(world_state["simulation"].keys())
    visibility_mask = np.zeros(
        (len(world_state["objects"]), len(all_timesteps)), dtype=int
    )

    visibility_percentage_mask = np.zeros(
        (len(world_state["objects"]), len(all_timesteps)), dtype=int
    )

    visibility_pixels_mask = np.zeros(
        (len(world_state["objects"]), len(all_timesteps)), dtype=int
    )

    for object in iter_objects(world_state):
        obj_id = object["id"]

        for t in all_timesteps:
            obj_states = world_state["simulation"][t]["objects"]

            # pixels_visible = obj_states[obj_id]['infov_pixels_visible'] + obj_states[obj_id]['infov_pixels_void']
            pixels_void = obj_states[obj_id]["infov_pixels_void"]
            pixels_visible = obj_states[obj_id]["infov_pixels_visible"]
            fov_visibility = obj_states[obj_id]["fov_visibility"]

            visibility_percentage_mask[int(obj_id) - 1, all_timesteps.index(t)] = int(
                fov_visibility * 100
            )
            visibility_pixels_mask[int(obj_id) - 1, all_timesteps.index(t)] = (
                pixels_visible
            )

            visible = (
                # Case 1: Object is mostly unoccluded
                # (fov_visibility >= VISIBILITY_THRESHOLD or pixels_visible >= MIN_PIXELS_VISIBLE)
                fov_visibility >= VISIBILITY_THRESHOLD and pixels_visible > pixels_void
            )
            bit = 1 if visible else 0
            index_timestep = all_timesteps.index(t)
            visibility_mask[int(obj_id) - 1, index_timestep] = bit

    return visibility_mask, visibility_percentage_mask, visibility_pixels_mask


def get_visibility_change(visibility_mask: np.array) -> Sequence[int]:
    kernel = np.array([[-1, 1]])
    padded_visibility = np.pad(visibility_mask, ((0, 0), (1, 1)), mode="edge")
    changes_in_visibility = convolve2d(padded_visibility, kernel, mode="valid")
    significant_changes = (
        -1 * np.sign(changes_in_visibility) * np.abs(changes_in_visibility)
    )

    return significant_changes


# I think this is the most important one
def check_visibility_sequence(
    world_state: WorldState,
    obj_id: str,
    all_timesteps: Sequence[str],
    min_ones: int = 4,
    min_zeros: int = 4,
    gap_limit: int = 1,
) -> Sequence[int]:
    phase = "ones"  # expecting ones-block first

    ones_count = zeros_count = 0
    ones_gaps = zeros_gaps = 0
    first_zero_timestep = None
    found = False

    for t in all_timesteps[::FRAME_INTERLEAVE]:
        obj_states = world_state["simulation"][t]["objects"]

        pixels_visible = (
            obj_states[obj_id]["infov_pixels_visible"]
            + obj_states[obj_id]["infov_pixels_void"]
        )
        fov_visibility = obj_states[obj_id]["fov_visibility"]

        visible = (
            obj_id in obj_states
            and pixels_visible >= MIN_PIXELS_VISIBLE
            and fov_visibility >= VISIBILITY_THRESHOLD
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
                ones_count = (
                    consecutive_visible  # reuse the visible streak we just observed
                )
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

        for _, obj_states in objects.items():
            pixels_visible = (
                obj_states["infov_pixels_visible"] + obj_states["infov_pixels_void"]
            )
            fov_visibility = obj_states["fov_visibility"]

            if (
                pixels_visible >= MIN_PIXELS_VISIBLE
                and fov_visibility >= VISIBILITY_THRESHOLD
            ):
                visible_count += 1

        counts.append(visible_count)

    return counts


def find_first_visibility_drop(counts):
    for i in range(1, len(counts)):
        if counts[i] < counts[i - 1]:  # drop ≥ 1
            return i  # index of timestep where drop starts
    return None
