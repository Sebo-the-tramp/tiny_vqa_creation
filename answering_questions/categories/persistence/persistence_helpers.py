import random
import numpy as np
from scipy.signal import convolve2d

from typing import Any, Mapping, Sequence

from utils.config import get_config
from utils.helpers import is_object_visible_v3, get_visibility_mask

from utils.my_exception import ImpossibleToAnswer

WorldState = Mapping[str, Any]

SAMPLING_RATE = get_config()["sampling_rate"]
RENDER_STEP = 1.0 / SAMPLING_RATE
FRAME_INTERLEAVE = 4  # custom only for temporal questions (heuristic)
MIN_PIXELS_VISIBLE = 300
VISIBILITY_THRESHOLD = 0.3
CLIP_LENGTH = get_config()["clip_length"]
HIGH_PIXEL_COUNT_THRESHOLD = 2000  # heuristic for occluded but distinct objects


def choose_best_window_object_id(world_state, object_proposed):
    # check fo unique object that appears and then disappears for time interval
    chosen_object_id = None

    all_objects_ids = set([str(i) for i in range(1, len(world_state["objects"]) + 1)])

    for obj_id in object_proposed:
        initial_timestep_index = object_proposed[str(obj_id)][2]
        final_timestep_index = object_proposed[str(obj_id)][3]

        for obj_id_check in object_proposed:
            if str(obj_id) == str(obj_id_check):
                continue

            initial_timestep_index_check = object_proposed[str(obj_id_check)][2]
            final_timestep_index_check = object_proposed[str(obj_id_check)][3]

            if final_timestep_index_check == -1 or initial_timestep_index_check == -1:
                all_objects_ids.discard(str(obj_id_check))
                continue

            # check for overlap
            if not (
                final_timestep_index < initial_timestep_index_check
                or final_timestep_index_check < initial_timestep_index
            ):
                all_objects_ids.discard(str(obj_id_check))
                all_objects_ids.discard(str(obj_id))

        if len(all_objects_ids) == 0:
            raise ImpossibleToAnswer(
                "More than one object found that appears and then disappears."
            )

    chosen_object_id = (
        random.choice(list(all_objects_ids)) if len(all_objects_ids) == 1 else None
    )
    return chosen_object_id


def get_maximum_windows_for_each_object(world_state: WorldState):
    """This function returns for each object the best start and end timestep
    where the object is highly visible at the start and not visible at the end."""

    _, visibility_percentage_matrix = get_visibility_mask(world_state)

    object_proposed = {}

    # get optimal timesteps
    for idx, object_percentage_array in enumerate(visibility_percentage_matrix):
        best_current_object_max = 0
        best_current_object_min = 100
        best_current_object_max_index = -1
        best_current_object_min_index = -1

        highest_visible_percentage_index = np.argmax(object_percentage_array)

        # here we could even look around the max like +/- 2 frame
        # here instead of everything I could get the 8-16-24th frames away and see which one is the lowest
        for i in range(-2, 2):
            start_index = highest_visible_percentage_index + i

            # check bounds
            if start_index < 0 or start_index >= len(object_percentage_array):
                continue

            start_index_visibility = object_percentage_array[start_index]

            # check if above threshold
            if start_index_visibility > 0.9 * 100:  # maybe we can tune this
                # this is to ensure we have enough frames to look ahead and not go OOB
                candidates = [
                    k
                    for k in (1, 2, 3, 4)
                    if start_index + k * CLIP_LENGTH - 1 < len(object_percentage_array)
                ]
                if not candidates:
                    continue  # no valid window

                max_k = max(candidates)
                end_index = (
                    (
                        np.argmin(
                            object_percentage_array[
                                start_index + (CLIP_LENGTH - 1) : start_index
                                + (max_k * CLIP_LENGTH)
                            ][:: (CLIP_LENGTH - 1)]
                        )
                        * (CLIP_LENGTH - 1)
                    )
                    + start_index
                    + (CLIP_LENGTH - 1)
                )  # adding back the start index
                end_index_visibility = object_percentage_array[end_index]

                if end_index_visibility < 0.1 * 100:
                    if start_index_visibility > best_current_object_max:
                        best_current_object_max = start_index_visibility
                        best_current_object_min = end_index_visibility
                        best_current_object_max_index = start_index
                        best_current_object_min_index = end_index

                        # exit early if first is >0.99 and last <0.01
                        if (
                            best_current_object_max > 0.99 * 100
                            and best_current_object_min < 0.01 * 100
                        ):
                            break

        object_proposed[str(idx + 1)] = (
            int(best_current_object_max),
            int(best_current_object_min),
            int(best_current_object_max_index),
            int(best_current_object_min_index),
        )

    return object_proposed


def generate_windows(T, window=8):
    for s in range(4, 0, -1):  # now includes s=1
        if s == 3:
            continue
        idx = np.arange(0, window * s, s)
        if idx[-1] < T:
            yield idx, s


def check_all_other_object_stable(visibility_mask, obj_id, start_idx, end_idx, stride):
    all_row_correct = True

    for row_id, obj_id_visibility in enumerate(visibility_mask):
        if row_id == obj_id:
            continue

        considered_visible_mask = obj_id_visibility[start_idx : end_idx + 1][::stride]
        sum_mask = np.sum(considered_visible_mask)

        row_correct = sum_mask == len(considered_visible_mask) or sum_mask == 0

        all_row_correct = all_row_correct and row_correct

    return all_row_correct


def get_optimal_timestep_interval_single(world_state: WorldState) -> Sequence[str]:
    visibility_mask, visibility_percentage_matrix = get_visibility_mask(world_state)

    T = len(visibility_mask[0])

    best = None
    best_score = float("inf")

    for obj_id in [int(i) - 1 for i in world_state["objects"].keys()]:
        visibility_mask_obj = visibility_mask[obj_id]

        for idx, current_stride in generate_windows(T):
            for s in range(0, T - (CLIP_LENGTH * current_stride) - 1):
                win = idx + s

                # check if it is a candidate
                if (
                    visibility_mask_obj[win[0]] == 1
                    and visibility_mask_obj[win[-1]] == 0
                ):
                    start_idx = win[0]
                    end_idx = win[-1]

                    if check_all_other_object_stable(
                        visibility_mask, obj_id, start_idx, end_idx, current_stride
                    ):
                        score = visibility_percentage_matrix[obj_id][idx + s][
                            -1
                        ]  # smaller is better
                        if score < best_score:
                            best_score = score
                            best = (
                                start_idx,
                                list(world_state["simulation"].keys())[start_idx],
                                end_idx,
                                list(world_state["simulation"].keys())[end_idx],
                                visibility_percentage_matrix[obj_id][idx + s][0],
                                score,
                                str(obj_id + 1),
                            )

    if best:
        return best

    raise ImpossibleToAnswer("Could not determine good interval")


def get_optimal_timestep_interval(world_state: WorldState) -> Sequence[str]:
    visibility_mask, visibility_percentage_matrix = get_visibility_mask(world_state)

    # Aggregate visibility metrics across all objects for each timestep
    visible_object_count = np.sum(visibility_mask, axis=0)
    percentage_visible = np.sum(visibility_percentage_matrix, axis=0)

    max_object_count = visible_object_count.max()
    first_max_index = np.argmax(visible_object_count == max_object_count)

    T = len(visible_object_count)

    best = None
    best_score = float("inf")

    for idx, current_stride in generate_windows(T):
        for s in range(
            first_max_index, T - (CLIP_LENGTH * current_stride) - first_max_index
        ):
            win = idx + s
            if (
                visible_object_count[win[0]] > visible_object_count[win[-1]]
                and visible_object_count[win[0]] > visible_object_count[win[-2]]
            ):
                score = percentage_visible[idx + s][-1]  # smaller is better

                if score < best_score:
                    best_score = score
                    best = (
                        (idx + s)[0],
                        list(world_state["simulation"].keys())[(idx + s)[0]],
                        (idx + s)[-1],
                        list(world_state["simulation"].keys())[(idx + s)[-1]],
                        percentage_visible[idx + s][0],
                        score,
                    )

    if best:
        return best

    raise ImpossibleToAnswer("Could not determine good interval")


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
        bit = 1 if is_object_visible_v3(world_state, obj_id, t) else 0

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

        for obj_id, obj_state in objects.items():
            if is_object_visible_v3(world_state, obj_id, t):
                visible_count += 1

        counts.append(visible_count)

    return counts


def find_first_visibility_drop(counts):
    for i in range(1, len(counts)):
        if counts[i] < counts[i - 1]:  # drop ≥ 1
            return i  # index of timestep where drop starts
    return None
