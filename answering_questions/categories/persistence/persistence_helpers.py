import random
import numpy as np

from typing import Any, Mapping

from utils.config import get_config
from utils.helpers import get_visibility_mask_soft

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

    _, visibility_percentage_matrix = get_visibility_mask_soft(world_state)

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
