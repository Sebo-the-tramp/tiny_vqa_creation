from __future__ import annotations

import re
import math
import random
from utils import seed_utils

from typing import (
    Any,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    List,
    cast,
)

import copy
import numpy as np

from copy import deepcopy
from utils.config import get_config
from utils.all_objects import get_gso_mapping

from utils.my_exception import ImpossibleToAnswer

from utils.frames_selection import (
    sample_frames_at_timesteps,
    sample_frames_before_timestep,
)

from utils.geometry import (
    world_to_camera_view,
    project_obb,
    external_points_2d,
    polygon_area,
)

Number = Union[int, float]
WorldState = Mapping[str, Any]
QuestionPayload = Mapping[str, Any]
Answer = Union[int, float, str]


SAMPLING_RATE = get_config()["sampling_rate"]
VISIBILITY_THRESHOLD = get_config()["visibility_threshold"]
FRAME_INTERLEAVE = get_config()["frame_interleave"]
CLIP_LENGTH = get_config()["clip_length"]
MIN_VISIBLE_PIXELS = get_config()["min_pixels_visible"]
TIMESTART = get_config()["timestart"]

SAMPLING_RATE = get_config()["sampling_rate"]
RENDER_STEP = 1.0 / SAMPLING_RATE

# I want to sample every quarter of a second
FRAME_STRIDE = int(
    -(-0.25 // RENDER_STEP)
)  # same as math.ceil(0.25 / RENDER_STEP) but better quarter of a second


def fill_questions(
    question,
    labels,
    correct_idx,
    world_state,
    final_timestep,
    resolved_attributes,
    initial_timestep=None,
    k_options=(1, 2, 3, 4),
) -> List:
    questions = []
    # 1) Keep the correct label before shuffling
    correct_label = labels[correct_idx]

    # there's a problem now with frame interleave we are allowing
    # dynamic frame interleave based on initial and final timestep
    if initial_timestep is None:
        final_timestep_index = world_state["simulation"][final_timestep]["frame_idx"]
        # we need to compute the closest initial timestep based the current timestep
        candidates = [
            k for k in k_options if final_timestep_index - (k * (CLIP_LENGTH - 1)) >= 0
        ]
        if len(candidates) == 0:
            raise ImpossibleToAnswer(
                "Not enough previous frames to determine visibility."
            )

        max_k = max(candidates)
        initial_timestep_index = final_timestep_index - (max_k * (CLIP_LENGTH - 1))
        initial_timestep = get_timestep_from_idx(initial_timestep_index)

    seed_material = "::".join(
        [
            str(question.get("_simulation_id", "")),
            str(question.get("_question_key", "")),
            str(final_timestep),
        ]
    )
    local_seed = seed_utils.seed_from_material(seed_material)
    local_rng = random.Random(local_seed)

    # 2) Shuffle a COPY so we don't mutate caller's list
    shuffled = labels[:]  # copy
    local_rng.shuffle(shuffled)

    # 3) Remap correct index AFTER shuffle using the saved label
    new_correct_idx = shuffled.index(correct_label)

    # Helper to build one item with its own copies
    def build_item(split):
        q_copy = copy.deepcopy(question)
        q_copy["task_splits"] = (
            split  # keep type consistent with your downstream expectations
        )
        # q_copy.pop("_question_key", None)
        # q_copy.pop("_simulation_id", None)
        fill_template(q_copy, resolved_attributes)

        if split == "single":
            frames = sample_frames_at_timesteps(world_state, [final_timestep])
        else:  # "multi"
            if initial_timestep is not None:
                initial_timestep_index = world_state["simulation"][initial_timestep][
                    "frame_idx"
                ]
                final_timestep_index = world_state["simulation"][final_timestep][
                    "frame_idx"
                ]
                effective_frame_interleave = (
                    final_timestep_index - initial_timestep_index
                ) // (CLIP_LENGTH - 1)
            else:
                effective_frame_interleave = FRAME_INTERLEAVE
            frames = sample_frames_before_timestep(
                world_state,
                final_timestep,
                num_frames=CLIP_LENGTH,
                frame_interleave=effective_frame_interleave,
            )

        # Pass a fresh copy of the shuffled labels for each item
        return [
            q_copy,
            shuffled[:],
            new_correct_idx,
            frames,
            world_state,
            resolved_attributes,
        ]

    if "single" in question["task_splits"]:
        questions.append(build_item("single"))

    if "multi" in question["task_splits"]:
        questions.append(build_item("multi"))

    return questions


def compute_counterfactual_string(
    question, resolved_attributes, world_state_og, world_state_modified, timestep
):
    # timestep = "0000.010"  # for test only
    transform_per_object = world_state_modified["config"]["scene"]["spawning"][
        "transform_per_object"
    ]

    object_id = list(transform_per_object.keys())[0]  # only one object changed

    real_object_id = str(int(object_id) + 1)
    object_name = world_state_modified["objects"][real_object_id]["name"]
    object_pos_og = get_position(world_state_og, real_object_id, timestep)
    object_pos_mod = get_position(world_state_modified, real_object_id, timestep)

    # project the position to the camera view
    # It only depends from the camera at the modified simulation.
    camera_position = get_camera_at_timestep(world_state_og, timestep)

    center_og = object_pos_og["center"]
    center_mod = object_pos_mod["center"]

    cam_pos_og = world_to_camera_view(np.array([center_og]), camera_position)
    cam_pos_mod = world_to_camera_view(np.array([center_mod]), camera_position)

    x_og, y_og, z_og = cam_pos_og[0]
    x_mod, y_mod, z_mod = cam_pos_mod[0]

    dx = x_mod - x_og
    dy = y_mod - y_og
    dz = z_mod - z_og

    horizontal_movement = "right" if dx > 0 else "left"
    vertical_movement = "down" if dy < 0 else "up"
    depth_movement = "closer to the camera" if dz < 0 else "further from the camera"

    parts = []

    if abs(dx) >= 0.1:
        parts.append(f"{abs(round_sig(dx, 2))} meters to the {horizontal_movement}")

    if abs(dy) >= 0.1:
        parts.append(f"{abs(round_sig(dy, 2))} meters {vertical_movement}")

    if abs(dz) >= 0.1:
        parts.append(f"{abs(round_sig(dz, 2))} meters {depth_movement}")

    def english_join(items):
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return " and ".join(items)
        return ", ".join(items[:-1]) + ", and " + items[-1]

    counterfact_phrase = f"Assume the {object_name} is moved"
    if parts:
        counterfact_phrase += " " + english_join(parts)
    counterfact_phrase += ". Under this new condition, "

    # print(counterfact_phrase)

    # print("Check pitagora:")
    # print("Distance in camera view:", math.sqrt(dx**2 + dy**2 + dz**2))
    # print(f"The above should be {d}")

    # # check the image and see the 2 center
    # # this is only for test

    # timestep_number = world_state_modified['simulation'][timestep]['frame_idx']

    # fake_photo = Image.open(
    #     f"/data0/sebastian.cavada/datasets/simulations_v3/dl3dv/random/2/c-1_no-2_d-4_s-dl3dv-all_models-hf-gso_MLP-10_smooth_h-10-40_seed-11_20251102_060343/render/{str(timestep_number).zfill(6)}.png"
    # )  # dummy image just to get width and height
    # numpy_image = np.array(fake_photo)

    # project_center_og_uv, z1 = project_points(np.array([center_og]), camera_position)
    # project_center_mod_uv, z2 = project_points(np.array([center_mod]), camera_position)

    # # add points to the image, red dots for object 1, blue dots for object 2
    # u1, v1 = int(project_center_og_uv[0][0]), int(project_center_og_uv[0][1])
    # if 0 <= u1 < numpy_image.shape[1] and 0 <= v1 < numpy_image.shape[0]:
    #     for i in range(-1, 2):
    #         for j in range(-1, 2):
    #             if 0 <= v1+i < numpy_image.shape[0] and 0 <= u1+j < numpy_image.shape[1]:
    #                 numpy_image[v1+i, u1+j] = [255, 0, 0]  # red dot

    # u2, v2 = int(project_center_mod_uv[0][0]), int(project_center_mod_uv[0][1])
    # if 0 <= u2 < numpy_image.shape[1] and 0 <= v2 < numpy_image.shape[0]:
    #     for i in range(-1, 2):
    #         for j in range(-1, 2):
    #             if 0 <= v2+i < numpy_image.shape[0] and 0 <= u2+j < numpy_image.shape[1]:
    #                 numpy_image[v2+i, u2+j] = [0, 0, 255]  # blue dot

    return counterfact_phrase


def fill_questions_cf(
    question,
    labels,
    correct_idx,
    world_state_og,
    world_state_modified,
    timestep,
    resolved_attributes,
    initial_timestep=None,
) -> List:
    questions = []
    # 1) Keep the correct label before shuffling
    correct_label = labels[correct_idx]

    seed_material = "::".join(
        [
            str(question.get("_simulation_id", "")),
            str(question.get("_question_key", "")),
            str(timestep),
        ]
    )
    local_seed = seed_utils.seed_from_material(seed_material)
    local_rng = random.Random(local_seed)

    # 2) Shuffle a COPY so we don't mutate caller's list
    shuffled = labels[:]  # copy
    local_rng.shuffle(shuffled)

    # 3) Remap correct index AFTER shuffle using the saved label
    new_correct_idx = shuffled.index(correct_label)

    # Helper to build one item with its own copies
    def build_item(split):
        q_copy = copy.deepcopy(question)
        q_copy["task_splits"] = (
            split  # keep type consistent with your downstream expectations
        )
        # check if spawning is present in the modified world state
        world_state_modified_spawning = (
            world_state_modified.get("config", {}).get("scene", {}).get("spawning", {})
        )

        if world_state_modified_spawning == {}:
            diff = "gravity"  # default
        else:
            if "metricscale" in world_state_modified_spawning.get(
                "transform_per_object", {}
            ).get("0", {}):
                diff = "2xsmaller"
            else:
                diff = "shift"

        if diff == "shift":
            counterfact = compute_counterfactual_string(
                question,
                resolved_attributes,
                world_state_og,
                world_state_modified,
                timestep,
            )
        elif diff == "2xsmaller":
            counterfact = "How would the answer change if the object is scaled down to half of its original size."
        elif diff == "gravity":
            counterfact = "How would the answer change if the gravity is reduced to 10% of its original value."

        fill_template_cf(q_copy, resolved_attributes, counterfact)

        if split == "single":
            frames = sample_frames_at_timesteps(world_state_modified, [timestep])
        else:  # "multi"
            if initial_timestep is not None:
                initial_ts_float = float(initial_timestep)
                ts_float = float(timestep)
                effective_frame_interleave = int(
                    (ts_float - initial_ts_float) * SAMPLING_RATE // (CLIP_LENGTH - 1)
                )
            else:
                effective_frame_interleave = FRAME_INTERLEAVE
            frames = sample_frames_before_timestep(
                world_state_modified,
                timestep,
                num_frames=CLIP_LENGTH,
                frame_interleave=effective_frame_interleave,
            )

        # Pass a fresh copy of the shuffled labels for each item
        return [
            q_copy,
            shuffled[:],
            new_correct_idx,
            frames,
            world_state_modified,
            resolved_attributes,
        ]

    if "single" in question["task_splits"]:
        questions.append(build_item("single"))

    if "multi" in question["task_splits"]:
        questions.append(build_item("multi"))

    return questions


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

# So far we have the following attributes:
# - <OBJECT> -> defines any unique object
#     <OBJECT_1> -> if there are multiple objects, _N will define the ID of the object
# - <OBJECT_CATEGORY> -> defines the category of objects (multiple unique <OBJECT>)
# - <TIME> -> the timesteps we want to measure
# - <MATERIAL> -> the material of the object
# - <MASS> -> the mass of the object.
# - <VOLUME> -> volume of the obejct
# - <SCENE> -> could be the scene itself or any segmented part of the scene
# - <VELOCITY> -> the velocity of an object
# - <CAMERA> -> the camera itself

# ----- General helpers -----

gso_mapping = get_gso_mapping()

units = {
    "DISTANCE": "meters",
    "MASS": "kilograms",
    "VOLUME": "cubic centimeters",
    "DENSITY": "kg/m3",
    "TIME": "seconds",
    "SPEED": "meters/second",
    "ACCELERATION": "meters/second^2",
}


def get_timestep_from_idx(idx: int) -> str:
    return f"{TIMESTART + float(idx) * RENDER_STEP:08.3f}"


def resolve_units(measurement: str) -> str:
    return units.get(measurement, "")


def get_random_integer(min_value: int, max_value: int) -> int:
    return random.randint(min_value, max_value)


def shuffle_array(array: List[int]) -> List[int]:
    random.shuffle(array)
    return array


def get_total_timesteps():
    # very important function for images<->state conversion
    # TODO supposing 100fps and 5 seconds of video
    return 100 * 5


def get_total_images():
    # very important function for images<->state conversion
    # TODO supposing 100fps and 5 seconds of video
    return 8


def get_visibility_mask(
    world_state: WorldState, max_timestep=None
) -> Mapping[str, Sequence[int]]:
    all_timesteps = list(world_state["simulation"].keys())
    max_timestep_index = (
        len(all_timesteps)
        if max_timestep is None
        else world_state["simulation"][max_timestep]["frame_idx"] + 1
    )  # +1 to include the max_timestep

    visibility_mask = np.zeros(
        (len(world_state["objects"]), max_timestep_index), dtype=int
    )

    visibility_percentage_matrix = np.zeros(
        (len(world_state["objects"]), max_timestep_index), dtype=int
    )

    for object in iter_objects(world_state):
        obj_id = object["id"]

        for t in all_timesteps[:max_timestep_index]:
            bit = 1 if is_object_visible(world_state, obj_id, t) else 0
            index_timestep = all_timesteps.index(t)
            visibility_mask[int(obj_id) - 1, index_timestep] = bit

            visibility_percentage_obj = (
                get_visibility_ratio_v3(world_state, obj_id, t) * 100.0
            )
            visibility_percentage_matrix[int(obj_id) - 1, all_timesteps.index(t)] = int(
                visibility_percentage_obj
            )

    return visibility_mask, visibility_percentage_matrix


def get_visibility_mask_soft(
    world_state: WorldState, max_timestep=None
) -> Mapping[str, Sequence[int]]:
    """
    This function is similar to get_visibility_mask but instead of hard cutting the visibility
    THRESHOLD of 1/1000 of the image size, it will allow also very small objects to be considered visible
    This is because it can be used for counting questions where even small objects are visible but
    NOT identifiable.
    """

    all_timesteps = list(world_state["simulation"].keys())
    max_timestep_index = (
        len(all_timesteps)
        if max_timestep is None
        else world_state["simulation"][max_timestep]["frame_idx"] + 1
    )  # +1 to include the max_timestep

    visibility_mask = np.zeros(
        (len(world_state["objects"]), max_timestep_index), dtype=int
    )

    visibility_percentage_matrix = np.zeros(
        (len(world_state["objects"]), max_timestep_index), dtype=int
    )

    for object in iter_objects(world_state):
        obj_id = object["id"]

        for t in all_timesteps[:max_timestep_index]:
            bit = 1 if is_object_visible_soft(world_state, obj_id, t) else 0
            index_timestep = all_timesteps.index(t)
            visibility_mask[int(obj_id) - 1, index_timestep] = bit

            visibility_percentage_obj = (
                get_visibility_ratio_v3_soft(world_state, obj_id, t) * 100.0
            )
            visibility_percentage_matrix[int(obj_id) - 1, all_timesteps.index(t)] = int(
                visibility_percentage_obj
            )

    return visibility_mask, visibility_percentage_matrix


def _clip_polygon_to_rect(points, width, height):
    def clip_edge(poly, edge_fn):
        if len(poly) == 0:
            return poly
        clipped = []
        for i in range(len(poly)):
            curr = poly[i]
            prev = poly[i - 1]
            curr_in = edge_fn(curr)
            prev_in = edge_fn(prev)
            if curr_in:
                if not prev_in:
                    clipped.append(intersect(prev, curr, edge_fn))
                clipped.append(curr)
            elif prev_in:
                clipped.append(intersect(prev, curr, edge_fn))
        return np.array(clipped, dtype=float)

    def intersect(p1, p2, edge_fn):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        if edge_fn == left:
            t = (0 - x1) / dx if dx != 0 else 0
        elif edge_fn == right:
            t = ((width - 1) - x1) / dx if dx != 0 else 0
        elif edge_fn == top:
            t = (0 - y1) / dy if dy != 0 else 0
        else:  # bottom
            t = ((height - 1) - y1) / dy if dy != 0 else 0
        return np.array([x1 + t * dx, y1 + t * dy], dtype=float)

    def left(p):
        return p[0] >= 0

    def right(p):
        return p[0] <= width - 1

    def top(p):
        return p[1] >= 0

    def bottom(p):
        return p[1] <= height - 1

    clipped = points.astype(float)
    for edge in (left, right, top, bottom):
        clipped = clip_edge(clipped, edge)
        if len(clipped) == 0:
            break
    return clipped


def _obb_inside_ratio(obb, cam):
    width = cam.get("width")
    height = cam.get("height")
    if width is None or height is None:
        return 0.0
    uv, z = project_obb(obb, cam)
    valid = np.isfinite(uv).all(axis=1) & np.isfinite(z) & (z > 0)
    if not np.any(valid):
        return 0.0
    uv = uv[valid]
    hull = external_points_2d(uv)
    total_area = polygon_area(hull)
    if total_area <= 0:
        return 0.0
    clipped = _clip_polygon_to_rect(hull, width, height)
    inside_area = polygon_area(clipped)
    inside_ratio = max(0.0, min(1.0, inside_area / total_area))
    return inside_ratio


def get_visibility_ratio_v3(world_state, obj_id, timestep):
    pixel_threshold = 2000.0
    """
    Calculates visibility based on two parallel criteria:
    1. Is the object geometrically complete? (Rewards small, fully visible objects)
    2. Is the object visually salient? (Rewards large, cropped objects)
    
    Returns the higher of the two scores.
    """
    step = world_state["simulation"][str(timestep)]
    obj_state = step["objects"][str(obj_id)]
    cam = step["camera"]

    if not obj_state or not cam or "obb" not in obj_state:
        return 0.0

    # 1. Raw Data
    fov_visibility = float(obj_state["fov_visibility"])
    pixels_visible = float(obj_state["infov_pixels_visible"])
    pixels_void = float(obj_state["infov_pixels_void"])
    inside_ratio = _obb_inside_ratio(obj_state["obb"], cam)

    # # 2. Gatekeeping
    # if pixels_visible < pixels_void:
    #      raise ImpossibleToAnswer("Uncertainty too high.")

    if pixels_visible < MIN_VISIBLE_PIXELS:
        return 0.0

    # # TODOD # just to check the fucking difference in this, cause they fuck up entire simulations just because there uncertain parts in it...
    if pixels_visible <= 10 and pixels_void >= 400:
        return 0.0
        # raise ImpossibleToAnswer("Uncertainty too high.")

    # --- PATH A: Geometric Completeness ---
    # Good for: Tiny objects that fit fully in the frame.
    # Bad for: Large objects that get cut off by the camera edge.
    score_geom = fov_visibility * inside_ratio

    # --- PATH B: Visual Salience ---
    # Good for: Large objects. If I see 2000px, I don't care if that's only 20% of the object.
    # Bad for: Tiny objects (50px is a low score here).
    score_pixel = min(1.0, pixels_visible / pixel_threshold)

    # 3. The "Or" Gate
    # We take the best of both worlds.
    return max(score_geom, score_pixel)


def get_visibility_ratio_v3_soft(world_state, obj_id, timestep):
    pixel_threshold = 2000.0
    """
    Calculates visibility based on two parallel criteria:
    1. Is the object geometrically complete? (Rewards small, fully visible objects)
    2. Is the object visually salient? (Rewards large, cropped objects)
    
    Returns the higher of the two scores.
    """
    step = world_state["simulation"][str(timestep)]
    obj_state = step["objects"][str(obj_id)]
    cam = step["camera"]

    if not obj_state or not cam or "obb" not in obj_state:
        return 0.0

    # 1. Raw Data
    fov_visibility = float(obj_state["fov_visibility"])
    pixels_visible = float(obj_state["infov_pixels_visible"])
    pixels_void = float(obj_state["infov_pixels_void"])
    inside_ratio = _obb_inside_ratio(obj_state["obb"], cam)

    # # TODOD # just to check the fucking difference in this, cause they fuck up entire simulations just because there uncertain parts in it...
    if pixels_visible <= 10 and pixels_void >= 400:
        raise ImpossibleToAnswer("Uncertainty too high.")

    # --- PATH A: Geometric Completeness ---
    # Good for: Tiny objects that fit fully in the frame.
    # Bad for: Large objects that get cut off by the camera edge.
    score_geom = fov_visibility * inside_ratio

    # --- PATH B: Visual Salience ---
    # Good for: Large objects. If I see 2000px, I don't care if that's only 20% of the object.
    # Bad for: Tiny objects (50px is a low score here).
    score_pixel = min(1.0, pixels_visible / pixel_threshold)

    # 3. The "Or" Gate
    # We take the best of both worlds.
    return max(score_geom, score_pixel)


def is_object_visible(world_state, obj_id, timestep):
    return (
        get_visibility_ratio_v3(world_state, obj_id, timestep) >= VISIBILITY_THRESHOLD
    )


def is_object_visible_soft(world_state, obj_id, timestep):
    return (
        get_visibility_ratio_v3_soft(world_state, obj_id, timestep)
        >= VISIBILITY_THRESHOLD
    )


def get_random_timestep_from_list(visible_timesteps: List[str], question: Any) -> str:
    # MAX_TIMESTEP = len(visible_timesteps) - 1
    MAX_TIMESTEP = min(
        len(visible_timesteps), 30
    )  # usually most things happen before 2 second/50 frames

    if "multi" in question.get("task_splits", ""):
        if len(visible_timesteps) < (CLIP_LENGTH):
            raise ImpossibleToAnswer("No timestep with visible objects.")
        timestep = random.choice(
            visible_timesteps[(CLIP_LENGTH - 1) : MAX_TIMESTEP + 1]
        )
    else:
        if len(visible_timesteps) == 0:
            raise ImpossibleToAnswer("No timestep with visible objects.")
        timestep = random.choice(visible_timesteps)

    return timestep


def extract_attributes(question: Mapping[str, Any]) -> Mapping[str, Any]:
    question_text = question["question"]

    # Extract all tokens enclosed in <...>
    attributes = re.findall(r"<(.*?)>", question_text)

    # Optional: remove duplicates while preserving order
    seen = set()
    attributes = [a for a in attributes if not (a in seen or seen.add(a))]

    return {"attributes": attributes}


def is_object_visible_at_timestep(
    object_id: str, timestep: str, world_state: Mapping[str, Any]
) -> bool:
    """Check if an object is visible at a specific timestep."""

    obj_states = world_state["simulation"][timestep]["objects"]

    pixels_visible = (
        obj_states[object_id]["infov_pixels_visible"]
        + obj_states[object_id]["infov_pixels_void"]
    )
    fov_visibility = obj_states[object_id]["fov_visibility"]

    visible = (
        # Case 1: Object is mostly unoccluded
        fov_visibility >= VISIBILITY_THRESHOLD or pixels_visible >= MIN_VISIBLE_PIXELS
    )

    return visible


def get_object_state_at_timestep(
    world_state: Mapping[str, Any], object_id: str, timestep: str
) -> Optional[Mapping[str, Any]]:
    """Retrieve the state of an object at a specific timestep."""
    simulation_steps = world_state.get("simulation", {})
    if not simulation_steps:
        return None

    step_data = simulation_steps.get(str(timestep), {})
    # print(step_data['objects'][object_id])
    objects = step_data["objects"]
    return objects[object_id]


def get_list_model_of_duplicate_objects(
    world_state, visible_objects_id: List[str]
) -> bool:
    object_names = set()
    duplicate_models = []
    for obj in iter_objects(world_state):
        obj_model = obj.get("model", "")
        if obj_model in object_names:
            duplicate_models.append(obj_model)
        object_names.add(obj_model)
    return duplicate_models


def get_position(
    world_state: Mapping[str, Any], object_id: str, timestep: str
) -> Optional[Mapping[str, Any]]:
    """Retrieve the position of an object at a specific timestep."""
    obj_state = get_object_state_at_timestep(world_state, object_id, timestep)
    if not obj_state:
        return None
    return obj_state["obb"]


def get_visible_timesteps_for_attributes_min_objects(
    attributes: List[Mapping[str, Any]],
    world_state: Mapping[str, Any],
    min_objects=1,
    min_n_frames=8,
    remove_last_n_frames=10,  # this is to avoid that the last frames, where everything is static, are considered
) -> List[str]:
    # I think attributes is not needed I just need to check that more than min_objects with
    # different models are visible at the same time

    visible_timesteps = []

    for timestep in world_state.get("simulation", {}).keys():
        visible_objects_id = []
        for obj in iter_objects(world_state):
            obj_id = obj.get("id")
            if not obj_id:
                continue

            if is_object_visible(world_state, obj_id, timestep):
                visible_objects_id.append(obj_id)

        # we shall check that also is not the same object name to remove for
        list_of_duplicated_objects_models = get_list_model_of_duplicate_objects(
            world_state, visible_objects_id
        )

        # remove duplicate objects by name
        visible_objects_id_and_non_duplicated = [
            obj_id
            for obj_id in visible_objects_id
            if world_state["objects"][obj_id]["model"]
            not in list_of_duplicated_objects_models
        ]

        if len(visible_objects_id_and_non_duplicated) >= min_objects:
            visible_timesteps.append(timestep)

    if len(visible_timesteps) < min_n_frames:
        raise ImpossibleToAnswer(
            f"Not enough timesteps found where the required objects are visible. Found {len(visible_timesteps)}, required at least {min_n_frames}."
        )

    if visible_timesteps == []:
        raise ImpossibleToAnswer(
            "No timesteps found where the required objects are visible."
        )
    if remove_last_n_frames >= len(visible_timesteps):
        raise ImpossibleToAnswer(
            "Not enough timesteps to remove the last frames where everything is static."
        )
    if remove_last_n_frames > 0:
        return visible_timesteps[
            :-remove_last_n_frames
        ]  # remove the last frames where everything is static
    else:
        return visible_timesteps


def get_continuous_subsequences_min_length(
    timesteps: List[str], min_length: int
) -> List[List[str]]:
    if timesteps == []:
        raise ImpossibleToAnswer("No timesteps provided.")

    sorted_timesteps = sorted(int(t.replace(".", "")) for t in timesteps)
    subsequences = []
    current_subseq = [str(timesteps[0])]

    time_interval_in_milliseconds = int(
        (1 / SAMPLING_RATE) * 1000
    )  # e.g., 100ms -> 100*10 = 1000

    # we live a buffer of 1 timestep to allow for small gaps
    for i in range(1, len(sorted_timesteps)):
        if sorted_timesteps[i] == sorted_timesteps[i - 1] + (
            1 * time_interval_in_milliseconds
        ) or sorted_timesteps[i] == sorted_timesteps[i - 1] + (
            2 * time_interval_in_milliseconds
        ):
            current_subseq.append(str(timesteps[i]))
        else:
            if len(current_subseq) >= min_length:
                subsequences.append(current_subseq)
            current_subseq = [str(timesteps[i])]

    if len(current_subseq) >= min_length:
        subsequences.append(current_subseq)

    if subsequences == []:
        raise ImpossibleToAnswer(
            "No continuous subsequences found with the required minimum length."
        )

    return subsequences


def resolve_attributes(
    attributes: List[Mapping[str, Any]], world_state: Mapping[str, Any]
) -> Mapping[str, Any]:
    attribute_resolved = {}
    copy_of_world_state = deepcopy(world_state)

    for attribute in attributes:
        attribute_resolved[attribute] = {}
        attribute_category = attribute.split("_")[
            0
        ]  # Get the part before any underscore
        result = resolver[attribute_category](copy_of_world_state)

        attribute_resolved[attribute]["choice"] = result
        attribute_resolved[attribute]["category"] = attribute_category

    return attribute_resolved


def resolve_attributes_visible_at_timestep(
    attributes: List[Mapping[str, Any]], world_state: Mapping[str, Any], timestep: str
) -> Mapping[str, Any]:
    attribute_resolved = {}

    copy_of_world_state = deepcopy(world_state)

    for attribute in attributes:
        attribute_resolved[attribute] = {}
        attribute_category = attribute.split("_")[
            0
        ]  # Get the part before any underscore
        result = resolver[attribute_category](
            copy_of_world_state, visible_at_timestep=timestep
        )

        attribute_resolved[attribute]["choice"] = result
        attribute_resolved[attribute]["category"] = attribute_category

    return attribute_resolved


def resolve_attributes_most_visible_at_timestep(
    attributes: List[Mapping[str, Any]], world_state: Mapping[str, Any], timestep: str
) -> Mapping[str, Any]:
    assert len(attributes) == 1, "Only one attribute is supported in this function."

    _, visibility_percentage_matrix = get_visibility_mask(
        world_state, max_timestep=timestep
    )

    attribute = attributes[0]
    attribute_resolved = {}

    attribute_resolved[attribute] = {}
    attribute_category = attribute.split("_")[0]

    timestep_index = int(world_state["simulation"][timestep]["frame_idx"])
    visibility_percentage_matrix_at_timestep = visibility_percentage_matrix[
        :, timestep_index
    ]

    most_visible_object_index = np.argmax(visibility_percentage_matrix_at_timestep)
    most_visible_object_id = str(most_visible_object_index + 1)
    chosen_object = world_state["objects"][most_visible_object_id]

    attribute_resolved[attribute]["choice"] = chosen_object
    attribute_resolved[attribute]["category"] = attribute_category

    return attribute_resolved


def fill_template(
    question: Mapping[str, Any], resolved_attributes: Mapping[str, Any]
) -> None:
    for attribute in resolved_attributes:
        if "OBJECT-CATEGORY" in attribute:
            question["question"] = question["question"].replace(
                f"<{attribute}>",
                resolved_attributes[attribute]["choice"],
            )
        elif "OBJECT" in attribute:
            mapped_name = f"\"{gso_mapping[resolved_attributes[attribute]['choice']['model']]['name']}\""
            # mapped_name = resolved_attributes[attribute]["choice"]["name"] --> OLD way
            question["question"] = question["question"].replace(
                f"<{attribute}>", mapped_name
            )
        else:
            question["question"] = question["question"].replace(
                f"<{attribute}>",
                str(resolved_attributes[attribute]["choice"])
                + resolve_units(attribute),
            )

    # check if there is a single frame or multi frame task
    if question["task_splits"] == "multi":
        question["question"] = (
            "Consider all frames, but answer only based on the last frame. "
            + question["question"]
        )


def fill_template_cf(
    question: Mapping[str, Any],
    resolved_attributes: Mapping[str, Any],
    counterfact: str,
) -> None:
    # Adding counterfactual at the end of the question
    question["question"] = (
        counterfact + question["question"][0].lower() + question["question"][1:]
        if question["question"]
        else counterfact
    )

    for attribute in resolved_attributes:
        if "OBJECT-CATEGORY" in attribute:
            question["question"] = question["question"].replace(
                f"<{attribute}>",
                resolved_attributes[attribute]["choice"],
            )
        elif "OBJECT-CF" in attribute:
            mapped_name = gso_mapping[
                resolved_attributes[attribute]["choice"]["model"]
            ]["name"]
            # mapped_name = resolved_attributes[attribute]["choice"]["name"] OLD way
            question["question"] = question["question"].replace(
                f"<{attribute}>", mapped_name
            )
        elif "OBJECT" in attribute:
            mapped_name = gso_mapping[
                resolved_attributes[attribute]["choice"]["model"]
            ]["name"]
            # mapped_name = resolved_attributes[attribute]["choice"]["name"] OLD way
            question["question"] = question["question"].replace(
                f"<{attribute}>", mapped_name
            )
        else:
            question["question"] = question["question"].replace(
                f"<{attribute}>",
                str(resolved_attributes[attribute]["choice"])
                + resolve_units(attribute),
            )

    # check if there is a single frame or multi frame task
    if question["task_splits"] == "multi":
        question["question"] = (
            "Consider all frames, but answer only based on the last frame. "
            + question["question"]
        )


def get_camera(world_state: Mapping[str, Any]) -> Mapping[str, Any]:
    # taking the first camera
    camera = world_state["simulation"]["0000.010"]["camera"]
    if not camera:
        raise ValueError("No camera found in the world state.")
    return camera


def get_camera_at_timestep(
    world_state: Mapping[str, Any], timestep: str
) -> Mapping[str, Any]:
    # taking the first camera
    camera = world_state["simulation"][timestep]["camera"]
    if not camera:
        raise ValueError("No camera found in the world state.")
    return camera


def get_random_material(world_state: Mapping[str, Any]) -> str:
    materials = set()
    for obj in iter_objects(world_state):
        material = as_lower(obj["description"]["material_group"])
        if material:
            materials.add(material)
    if not materials:
        raise ValueError("No materials found in the world state.")
    return random.choice(list(materials))


def get_random_object_and_remove(
    world_state: Mapping[str, Any],
    OBJECT_CATEGORY: Optional[str] = None,
    visible_at_timestep: str = None,
) -> Mapping[str, Any]:
    objects = world_state["objects"]
    if visible_at_timestep is not None:
        visible_objects = []
        visible_objects_ids = []
        for obj_id, object in objects.items():
            # better visibility check
            if is_object_visible(world_state, obj_id, visible_at_timestep):
                obj_copy = object.copy()
                obj_copy["id"] = obj_id
                visible_objects.append(obj_copy)
                visible_objects_ids.append(obj_id)

        list_of_duplicate_object_models = get_list_model_of_duplicate_objects(
            world_state, visible_objects_ids
        )
        # remove duplicate objects by name
        visible_objects = [
            obj
            for obj in visible_objects
            if obj["model"] not in list_of_duplicate_object_models
        ]

        objects = {obj["id"]: obj for obj in visible_objects}

    # also if no visible objects found, we raise an error
    if not objects:
        raise ImpossibleToAnswer(f"No objects found of type '{OBJECT_CATEGORY}'")

    object_chosen = random.choice(list(objects.values()))

    del world_state["objects"][object_chosen["id"]]

    return object_chosen


def get_random_most_visible_object_and_remove(
    world_state: Mapping[str, Any],
    OBJECT_CATEGORY: Optional[str] = None,
    visible_at_timestep: str = None,
) -> Mapping[str, Any]:
    objects = world_state["objects"]
    if visible_at_timestep is not None:
        visible_objects = []
        visible_objects_ids = []
        for obj_id, object in objects.items():
            # better visibility check
            if is_object_visible(world_state, obj_id, visible_at_timestep):
                obj_copy = object.copy()
                obj_copy["id"] = obj_id
                obj_copy["visibility_ratio"] = get_visibility_ratio_v3(
                    world_state, obj_id, visible_at_timestep
                )
                visible_objects.append(obj_copy)
                visible_objects_ids.append(obj_id)

        list_of_duplicate_object_models = get_list_model_of_duplicate_objects(
            world_state, visible_objects_ids
        )
        # remove duplicate objects by name
        visible_objects = [
            obj
            for obj in visible_objects
            if obj["model"] not in list_of_duplicate_object_models
        ]

        objects = {obj["id"]: obj for obj in visible_objects}

    # also if no visible objects found, we raise an error
    if not objects:
        raise ImpossibleToAnswer(f"No objects found of type '{OBJECT_CATEGORY}'")

    # filter objects to only those that have visibility ratio above threshold
    filtered_objects = {
        obj_id: obj for obj_id, obj in objects.items() if obj["visibility_ratio"] >= 0.9
    }
    if not filtered_objects:
        raise ImpossibleToAnswer(
            f"No objects found of type '{OBJECT_CATEGORY}' with sufficient visibility."
        )

    object_chosen = random.choice(list(filtered_objects.values()))

    del world_state["objects"][object_chosen["id"]]

    return object_chosen


def get_first_object_and_remove(
    world_state: Mapping[str, Any],
    OBJECT_CATEGORY: Optional[str] = None,
    visible_at_timestep: str = None,
) -> Mapping[str, Any]:
    objects = world_state["objects"]
    if visible_at_timestep is not None:
        visible_objects = []
        visible_objects_ids = []
        obj_id = "1"
        object = objects[obj_id]
        obj_state = get_object_state_at_timestep(
            world_state, obj_id, visible_at_timestep
        )
        if (
            obj_state["fov_visibility"] > VISIBILITY_THRESHOLD
            and obj_state["infov_pixels"] >= MIN_VISIBLE_PIXELS
        ):
            obj_copy = object.copy()
            obj_copy["id"] = obj_id
            visible_objects.append(obj_copy)
            visible_objects_ids.append(obj_id)

        list_of_duplicate_object_models = get_list_model_of_duplicate_objects(
            world_state, visible_objects_ids
        )
        # remove duplicate objects by name
        visible_objects = [
            obj
            for obj in visible_objects
            if obj["model"] not in list_of_duplicate_object_models
        ]

        objects = {obj["id"]: obj for obj in visible_objects}

    # also if no visible objects found, we raise an error
    if not objects:
        raise ImpossibleToAnswer(f"No objects found of type '{OBJECT_CATEGORY}'")

    object_chosen = list(objects.values())[0]

    del world_state["objects"][object_chosen["id"]]

    return object_chosen


def get_random_OBJECT_CATEGORY(world_state: Mapping[str, Any]) -> str:
    OBJECT_CATEGORYs = set()
    for obj in iter_objects(world_state):
        obj_type = as_lower(obj["description"]["category_gso"])
        if obj_type:
            OBJECT_CATEGORYs.add(obj_type)
    if not OBJECT_CATEGORYs:
        raise ValueError("No object types found in the world state.")
    return random.choice(list(OBJECT_CATEGORYs))


# TODO Those random values are just hardcoded
resolver = {
    "CAMERA": get_camera,
    "CATEGORY": get_random_OBJECT_CATEGORY,
    "DENSITY": lambda: round(
        random.uniform(10, 600), 1
    ),  # random density between 10 and 600 kg/m3
    "DISTANCE": lambda: round(
        random.uniform(1.0, 5.0), 1
    ),  # random distance between 1 and 5 meters, 1 decimal place
    "MASS": lambda: round(
        random.uniform(0.1, 3.0), 1
    ),  # random mass between 0.1 and 5 kg
    "MATERIAL": get_random_material,
    "OBJECT-CATEGORY": get_random_OBJECT_CATEGORY,
    "OBJECT-RANDOM": get_random_object_and_remove,
    "OBJECT": get_random_most_visible_object_and_remove,
    "OBJECT-CF": get_first_object_and_remove,
    "STRESS-THRESHOLD": lambda: round(
        random.uniform(0.0, 10.0), 1
    ),  # random stress threshold between 10 and 100 MPa
    "VOLUME": lambda: round(
        random.uniform(0.001, 0.5), 1
    ),  # random volume between 0.001 and .5 cubic meters
}


def round_sig(x: float, sig: int = 3) -> float:
    """Round to `sig` significant digits, preserving sign."""
    if x == 0:
        return 0.0
    return round(x, sig - 1 - int(math.floor(math.log10(abs(x)))))


def decimals_for_sig(x: float, sig: int = 3) -> int:
    """Number of decimal places that keeps `sig` significant digits when using round(x, decimals)."""
    if x == 0:
        # e.g., for sig=3, show 2 decimals by default
        return sig - 1
    return max(0, sig - 1 - int(math.floor(math.log10(abs(x)))))


def iter_objects(world_state: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    objects = world_state.get("objects", [])
    if isinstance(objects, Mapping):
        iterable: Iterable[Any] = objects.values()
    else:
        iterable = cast(Iterable[Any], objects)

    for obj in iterable:
        if isinstance(obj, Mapping):
            yield obj


def get_motion(obj: Mapping[str, Any]) -> Mapping[str, Any]:
    motion = obj.get("motion")
    if isinstance(motion, Mapping):
        return motion
    return {}


def coerce_to_float(value: Any) -> Optional[float]:
    if value is None:
        return None

    if isinstance(value, bool):
        return float(value)

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, Mapping):
        for key in ("magnitude", "value", "amount", "norm", "length"):
            if key in value:
                coerced = coerce_to_float(value[key])
                if coerced is not None:
                    return coerced

        if all(isinstance(v, (int, float)) for v in value.values()):
            numeric_values = tuple(cast(Number, v) for v in value.values())
            return vector_magnitude(numeric_values)

        return None

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if not value:
            return None

        if all(isinstance(item, (int, float)) for item in value):
            numeric_values = tuple(cast(Number, item) for item in value)
            return vector_magnitude(numeric_values)

    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None

    return None


def vector_magnitude(components: Tuple[Number, ...]) -> float:
    if not components:
        return 0.0
    return math.sqrt(sum(float(component) ** 2 for component in components))


def as_vector(value: Any) -> Optional[Tuple[Number, ...]]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if all(isinstance(item, (int, float)) for item in value):
            return tuple(cast(Number, item) for item in value)
    if isinstance(value, Mapping):
        numeric_items = [v for v in value.values() if isinstance(v, (int, float))]
        if numeric_items:
            return tuple(cast(Number, v) for v in numeric_items)
    return None


def as_lower(value: Any) -> Optional[str]:
    if isinstance(value, str):
        candidate = value.strip()
        if candidate:
            return candidate.casefold()
    return None


def minimum_distance_between_OBBs(
    obb1: Mapping[str, Any], obb2: Mapping[str, Any]
) -> float:
    min_distance = -np.inf
    eps = 1e-6

    center_1, extents_1, R1 = obb1["center"], np.array(obb1["extents"]) / 2.0, obb1["R"]
    center_2, extents_2, R2 = obb2["center"], np.array(obb2["extents"]) / 2.0, obb2["R"]

    axes_1 = np.array(R1)
    axes_2 = np.array(R2)

    distance_centers = np.array(center_2) - np.array(center_1)

    R = axes_1 @ axes_2.T
    abs_R = np.abs(R) + eps

    for axis_i in range(3):
        axis = axes_1[axis_i]
        distance_centers_projected = abs(np.dot(distance_centers, axis))
        radius_1 = extents_1[axis_i]
        radius_2 = np.dot(extents_2, abs_R[axis_i])

        gap = distance_centers_projected - (radius_1 + radius_2)
        min_distance = max(gap, min_distance)

    for axis_j in range(3):
        axis = axes_2[axis_j]
        distance_centers_projected = abs(np.dot(distance_centers, axis))
        radius_1 = np.dot(extents_1, abs_R[:, axis_j])
        radius_2 = extents_2[axis_j]

        gap = distance_centers_projected - (radius_1 + radius_2)
        min_distance = max(gap, min_distance)

    for i in range(3):
        for j in range(3):
            axis = np.cross(axes_1[i], axes_2[j])
            axis_norm = np.linalg.norm(axis)
            if axis_norm < eps:
                continue

            axis = axis / axis_norm

            center_sep = abs(np.dot(distance_centers, axis))

            radius_1 = (
                extents_1[(i + 1) % 3] * abs_R[(i + 2) % 3, j]
                + extents_1[(i + 2) % 3] * abs_R[(i + 1) % 3, j]
            )
            radius_2 = (
                extents_2[(j + 2) % 3] * abs_R[i, (j + 1) % 3]
                + extents_2[(j + 1) % 3] * abs_R[i, (j + 2) % 3]
            )

            gap = center_sep - (radius_1 + radius_2)
            min_distance = max(min_distance, gap)

    return max(min_distance, 0.0)


def distance_between(
    first: Optional[Sequence[Number]],
    second: Optional[Sequence[Number]],
) -> float:
    if not first or not second:
        return 0.0
    a = ensure_vector_size(tuple(cast(Number, component) for component in first))
    b = ensure_vector_size(tuple(cast(Number, component) for component in second))
    return math.sqrt(
        sum((a_component - b_component) ** 2 for a_component, b_component in zip(a, b))
    )


def ensure_vector_size(
    components: Tuple[Number, ...], size: int = 3
) -> Tuple[float, ...]:
    """Normalise a vector to the desired size by truncating or padding with zeros."""
    floats = tuple(float(component) for component in components)
    if len(floats) >= size:
        return floats[:size]
    if not floats:
        return tuple(0.0 for _ in range(size))
    padded = list(floats)
    while len(padded) < size:
        padded.append(0.0)
    return tuple(padded)
