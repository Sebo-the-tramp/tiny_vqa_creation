import re
import os
from pathlib import Path
import cv2
import numpy as np
import PIL
import json
import glob
from utils.config import get_config
from utils.geometry import project_obb

from utils.my_exception import ImpossibleToAnswer


sampling_rate = get_config()["sampling_rate"]
time_interval = 1.0 / sampling_rate

scene_description_path = "./utils/json/scene_context.json"
scene_description_cache = {}
with open(scene_description_path, "r") as f:
    scene_description_cache = json.load(f)


def get_counterfactual_image_paths(file_names):
    patterns = ["shift-x", "shift-z", "low-gravity", "2xsmaller"]
    pattern_re = re.compile(r"/(" + "|".join(map(re.escape, patterns)) + r")(?=/|$)")

    counterfactual_image_paths = []

    for file in file_names:
        new_file_name = pattern_re.sub("", file).replace("//", "/")
        simulation_id_path_og = new_file_name.replace("dl3dv-counterfact", "dl3dv")

        # I need to check the folder that contains the original simulation
        base_dir = simulation_id_path_og.split("seed-")[0]
        seed = simulation_id_path_og.split("seed-")[1].split("_")[0]

        matches = glob.glob(base_dir + "seed-" + seed + "_*")
        if len(matches) == 0:
            raise FileNotFoundError(
                f"Original simulation folder not found for {simulation_id_path_og}"
            )

        file_name = simulation_id_path_og.split("/")[-1]
        counterfactual_image_paths.append(matches[0] + "/render/" + file_name)

    return counterfactual_image_paths

    # Implement logic to get counterfactual image paths based on the question


def augment_image_VQA_with_context(
    question, world_state, resolved_attributes, file_names, augmentation=None
):
    if augmentation is None:
        return file_names

    # let's route here based on the flags
    if augmentation == "roi_circling_text":
        file_names = augment_roi_circling(
            question,
            world_state,
            resolved_attributes,
            file_names,
            text=True,
            layout_position=False,
        )
    if augmentation == "roi_circling_no_text":
        file_names = augment_roi_circling(
            question,
            world_state,
            resolved_attributes,
            file_names,
            text=False,
            layout_position=False,
        )
    if augmentation == "roi_circling_text_layout_position":
        file_names = augment_roi_circling(
            question,
            world_state,
            resolved_attributes,
            file_names,
            text=True,
            layout_position=True,
        )
    if augmentation == "roi_circling_no_text_layout_position":
        file_names = augment_roi_circling(
            question,
            world_state,
            resolved_attributes,
            file_names,
            text=False,
            layout_position=True,
        )

    # this is just to add same pathwas but with NO AUGMENTATIONs
    if augmentation == "ablation":
        file_names = augment_ablation(
            question,
            world_state,
            resolved_attributes,
            file_names,
            text=False,
            layout_position=False,
        )

    # other augmentations
    if augmentation == "grounding_physics":
        file_names = augment_scene_context(
            question, world_state, resolved_attributes, file_names
        )

    return file_names


def augment_roi_circling(
    question,
    world_state,
    resolved_attributes,
    file_names,
    text=True,
    layout_position=False,
):
    # just check for folder existance
    new_dir = (
        Path(file_names[0]).parent.as_posix().replace("render", "render_roi_circled")
    )

    if os.path.exists(new_dir) is False:
        os.makedirs(new_dir, exist_ok=True)

    if resolved_attributes == {}:
        return file_names

    if len(resolved_attributes) == 0:
        raise ImpossibleToAnswer(
            "No resolved attributes for ROI circling. So no need to circle anything, and to ask questions about it"
        )

    for file in file_names:
        original_image = np.array(PIL.Image.open(file))

        for idx, (resolved_attr, value) in enumerate(resolved_attributes.items()):
            if "OBJECT" in resolved_attr:
                object_id = value["choice"]["id"]
                render_name = file.split("/")[-1]
                instance_image_path = file.replace("render", "instances")

                # print(world_state["encoding"])
                # rgb_object_class = world_state["encoding"]["semantic_classes"][
                rgb_object_class = world_state["encoding"]["classes"][
                    int(object_id) + 1
                ]

                visible_object_mask = (
                    np.array(PIL.Image.open(instance_image_path).convert("RGB"))
                    == rgb_object_class
                )
                visible_object_mask = np.all(visible_object_mask, axis=-1)

                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                eroded = cv2.erode(
                    visible_object_mask.astype(np.uint8), kernel, iterations=1
                )
                inner_border_mask = visible_object_mask & ~eroded

                # Create a binary mask where the object is located
                binary_mask = inner_border_mask > 0

                contours, _ = cv2.findContours(
                    binary_mask.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )

                if len(contours) == 0:
                    raise ImpossibleToAnswer("No visible object found.")

                pts = np.vstack([c.reshape(-1, 2) for c in contours]).astype(np.float32)
                center, radius = cv2.minEnclosingCircle(pts)

                # Draw the bounding box on the image
                original_image = draw_roi_circle(
                    original_image, center, radius * 1.5, idx
                )

                object_name = value["choice"]["name"]
                pattern = re.compile(re.escape(object_name), re.IGNORECASE)
                if text:
                    # modify the question such that the name of the object is removed and replaced with "the circled object"
                    if layout_position:
                        zone_to_focus = get_object_zone(
                            world_state, object_id, int(render_name.replace(".png", ""))
                        )
                        new_question = pattern.sub(
                            f"{object_name} (circled in red and located at the {zone_to_focus})",
                            question["question"],
                        )
                    else:
                        new_question = pattern.sub(
                            f"{object_name} (circled in red)", question["question"]
                        )
                else:
                    if layout_position:
                        # append after the name of the object that it is circled in the image
                        zone_to_focus = get_object_zone(
                            world_state, object_id, int(render_name.replace(".png", ""))
                        )
                        new_question = pattern.sub(
                            f"object circled in red (located at the {zone_to_focus})",
                            question["question"],
                        )
                    else:
                        # append after the name of the object that it is circled in the image
                        new_question = pattern.sub(
                            "object circled in red", question["question"]
                        )

        new_file_name = file.replace("render", "render_roi_circled").replace(
            ".png", f"_{question['_question_key']}.png"
        )
        original_image = PIL.Image.fromarray(original_image)
        original_image.save(new_file_name)

        file_names[file_names.index(file)] = new_file_name

    # if new_question is None:
    #     raise ImpossibleToAnswer("No modifications done to the question in ROI circling augmentation.")

    if len(resolved_attributes) > 0:
        question["question"] = new_question

    return file_names


def augment_ablation(
    question,
    world_state,
    resolved_attributes,
    file_names,
    text=True,
    layout_position=False,
):
    # just check for folder existance
    new_dir = (
        Path(file_names[0]).parent.as_posix().replace("render", "render_roi_circled")
    )

    if os.path.exists(new_dir) is False:
        os.makedirs(new_dir, exist_ok=True)

    if resolved_attributes == {}:
        return file_names

    if len(resolved_attributes) == 0:
        raise ImpossibleToAnswer(
            "No resolved attributes for ROI circling. So no need to circle anything, and to ask questions about it"
        )

    for file in file_names:
        original_image = np.array(PIL.Image.open(file))

        for idx, (resolved_attr, value) in enumerate(resolved_attributes.items()):
            if "OBJECT" in resolved_attr:
                object_id = value["choice"]["id"]
                instance_image_path = file.replace("render", "instances")

                # print(world_state["encoding"])
                # rgb_object_class = world_state["encoding"]["semantic_classes"][
                rgb_object_class = world_state["encoding"]["classes"][
                    int(object_id) + 1
                ]

                visible_object_mask = (
                    np.array(PIL.Image.open(instance_image_path).convert("RGB"))
                    == rgb_object_class
                )
                visible_object_mask = np.all(visible_object_mask, axis=-1)

                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                eroded = cv2.erode(
                    visible_object_mask.astype(np.uint8), kernel, iterations=1
                )
                inner_border_mask = visible_object_mask & ~eroded

                # Create a binary mask where the object is located
                binary_mask = inner_border_mask > 0

                contours, _ = cv2.findContours(
                    binary_mask.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )

                if len(contours) == 0:
                    raise ImpossibleToAnswer("No visible object found.")

                pts = np.vstack([c.reshape(-1, 2) for c in contours]).astype(np.float32)
                center, radius = cv2.minEnclosingCircle(pts)

                # Draw the bounding box on the image
                original_image = draw_roi_circle(
                    original_image, center, radius * 1.5, idx
                )

    return file_names


def draw_roi_circle(original_image, center, radius=10, idx=0):
    cx, cy = map(int, center)
    cv2.circle(original_image, (cx, cy), int(radius), (255, 0, 0), 5, cv2.LINE_AA)

    return original_image


def augment_scene_context(question, world_state, resolved_attributes, file_names):
    scene_id = world_state["scene"]["scene"]
    if scene_id in scene_description_cache:
        scene_description = scene_description_cache[scene_id]
        context_text = scene_description["scene_context_short"]
        if context_text:
            question["question"] = (
                f"Scene Context: {context_text} " + question["question"]
            )

    return file_names


def get_object_zone(world_state, object_id, timestep_index):
    timestep = list(world_state["simulation"].keys())[timestep_index]

    possible_zones = [
        "top-left",
        "top-center",
        "top-right",
        "middle-left",
        "center",
        "middle-right",
        "bottom-left",
        "bottom-center",
        "bottom-right",
    ]
    img_width, img_height = 1000, 562  # assuming fixed image size for now
    zone_to_focus = ""

    object_at_timestep = world_state["simulation"][timestep]["objects"][object_id]

    obb = object_at_timestep["obb"]
    cam = world_state["simulation"][str(timestep)]["camera"]
    # Here we would add the circling logic around the object

    uv, _ = project_obb(obb, cam)
    # Draw the bounding box on the image
    (center_x, center_y), _ = cv2.minEnclosingCircle(uv.astype("float32"))

    # this should not happen but it does edge cases, where the object is slightly visible
    # so we just put centerx and centery a the border
    if center_x < 0 or center_x > img_width or center_y < 0 or center_y > img_height:
        center_x = min(max(center_x, 0), img_width - (10))
        center_y = min(max(center_y, 0), img_height - (10))

    # Determine which zone the center falls into
    # Define zone boundaries
    zone_width = img_width / 3
    zone_height = img_height / 3

    col = int(center_x // zone_width)
    row = int(center_y // zone_height)

    zone_index = row * 3 + col
    try:
        zone_to_focus = possible_zones[zone_index]
    except IndexError:
        print(f"IndexError: zone_index {zone_index} out of range for possible_zones.")
        print(f"center_x: {center_x}, center_y: {center_y}, row: {row}, col: {col}")
        raise IndexError

    return zone_to_focus
