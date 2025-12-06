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


sampling_rate = get_config()["sampling_rate"]
time_interval = 1.0 / sampling_rate

scene_description_path = "./utils/json/scene_context.json"
scene_description_cache = {}
with open(scene_description_path, "r") as f:
    scene_description_cache = json.load(f)

def get_counterfactual_image_paths(file_names):

    patterns = ["shift-x", "shift-z", "low-gravity", "2xsmaller"]
    pattern_re = re.compile(
        r"/(" + "|".join(map(re.escape, patterns)) + r")(?=/|$)"
    )

    counterfactual_image_paths = []

    for file in file_names:
        new_file_name = pattern_re.sub("", file).replace("//", "/")
        simulation_id_path_og = new_file_name.replace("dl3dv-counterfact", "dl3dv")
         
        # I need to check the folder that contains the original simulation
        base_dir = simulation_id_path_og.split("seed-")[0]
        seed = simulation_id_path_og.split("seed-")[1].split("_")[0]

        matches = glob.glob(base_dir + "seed-" + seed + "_*")
        if len(matches) == 0:
            raise FileNotFoundError(f"Original simulation folder not found for {simulation_id_path_og}")
        
        file_name = simulation_id_path_og.split("/")[-1]
        counterfactual_image_paths.append(matches[0] + "/render/" + file_name)

    return counterfactual_image_paths
    
    # Implement logic to get counterfactual image paths based on the question

def augment_image_VQA_with_context(
    question, world_state, resolved_attributes, file_names, augmentation=None
):
    # Implement the augmentation logic here
    # print("Augmenting VQA with context...")
    # print(f"Augmentation: {augmentation}")

    if augmentation is None:
        return file_names

    # let's route here based on the flags
    if augmentation == "roi_circling":
        file_names = augment_roi_circling(
            question, world_state, resolved_attributes, file_names
        )
    if augmentation == "contour":
        file_names = augment_contour(
            question, world_state, resolved_attributes, file_names
        )
    if augmentation == "textual_context":
        file_names = augment_textual_context(
            question, world_state, resolved_attributes, file_names
        )
    if augmentation == "scene_context":
        file_names = augment_scene_context(
            question, world_state, resolved_attributes, file_names
        )

    return file_names


def augment_roi_circling(question, world_state, resolved_attributes, file_names):
    # just check for folder existance
    new_dir = (
        Path(file_names[0]).parent.as_posix().replace("render", "render_roi_circled")
    )

    if os.path.exists(new_dir) is False:
        os.makedirs(new_dir, exist_ok=True)

    if resolved_attributes == {}:
        return file_names

    for file in file_names:
        original_image = np.array(PIL.Image.open(file))

        for idx, (resolved_attr, value) in enumerate(resolved_attributes.items()):
            if "OBJECT" in resolved_attr:
                object_id = value["choice"]["id"]
                render_name = file.split("/")[-1]
                instance_image_path = file.replace("render", "instances")

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
                    # print("PROBABLY THE OBJECT IS NOT VISIBLE ENOUGH")
                    # print(f"question: {question['question']}")
                    # print(f"labels: {file_names}")
                    # print(f"{file}, timestep: {render_name}")
                    # print("No contours found for ROI circling!")
                    print(
                        "This question just has to be like that, ther is nothing in the image and the VLM has to "
                        "understand that."
                    )
                    continue

                pts = np.vstack([c.reshape(-1, 2) for c in contours]).astype(np.float32)
                center, radius = cv2.minEnclosingCircle(pts)

                # Draw the bounding box on the image
                original_image = draw_roi_circle(
                    original_image, center, radius * 1.5, idx
                )

        new_file_name = file.replace("render", "render_roi_circled").replace(
            ".png", f"_{question['_question_key']}.png"
        )
        original_image = PIL.Image.fromarray(original_image)
        original_image.save(new_file_name)

        file_names[file_names.index(file)] = new_file_name
    if len(resolved_attributes) > 0:
        new_question = f"In the image, the region circled in red indicates the area of interest. {question['question']}"

        question["question"] = new_question

    return file_names


def draw_roi_circle(original_image, center, radius=10, idx=0):
    # letters = ['A', 'B', 'C', 'D', 'E']
    # letter = letters[idx % len(letters)]

    cx, cy = map(int, center)

    # draw main circle
    cv2.circle(original_image, (cx, cy), int(radius), (255, 0, 0), 5, cv2.LINE_AA)

    # # box parameters proportional to circle
    # box_size = 40
    # offset_y = int(radius * 1.6)

    # # box position above circle
    # bx1, by1 = cx - box_size // 2, cy - offset_y - box_size // 2
    # bx2, by2 = cx + box_size // 2, cy - offset_y + box_size // 2

    # # black filled box with blue border
    # cv2.rectangle(original_image, (bx1, by1), (bx2, by2), (0, 0, 0), -1)
    # cv2.rectangle(original_image, (bx1, by1), (bx2, by2), (255, 0, 0), 3)

    # # text parameters adjusted to box size
    # font_scale = box_size / 60
    # text_size = cv2.getTextSize(letter, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 3)[0]
    # text_x = cx - text_size[0] // 2
    # text_y = cy - offset_y + text_size[1] // 2

    # cv2.putText(original_image, letter, (text_x, text_y),
    #             cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 3, cv2.LINE_AA)

    return original_image




def augment_contour(question, world_state, resolved_attributes, file_names):
    # just check for folder existance
    new_dir = Path(file_names[0]).parent.as_posix().replace("render", "render_masked")
    # print(f"Creating masked images in {new_dir}")

    if os.path.exists(new_dir) is False:
        os.makedirs(new_dir, exist_ok=True)

    if resolved_attributes == {}:
        return file_names

    for file in file_names:
        original_image = np.array(PIL.Image.open(file))
        for resolved_attr, value in resolved_attributes.items():
            if "OBJECT" in resolved_attr:
                object_id = value["choice"]["id"]

                instance_image_path = file.replace("render", "instances")

                # rgb_object_class = world_state['encoding']['classes'][int(object_id) +1]
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
                # Apply the mask to the original image (set background to black)
                original_image[binary_mask] = [255, 0, 0]

        new_file_name = file.replace("render", "render_masked").replace(
            ".png", f"_{question['_question_key']}.png"
        )
        original_image = PIL.Image.fromarray(original_image)
        original_image.save(new_file_name)

        file_names[file_names.index(file)] = new_file_name

    return file_names


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


def augment_textual_context(question, world_state, resolved_attributes, file_names):
    file = file_names[0]  # assuming single image for textual context

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
    zones_to_focus = []

    object_names = []

    for resolved_attr, value in resolved_attributes.items():
        if "OBJECT" in resolved_attr:
            render_name = file.split("/")[-1]
            timestep = (
                str(((int(render_name.split(".")[0]) * 4) + 1) / 100).zfill(7) + "0"
            )

            object_id = value["choice"]["id"]
            object_names.append(value["choice"]["name"])
            object_at_timestep = world_state["simulation"][str(timestep)]["objects"][
                object_id
            ]

            obb = object_at_timestep["obb"]
            cam = world_state["simulation"][str(timestep)]["camera"]
            # Here we would add the circling logic around the object

            uv, _ = project_obb(obb, cam)
            # Draw the bounding box on the image
            (center_x, center_y), _ = cv2.minEnclosingCircle(uv.astype("float32"))

            # this should not happen but it does edge cases, where the object is slightly visible
            # so we just put centerx and centery a the border
            if (
                center_x < 0
                or center_x > img_width
                or center_y < 0
                or center_y > img_height
            ):
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
                zones_to_focus.append(possible_zones[zone_index])
            except IndexError:
                print(
                    f"IndexError: zone_index {zone_index} out of range for possible_zones."
                )
                print(
                    f"center_x: {center_x}, center_y: {center_y}, row: {row}, col: {col}"
                )
                raise IndexError

    # Now, we can augment the question with the zones to focus on
    if zones_to_focus:
        focus_text = (
            "To answer the following question, focus on the "
            + " and the ".join(zones_to_focus)
            + " of the image and put special attention to the "
            + " and the ".join(object_names)
            + "."
        )
        question["question"] = focus_text + " " + question["question"]

    return file_names
