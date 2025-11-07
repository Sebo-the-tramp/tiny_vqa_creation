
import os
import PIL
import cv2
import json
import numpy as np
from utils.config import get_config
from utils.geometry import (
    project_obb,
)
from pathlib import Path

sampling_rate = get_config()["sampling_rate"]
time_interval = 1.0 / sampling_rate

scene_description_path = "./utils/json/scene_context.json"
scene_description_cache = {}
with open(scene_description_path, 'r') as f:
    scene_description_cache = json.load(f)


def augment_image_VQA_with_context( 
    question,
    world_state,
    resolved_attributes,
    file_names,
    augmentation=None
):
    # Implement the augmentation logic here    
    # print("Augmenting VQA with context...")
    # print(f"Augmentation: {augmentation}")

    if augmentation is None:
        return file_names

    # let's route here based on the flags
    if augmentation == "roi_circling":
        file_names = augment_roi_circling(
            question,
            world_state,
            resolved_attributes,
            file_names
        )
    if augmentation == "contour":
        file_names = augment_contour(
            question,
            world_state,
            resolved_attributes,
            file_names
        )
    if augmentation == "textual_context":
        file_names = augment_textual_context(
            question,
            world_state,
            resolved_attributes,
            file_names
        )
    if augmentation == "scene_context":
        file_names = augment_scene_context(
            question,
            world_state,
            resolved_attributes,
            file_names
        )

    return file_names


def augment_roi_circling(question,
    world_state,
    resolved_attributes,
    file_names):
    
    # just check for folder existance
    new_dir = Path(file_names[0]).parent.as_posix().replace("render", "render_roi_circled")

    if os.path.exists(new_dir) is False:
        os.makedirs(new_dir, exist_ok=True)

    if resolved_attributes == {}:
        return file_names

    for file in file_names:
        original_image = np.array(PIL.Image.open(file))        

        for resolved_attr, value in resolved_attributes.items():
            if "OBJECT" in resolved_attr:
                render_name = file.split("/")[-1]
                timestep = str(((int(render_name.split(".")[0]) * 4) + 1) / 100).zfill(7) + "0"

                object_id = value['choice']['id']
                object_at_timestep = world_state["simulation"][str(timestep)]["objects"][object_id]

                obb = object_at_timestep["obb"]
                cam = world_state["simulation"][str(timestep)]["camera"]
                # Here we would add the circling logic around the object

                uv, _ = project_obb(obb, cam)
                # Draw the bounding box on the image
                original_image = draw_roi_circle(original_image, uv)
                                    

        # Save the augmented image back
        # later on some caching mechanism can be added

        new_file_name = file.replace("render", "render_roi_circled").replace(".png", f"_{question['_question_key']}.png")
        original_image = PIL.Image.fromarray(original_image)
        original_image.save(new_file_name)

        file_names[file_names.index(file)] = new_file_name
    
    return file_names

def draw_roi_circle(original_image, uv):
    (center_x, center_y), radius = cv2.minEnclosingCircle(uv.astype('float32'))
    center = (float(center_x), float(center_y))
    radius = float(radius)
    cv2.circle(original_image, (int(center_x), int(center_y)), int(radius), (255,0,0), 2)
    return original_image


def augment_masking(question,
    world_state,
    resolved_attributes,
    file_names):
    
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
                object_id = value['choice']['id']        

                mask_image_path = file.replace("render", f"semantics/obj_{object_id}")
                mask_image = np.array(PIL.Image.open(mask_image_path).convert("L"))

                # Create a binary mask where the object is located
                binary_mask = mask_image > 0
                # Apply the mask to the original image (set background to black)
                original_image[~binary_mask] = 0

        new_file_name = file.replace("render", "render_masked").replace(".png", f"_{question['_question_key']}.png")
        original_image = PIL.Image.fromarray(original_image)
        # print(f"Saving masked image to {new_file_name}")
        original_image.save(new_file_name)

        file_names[file_names.index(file)] = new_file_name
    
    return file_names

def augment_scene_context(question,
    world_state,
    resolved_attributes,
    file_names):

    scene_id = world_state["scene"]['scene']
    if scene_id in scene_description_cache:
        scene_description = scene_description_cache[scene_id]
        context_text = scene_description["scene_context_short"]
        if context_text:
            question["question"] = f"Scene Context: {context_text} " + question["question"]

    return file_names

def augment_textual_context(question,
    world_state,
    resolved_attributes,
    file_names):

    file = file_names[0]  # assuming single image for textual context

    possible_zones = [
        'top-left', 'top-center', 'top-right',
        'middle-left', 'center', 'middle-right',
        'bottom-left', 'bottom-center', 'bottom-right'
    ]
    img_width, img_height = 1000, 562 # assuming fixed image size for now
    zones_to_focus = []

    for resolved_attr, value in resolved_attributes.items():
        if "OBJECT" in resolved_attr:
            render_name = file.split("/")[-1]
            timestep = str(((int(render_name.split(".")[0]) * 4) + 1) / 100).zfill(7) + "0"

            object_id = value['choice']['id']
            object_at_timestep = world_state["simulation"][str(timestep)]["objects"][object_id]

            obb = object_at_timestep["obb"]
            cam = world_state["simulation"][str(timestep)]["camera"]
            # Here we would add the circling logic around the object

            uv, _ = project_obb(obb, cam)
            # Draw the bounding box on the image
            (center_x, center_y), radius = cv2.minEnclosingCircle(uv.astype('float32'))
            center = (float(center_x), float(center_y))

            # this should not happen but it does edge cases, where the object is slightly visible
            # so we just put centerx and centery a the border
            if center_x < 0 or center_x > img_width or center_y < 0 or center_y > img_height:
                center_x = min(max(center_x, 0), img_width)
                center_y = min(max(center_y, 0), img_height)                            

            # Determine which zone the center falls into
            # Define zone boundaries
            zone_width = img_width / 3
            zone_height = img_height / 3

            col = int(center_x // zone_width)
            row = int(center_y // zone_height)

            zone_index = row * 3 + col
            zones_to_focus.append(possible_zones[zone_index])

    # Now, we can augment the question with the zones to focus on
    if zones_to_focus:
        focus_text = "To answer the following question, focus on the following areas of the image: " + ", ".join(zones_to_focus) + "."
        question["question"] = focus_text + " " + question["question"]       

    return file_names