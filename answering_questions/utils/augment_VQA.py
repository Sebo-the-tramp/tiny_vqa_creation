import re
from pathlib import Path
import cv2
import numpy as np
import PIL.Image as Image
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


# Canonical augmentation names
AUG_ROI_CIRCLING_TEXT = "roi_circling_text"
AUG_ROI_CIRCLING_NO_TEXT = "roi_circling_no_text"
AUG_ROI_CIRCLING_TEXT_LAYOUT = "roi_circling_text_layout_position"
AUG_ROI_CIRCLING_NO_TEXT_LAYOUT = "roi_circling_no_text_layout_position"
AUG_ABLATION_TEXT_LAYOUT = "ablation_text_layout_position"
AUG_ABLATION_NO_TEXT_LAYOUT = "ablation_no_text_layout_position"
AUG_ABLATION_TEXT = "ablation"
AUG_ABLATION_REMOVING_OBJECT = "ablation_no_object"
AUG_GROUNDING_PHYSICS = "grounding_physics"
AUG_ABLATION_PHYSICS_DURATION_TEXT = "ablation_physics_duration_text"
AUG_ABLATION_PHYSICS_MASS_TEXT = "ablation_physics_mass_text"
AUG_ABLATION_PHYSICS_MASS_APPROX_TEXT = "ablation_physics_mass_approx_text"

_AUGMENTATION_ALIASES = {
    # Backward compatibility with previous naming.
    "no_roi_circling_yes_text_layout_position": AUG_ABLATION_TEXT_LAYOUT,
    "no_roi_circling_no_text_layout_position": AUG_ABLATION_NO_TEXT_LAYOUT,
    # Short aliases for physics-cue ablations.
    "duration_text": AUG_ABLATION_PHYSICS_DURATION_TEXT,
    "mass_text": AUG_ABLATION_PHYSICS_MASS_TEXT,
    "ablation_duration_text": AUG_ABLATION_PHYSICS_DURATION_TEXT,
    "ablation_mass_text": AUG_ABLATION_PHYSICS_MASS_TEXT,
}


def _normalize_augmentation_name(augmentation):
    if augmentation is None:
        return None
    return _AUGMENTATION_ALIASES.get(augmentation, augmentation)


def get_counterfactual_image_paths(file_names):
    counterfactual_image_paths = []

    for file in file_names:
        # Map any counterfactual branch, e.g.:
        # /dl3dv-counterfact/jitter-xy/... -> /dl3dv/...
        simulation_id_path_og = re.sub(
            r"/dl3dv-counterfact/[^/]+/", "/dl3dv/", file
        ).replace("//", "/")

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
    augmentation = _normalize_augmentation_name(augmentation)
    if augmentation is None:
        return file_names

    # Let's route here based on the flags.
    if augmentation == AUG_ROI_CIRCLING_TEXT:
        file_names = augment_roi_circling(
            question,
            world_state,
            resolved_attributes,
            file_names,
            text=True,
            layout_position=False,
            save_images=True,
        )
    elif augmentation == AUG_ROI_CIRCLING_NO_TEXT:
        file_names = augment_roi_circling(
            question,
            world_state,
            resolved_attributes,
            file_names,
            text=False,
            layout_position=False,
        )
    elif augmentation == AUG_ROI_CIRCLING_TEXT_LAYOUT:
        file_names = augment_roi_circling(
            question,
            world_state,
            resolved_attributes,
            file_names,
            text=True,
            layout_position=True,
        )
    elif augmentation == AUG_ROI_CIRCLING_NO_TEXT_LAYOUT:
        file_names = augment_roi_circling(
            question,
            world_state,
            resolved_attributes,
            file_names,
            text=False,
            layout_position=True,            
        )
    elif augmentation == AUG_ABLATION_REMOVING_OBJECT:
        file_names = remove_objects_ablation(
            question,
            world_state,
            resolved_attributes,
            file_names,
            text=True,
            layout_position=False, 

        ) 
    elif augmentation == AUG_ABLATION_TEXT_LAYOUT:
        file_names = augment_ablation(
            question,
            world_state,
            resolved_attributes,
            file_names,
            text=True,
            layout_position=True,
        )
    elif augmentation == AUG_ABLATION_NO_TEXT_LAYOUT:
        file_names = augment_ablation(
            question,
            world_state,
            resolved_attributes,
            file_names,
            text=False,
            layout_position=True,
        )

    # # This is just to keep the same paths but with no augmentation.
    elif augmentation == AUG_ABLATION_TEXT:
        file_names = augment_ablation(
            question,
            world_state,
            resolved_attributes,
            file_names,
            text=True,
            layout_position=False,
        )

    # Other augmentations.
    elif augmentation == AUG_GROUNDING_PHYSICS:
        file_names = augment_scene_context(
            question, world_state, resolved_attributes, file_names
        )
    elif augmentation == AUG_ABLATION_PHYSICS_DURATION_TEXT:
        file_names = augment_duration_context(question, world_state, resolved_attributes, file_names)
    elif augmentation == AUG_ABLATION_PHYSICS_MASS_TEXT:
        file_names = augment_mass_context(question, world_state, resolved_attributes, file_names)
    elif augmentation == AUG_ABLATION_PHYSICS_MASS_APPROX_TEXT:
        file_names = augment_mass_approx_context(question, world_state, resolved_attributes, file_names) 

    return file_names


def _get_roi_output_file_name(file_name, question_key, object_name):
    # Use a single canonical ROI folder so all ROI variants share image assets.
    safe_object_name = object_name.replace(" ", "-")
    return (
        file_name.replace("render", "render_circling")
        .replace(".png", f"_{question_key}_{safe_object_name}.png")
        .replace("simulations_v4", "simulations_v4_augmented")
        .replace(
            "simulation_v4", "simulation_v4_augmented"
        )  # account for Karolina and local path
    )

def _get_roi_filled_output_file_name(file_name, question_key, object_name):
    # Use a single canonical ROI folder so all ROI variants share image assets.
    safe_object_name = object_name.replace(" ", "-")
    return (
        file_name.replace("render", "render_circling_filled")
        .replace(".png", f"_{question_key}_{safe_object_name}.png")
        .replace("simulations_v4", "simulations_v4_augmented")
        .replace(
            "simulation_v4", "simulation_v4_augmented"
        )  # account for Karolina and local path
    )


def augment_roi_circling(
    question,
    world_state,
    resolved_attributes,
    file_names,
    text=True,
    layout_position=False,
    save_images=False,
):
    if resolved_attributes == {}:
        return file_names

    if len(resolved_attributes) == 0:
        raise ImpossibleToAnswer(
            "No resolved attributes for ROI circling. So no need to circle anything, and to ask questions about it"
        )

    new_question = question["question"]

    for file_idx, file in enumerate(file_names):
        original_image = np.array(Image.open(file))
        object_name = None

        for idx, (resolved_attr, value) in enumerate(resolved_attributes.items()):
            if "OBJECT" in resolved_attr:
                object_id = value["choice"]["id"]
                render_name = file.split("/")[-1]

                # if save_images:
                instance_image_path = file.replace("render", "instances")
                rgb_object_class = world_state["encoding"]["classes"][
                    int(object_id) + 1
                ]
                visible_object_mask = (
                    np.array(Image.open(instance_image_path).convert("RGB"))
                    == rgb_object_class
                )
                visible_object_mask = np.all(visible_object_mask, axis=-1)

                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                eroded = cv2.erode(
                    visible_object_mask.astype(np.uint8), kernel, iterations=1
                )
                inner_border_mask = visible_object_mask & ~eroded

                # Create a binary mask where the object is located.
                binary_mask = inner_border_mask > 0

                contours, _ = cv2.findContours(
                    binary_mask.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )

                if len(contours) == 0:
                    raise ImpossibleToAnswer("No visible object found.")

                pts = np.vstack([c.reshape(-1, 2) for c in contours]).astype(
                    np.float32
                )
                center, radius = cv2.minEnclosingCircle(pts)

                # Draw the ROI circle on the image.
                augmented_image = draw_roi_circle(
                    original_image, center, radius * 1.5, idx
                )

                object_name = value["choice"]["name"]
                pattern = re.compile(re.escape('"' + object_name + '"'), re.IGNORECASE)
                if text:
                    # Modify the question text to include ROI reference.
                    if layout_position:
                        zone_to_focus = get_object_zone(
                            world_state, object_id, int(render_name.replace(".png", ""))
                        )
                        new_question = pattern.sub(
                            f"\"{object_name}\" (circled in red and located at the {zone_to_focus})",
                            question["question"],
                        )
                    else:
                        new_question = pattern.sub(
                            f"\"{object_name}\" (circled in red)", question["question"]
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

        if object_name is None:
            continue

        new_file_name = _get_roi_output_file_name(
            file, question["_question_key"], object_name
        )

        if save_images:
            # print("New path name", new_file_name)
            augmented_image = Image.fromarray(augmented_image)
            path = Path(new_file_name)
            path.parent.mkdir(parents=True, exist_ok=True)
            augmented_image.save(new_file_name)
            file_names[file_idx] = new_file_name
            # print("No problems")
        else:
            # If ROI assets were pre-generated, reuse that path.
            # Otherwise keep the original image path to avoid dangling references.
            if Path(new_file_name).exists():
                file_names[file_idx] = new_file_name

    # if new_question is None:
    #     raise ImpossibleToAnswer("No modifications done to the question in ROI circling augmentation.")

    if len(resolved_attributes) > 0:
        question["question"] = new_question

    return file_names


def remove_objects_ablation(
    question,
    world_state,
    resolved_attributes,
    file_names,
    text=True,
    layout_position=False,
    save_images=True,
):
    if resolved_attributes == {}:
        return file_names

    if len(resolved_attributes) == 0:
        raise ImpossibleToAnswer(
            "No resolved attributes for ROI circling. So no need to circle anything, and to ask questions about it"
        )

    new_question = question["question"]

    for file_idx, file in enumerate(file_names):
        original_image = np.array(Image.open(file))
        object_name = None

        for idx, (resolved_attr, value) in enumerate(resolved_attributes.items()):
            if "OBJECT" in resolved_attr:
                object_id = value["choice"]["id"]
                render_name = file.split("/")[-1]

                # if save_images:
                instance_image_path = file.replace("render", "instances")
                rgb_object_class = world_state["encoding"]["classes"][
                    int(object_id) + 1
                ]
                visible_object_mask = (
                    np.array(Image.open(instance_image_path).convert("RGB"))
                    == rgb_object_class
                )
                visible_object_mask = np.all(visible_object_mask, axis=-1)

                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                eroded = cv2.erode(
                    visible_object_mask.astype(np.uint8), kernel, iterations=1
                )
                inner_border_mask = visible_object_mask & ~eroded

                # Create a binary mask where the object is located.
                binary_mask = inner_border_mask > 0

                contours, _ = cv2.findContours(
                    binary_mask.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )

                if len(contours) == 0:
                    raise ImpossibleToAnswer("No visible object found.")

                pts = np.vstack([c.reshape(-1, 2) for c in contours]).astype(
                    np.float32
                )
                center, radius = cv2.minEnclosingCircle(pts)

                # Draw the ROI circle on the image.
                # augmented_image = draw_roi_circle(
                #     original_image, center, radius * 1.5, idx
                # )

                # Draw the filled circle on the image.
                augmented_image = draw_roi_circle_filled(
                    original_image, center, radius * 1.5, idx 
                )

                object_name = value["choice"]["name"]
                pattern = re.compile(re.escape('"' + object_name + '"'), re.IGNORECASE)
                if text:
                    # Modify the question text to include ROI reference.
                    if layout_position:
                        zone_to_focus = get_object_zone(
                            world_state, object_id, int(render_name.replace(".png", ""))
                        )
                        new_question = pattern.sub(
                            f"\"{object_name}\" (circled in red and located at the {zone_to_focus})",
                            question["question"],
                        )
                    else:
                        new_question = pattern.sub(
                            f"\"{object_name}\" (circled in red)", question["question"]
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

        if object_name is None:
            continue

        new_file_name = _get_roi_filled_output_file_name(
            file, question["_question_key"], object_name
        )

        if save_images:
            # print("New path name", new_file_name)
            augmented_image = Image.fromarray(augmented_image)
            path = Path(new_file_name)
            path.parent.mkdir(parents=True, exist_ok=True)
            augmented_image.save(new_file_name)
            file_names[file_idx] = new_file_name
            # print("No problems")
        else:
            # If ROI assets were pre-generated, reuse that path.
            # Otherwise keep the original image path to avoid dangling references.
            if Path(new_file_name).exists():
                file_names[file_idx] = new_file_name

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
    save_images=False,
):
    if resolved_attributes == {}:
        return file_names

    if len(resolved_attributes) == 0:
        raise ImpossibleToAnswer(
            "No resolved attributes for ROI circling. So no need to circle anything, and to ask questions about it"
        )

    new_question = question["question"]

    for file_idx, file in enumerate(file_names):
        original_image = np.array(Image.open(file))
        object_name = None

        for idx, (resolved_attr, value) in enumerate(resolved_attributes.items()):
            if "OBJECT" in resolved_attr:
                object_id = value["choice"]["id"]
                render_name = file.split("/")[-1]

                # if save_images:
                instance_image_path = file.replace("render", "instances")
                rgb_object_class = world_state["encoding"]["classes"][
                    int(object_id) + 1
                ]
                visible_object_mask = (
                    np.array(Image.open(instance_image_path).convert("RGB"))
                    == rgb_object_class
                )
                visible_object_mask = np.all(visible_object_mask, axis=-1)

                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                eroded = cv2.erode(
                    visible_object_mask.astype(np.uint8), kernel, iterations=1
                )
                inner_border_mask = visible_object_mask & ~eroded

                # Create a binary mask where the object is located.
                binary_mask = inner_border_mask > 0

                contours, _ = cv2.findContours(
                    binary_mask.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )

                if len(contours) == 0:
                    raise ImpossibleToAnswer("No visible object found.")

                pts = np.vstack([c.reshape(-1, 2) for c in contours]).astype(
                    np.float32
                )
                center, radius = cv2.minEnclosingCircle(pts)

                # Draw the ROI circle on the image.
                augmented_image = draw_roi_circle(
                    original_image, center, radius * 1.5, idx
                )

                object_name = value["choice"]["name"]
                pattern = re.compile(re.escape('"' + object_name + '"'), re.IGNORECASE)
                if text:
                    # Modify the question text to include ROI reference.
                    if layout_position:
                        zone_to_focus = get_object_zone(
                            world_state, object_id, int(render_name.replace(".png", ""))
                        )
                        new_question = pattern.sub(
                            f"\"{object_name}\" (located at the {zone_to_focus})",
                            question["question"],
                        )
                    else:
                        new_question = pattern.sub(
                            f"\"{object_name}\"", question["question"]
                        )
                else:
                    if layout_position:
                        # append after the name of the object that it is circled in the image
                        zone_to_focus = get_object_zone(
                            world_state, object_id, int(render_name.replace(".png", ""))
                        )
                        new_question = pattern.sub(
                            f"(located at the {zone_to_focus})",
                            question["question"],
                        )
                    else:
                        # append after the name of the object that it is circled in the image
                        new_question = pattern.sub(
                            "object circled in red", question["question"]
                        )

        if object_name is None:
            continue

        # new_file_name = _get_roi_output_file_name(
        #     file, question["_question_key"], object_name
        # )

        # if save_images:
        #     # print("New path name", new_file_name)
        #     augmented_image = Image.fromarray(augmented_image)
        #     path = Path(new_file_name)
        #     path.parent.mkdir(parents=True, exist_ok=True)
        #     augmented_image.save(new_file_name)
        #     file_names[file_idx] = new_file_name
        #     # print("No problems")
        # else:
        #     # If ROI assets were pre-generated, reuse that path.
        #     # Otherwise keep the original image path to avoid dangling references.
        #     if Path(new_file_name).exists():
        #         file_names[file_idx] = new_file_name

    # if new_question is None:
    #     raise ImpossibleToAnswer("No modifications done to the question in ROI circling augmentation.")

    if len(resolved_attributes) > 0:
        question["question"] = new_question

    return file_names


def _extract_frame_indices(file_names):
    frame_indices = []
    for file_name in file_names:
        stem = Path(file_name).stem
        if stem.isdigit():
            frame_indices.append(int(stem))
    return frame_indices


def augment_duration_context(question,
        world_state,
        resolved_attributes, 
        file_names, 
        text=False,
        layout_position=False
    ):
    frame_indices = _extract_frame_indices(file_names)
    duration_seconds = 0.0
    if len(frame_indices) > 1:
        duration_seconds = (max(frame_indices) - min(frame_indices)) * time_interval

    question["question"] = (
        f"Physics cue: The provided images span {duration_seconds:.2f} seconds. "
        + question["question"]
    )

    for file_idx, file in enumerate(file_names):
        original_image = np.array(Image.open(file))
        object_name = None

        for idx, (resolved_attr, value) in enumerate(resolved_attributes.items()):
            if "OBJECT" in resolved_attr:
                object_id = value["choice"]["id"]
                render_name = file.split("/")[-1]

                # if save_images:
                instance_image_path = file.replace("render", "instances")
                rgb_object_class = world_state["encoding"]["classes"][
                    int(object_id) + 1
                ]
                visible_object_mask = (
                    np.array(Image.open(instance_image_path).convert("RGB"))
                    == rgb_object_class
                )
                visible_object_mask = np.all(visible_object_mask, axis=-1)

                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                eroded = cv2.erode(
                    visible_object_mask.astype(np.uint8), kernel, iterations=1
                )
                inner_border_mask = visible_object_mask & ~eroded

                # Create a binary mask where the object is located.
                binary_mask = inner_border_mask > 0

                contours, _ = cv2.findContours(
                    binary_mask.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )

                if len(contours) == 0:
                    raise ImpossibleToAnswer("No visible object found.")

                pts = np.vstack([c.reshape(-1, 2) for c in contours]).astype(
                    np.float32
                )
                center, radius = cv2.minEnclosingCircle(pts)

                # Draw the ROI circle on the image.
                augmented_image = draw_roi_circle(
                    original_image, center, radius * 1.5, idx
                )

                object_name = value["choice"]["name"]
                pattern = re.compile(re.escape('"' + object_name + '"'), re.IGNORECASE)
                if text:
                    # Modify the question text to include ROI reference.
                    if layout_position:
                        zone_to_focus = get_object_zone(
                            world_state, object_id, int(render_name.replace(".png", ""))
                        )
                        new_question = pattern.sub(
                            f"\"{object_name}\" (circled in red and located at the {zone_to_focus})",
                            question["question"],
                        )
                    else:
                        new_question = pattern.sub(
                            f"\"{object_name}\"", question["question"]
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

        if object_name is None:
            continue


    return file_names


def augment_mass_context(question, world_state, resolved_attributes, file_names, layout_position=False, text=False):
    cues = []
    seen_ids = set()

    for resolved_attr, value in resolved_attributes.items():
        if "OBJECT" not in resolved_attr:
            continue

        choice = value.get("choice", {})
        object_id = choice.get("id")
        if object_id in seen_ids:
            continue

        mass = choice.get("mass")
        object_name = choice.get("name", "object")
        if mass is None:
            continue

        try:
            mass = float(mass)
        except (TypeError, ValueError):
            continue

        cues.append(f'"{object_name}" has mass {mass:.2f} kg')
        seen_ids.add(object_id)

    if cues:
        question["question"] = f"Physics cue: {'; '.join(cues)}. " + question["question"]

    for file_idx, file in enumerate(file_names):
        original_image = np.array(Image.open(file))
        object_name = None

        for idx, (resolved_attr, value) in enumerate(resolved_attributes.items()):
            if "OBJECT" in resolved_attr:
                object_id = value["choice"]["id"]
                render_name = file.split("/")[-1]

                # if save_images:
                instance_image_path = file.replace("render", "instances")
                rgb_object_class = world_state["encoding"]["classes"][
                    int(object_id) + 1
                ]
                visible_object_mask = (
                    np.array(Image.open(instance_image_path).convert("RGB"))
                    == rgb_object_class
                )
                visible_object_mask = np.all(visible_object_mask, axis=-1)

                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                eroded = cv2.erode(
                    visible_object_mask.astype(np.uint8), kernel, iterations=1
                )
                inner_border_mask = visible_object_mask & ~eroded

                # Create a binary mask where the object is located.
                binary_mask = inner_border_mask > 0

                contours, _ = cv2.findContours(
                    binary_mask.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )

                if len(contours) == 0:
                    raise ImpossibleToAnswer("No visible object found.")

                pts = np.vstack([c.reshape(-1, 2) for c in contours]).astype(
                    np.float32
                )
                center, radius = cv2.minEnclosingCircle(pts)

                # Draw the ROI circle on the image.
                augmented_image = draw_roi_circle(
                    original_image, center, radius * 1.5, idx
                )

                object_name = value["choice"]["name"]
                pattern = re.compile(re.escape('"' + object_name + '"'), re.IGNORECASE)
                if text:
                    # Modify the question text to include ROI reference.
                    if layout_position:
                        zone_to_focus = get_object_zone(
                            world_state, object_id, int(render_name.replace(".png", ""))
                        )
                        new_question = pattern.sub(
                            f"\"{object_name}\" (circled in red and located at the {zone_to_focus})",
                            question["question"],
                        )
                    else:
                        new_question = pattern.sub(
                            f"\"{object_name}\"", question["question"]
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

        if object_name is None:
            continue

    return file_names

def augment_mass_approx_context(question, world_state, resolved_attributes, file_names, layout_position=False, text=False):
    cues = []
    seen_ids = set()

    for resolved_attr, value in resolved_attributes.items():
        if "OBJECT" not in resolved_attr:
            continue

        choice = value.get("choice", {})
        object_id = choice.get("id")
        if object_id in seen_ids:
            continue

        mass = choice.get("mass")
        object_name = choice.get("name", "object")
        if mass is None:
            continue

        try:
            mass = float(mass)
        except (TypeError, ValueError):
            continue

        approx_mass = round(mass * 2.0) / 2.0
        if approx_mass.is_integer():
            approx_mass_str = f"{int(approx_mass)}"
        else:
            approx_mass_str = f"{approx_mass:.1f}"
        cues.append(f'"{object_name}" has mass approx. {approx_mass_str} kg')
        seen_ids.add(object_id)

    if cues:
        question["question"] = f"Physics cue: {'; '.join(cues)}. " + question["question"]

    for file_idx, file in enumerate(file_names):
        original_image = np.array(Image.open(file))
        object_name = None

        for idx, (resolved_attr, value) in enumerate(resolved_attributes.items()):
            if "OBJECT" in resolved_attr:
                object_id = value["choice"]["id"]
                render_name = file.split("/")[-1]

                # if save_images:
                instance_image_path = file.replace("render", "instances")
                rgb_object_class = world_state["encoding"]["classes"][
                    int(object_id) + 1
                ]
                visible_object_mask = (
                    np.array(Image.open(instance_image_path).convert("RGB"))
                    == rgb_object_class
                )
                visible_object_mask = np.all(visible_object_mask, axis=-1)

                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                eroded = cv2.erode(
                    visible_object_mask.astype(np.uint8), kernel, iterations=1
                )
                inner_border_mask = visible_object_mask & ~eroded

                # Create a binary mask where the object is located.
                binary_mask = inner_border_mask > 0

                contours, _ = cv2.findContours(
                    binary_mask.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )

                if len(contours) == 0:
                    raise ImpossibleToAnswer("No visible object found.")

                pts = np.vstack([c.reshape(-1, 2) for c in contours]).astype(
                    np.float32
                )
                center, radius = cv2.minEnclosingCircle(pts)

                # Draw the ROI circle on the image.
                augmented_image = draw_roi_circle(
                    original_image, center, radius * 1.5, idx
                )

                object_name = value["choice"]["name"]
                pattern = re.compile(re.escape('"' + object_name + '"'), re.IGNORECASE)
                if text:
                    # Modify the question text to include ROI reference.
                    if layout_position:
                        zone_to_focus = get_object_zone(
                            world_state, object_id, int(render_name.replace(".png", ""))
                        )
                        new_question = pattern.sub(
                            f"\"{object_name}\" (circled in red and located at the {zone_to_focus})",
                            question["question"],
                        )
                    else:
                        new_question = pattern.sub(
                            f"\"{object_name}\"", question["question"]
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

        if object_name is None:
            continue

    return file_names

def draw_roi_circle(original_image, center, radius=10, idx=0):
    cx, cy = map(int, center)
    cv2.circle(original_image, (cx, cy), int(radius), (255, 0, 0), 5, cv2.LINE_AA)

    return original_image

def draw_roi_circle_filled(original_image, center, radius=10, idx=0):
    cx, cy = map(int, center)
    cv2.circle(original_image, (cx, cy), int(radius), (0, 0, 0), -1, cv2.LINE_AA)

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
