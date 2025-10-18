import os
import re
import json
import csv
import argparse
import string

from PIL import Image

from utils.encoding_vlm import encode_image_file_to_base64

# Import categories - alphabetically

from categories.collision.collision import (
    get_function_by_name_collision,
    get_result_by_name_collision,
)

from categories.force.force import (
    get_function_by_name_force,
    get_result_by_name_force,
)

from categories.kinematics.kinematics import (
    get_function_by_name_kinematics,
    get_result_by_name_kinematics,
)

from categories.mass.mass import (
    get_function_by_name_mass,
    get_result_by_name_mass,
)

from categories.material.material import (
    get_function_by_name_material,
    get_result_by_name_material,
)

from categories.meta.meta import (
    get_function_by_name_meta,
    get_result_by_name_meta,
)

from categories.spatial.spatial import (
    get_function_by_name_spatial,
    get_result_by_name_spatial,
)

from categories.temporal.temporal import (
    get_function_by_name_temporal,
    get_result_by_name_temporal,
)

from categories.visibility.visibility import (
    get_function_by_name_visibility,
    get_result_by_name_visibility,
)

from categories.volume.volume import (
    get_function_by_name_volume,
    get_result_by_name_volume,
)


# ----- UTILS FUNCTIONS
def read_questions(vqa_path):
    with open(vqa_path, "r") as f:
        questions = json.load(f)
    return questions


def read_simulation(simulation_path):
    with open(simulation_path, "r") as f:
        simulation_steps = json.load(f)
    return simulation_steps

""" QUESTION JSON
    {
        "scene": "black background",
        "object": [
            "glass",
            "rubber bullet"
        ],
        "source": "web",
        "file_name": [
            "iNINChj51Aqn.mp4",
            "iNINChj51Aqj.png",
            "iNINChj51Aqk.png",
            "iNINChj51Aql.png",
            "iNINChj51Aqm.png"
        ],
        "description": null,
        "question": "Following the content of the <video>, which option's corresponding picture will happen first?\nA. <image>\nB. <image>\nC. <image>\nD. <image>\n",
        "mode": "general",
        "idx": 0,
        "split": "val"
    },
"""

""" ANSWER JSON
{
        "idx": 0,
        "answer": "A",
        "task_type": "dynamics",
        "sub_type": "collision",
        "ability_type": "prediction",
        "mode": "general"
},
"""


def _determine_tsv_fieldnames(question_records, answer_records):
    preferred_order = [
        "index",
        "image",
        "image_path",
        "question",
        "hint",
        "multi-choice options",
        "options",
        "answer",
        "category",
        "l2-category",
        "split",
    ]

    question_keys = set()
    for record in question_records:
        question_keys.update(record.keys())

    answer_keys = set()
    for answer in answer_records:
        if isinstance(answer, dict):
            answer_keys.update(answer.keys())
        elif answer is not None:
            answer_keys.add("answer")

    question_keys.discard("index")
    answer_keys.discard("index")

    ordered = []
    for field in preferred_order:
        if field == "index" or field in question_keys or field in answer_keys:
            if field != "index":
                question_keys.discard(field)
                answer_keys.discard(field)
            ordered.append(field)

    remaining = sorted(question_keys.union(answer_keys))
    for field in remaining:
        if field not in ordered:
            ordered.append(field)

    return ordered


def _stringify_tsv_value(value):
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _build_tsv_row(record, answer, fieldnames):
    row = {field: "" for field in fieldnames}
    for key, value in record.items():
        if key in row:
            row[key] = _stringify_tsv_value(value)
    if answer is not None:
        if isinstance(answer, dict):
            for key, value in answer.items():
                if key in row:
                    row[key] = _stringify_tsv_value(value)
        else:
            if "answer" in row:
                row["answer"] = _stringify_tsv_value(answer)
    if "index" in row and "index" in record:
        row["index"] = _stringify_tsv_value(record["index"])
    return row


def save_questions_answers_json(
    all_vqa,
    simulation_steps,
    output_path,
    export_format="json",
    image_output="base64",
    number_of_images_max=8,
    root_image_path="",
):
    os.makedirs(output_path, exist_ok=True)
    export_targets = {"json", "tsv"} if export_format == "both" else {export_format}
    normalized_questions = []
    answers = []

    for idx, entry in enumerate(all_vqa):
        question_record, answer_record = normalize_question_json(
            entry,
            idx=idx,
            simulation_steps=simulation_steps,
            image_output=image_output,
            number_of_images_max=number_of_images_max,
            root_image_path=root_image_path,
        )

        normalized_questions.append(question_record) 
        answers.append(answer_record)

    
    questions_path = os.path.join(output_path, "test.json")
    answers_path = os.path.join(output_path, "val_answer.json")

    with open(questions_path, "w") as f:
        json.dump(normalized_questions, f, indent=4)

    with open(answers_path, "w") as f:
        json.dump(answers, f, indent=4)

    if "tsv" in export_targets:
        fieldnames = _determine_tsv_fieldnames(normalized_questions, answers)
        tsv_path = os.path.join(output_path, "questions.tsv")
        with open(tsv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            for idx, record in enumerate(normalized_questions):
                answer = answers[idx] if idx < len(answers) else None
                row = _build_tsv_row(record, answer, fieldnames)
                writer.writerow(row)

def normalize_question_json(
    vqa_entry,
    idx,
    simulation_steps,
    image_output="base64",
    number_of_images_max=8,
    root_image_path="",
):
    question_payload = vqa_entry.get("question", {})
    question_text = question_payload.get("question", "").strip()
    labels = vqa_entry.get("labels", [])
    answer_index = vqa_entry.get("answer_index")
    image_indexes = vqa_entry.get("image_indexes", []) or []
    letters = list(string.ascii_uppercase)
    
    #regex to check if in the label we have an image
    pattern = re.compile(r'^\d{6}$')

    for idx, label in enumerate(labels):
        print("label", label)
        if pattern.match(label):
            image_indexes.append(str(label))
            labels[idx] = f"<image>"

    option_letters = [letters[i] for i in range(min(len(labels), len(letters)))]
    option_lines = []
    for letter, label in zip(option_letters, labels):
        option_lines.append(f"{letter}. {label}")

    formatted_question = question_text
    if option_lines:
        formatted_question = f"{formatted_question}\n" + "\n".join(option_lines)

    # add <image> tags in place of images
    formatted_question = "".join(["<image>" for _ in image_indexes]) + "\n" + formatted_question

    scene_info = simulation_steps.get("scene", {}) if simulation_steps else {}
    scene_name = scene_info.get("scene") or scene_info.get("name") or "simulation_scene"


    file_names = [root_image_path + f"/render/{int(frame_idx):06d}.png" for frame_idx in image_indexes]

    ability_map = {
        "prediction": "prediction",
        "counting": "counting",
        "estimation": "estimation",
        "attribute": "attribute",
    }
    measurement = question_payload.get("measurement")
    ability_type = ability_map.get(measurement, "general")

    answer_letter = None
    if (
        answer_index is not None
        and 0 <= answer_index < len(option_letters)
    ):
        answer_letter = option_letters[answer_index]

    question_record = {
        "scene": scene_name,
        "source": "simulation",
        "file_name": file_names,
        "description": question_payload.get("description"),
        "question": formatted_question,
        "mode": "image-only",
        "idx": idx,
        "split": question_payload.get("split", "val"),
    }

    answer_record = {
        "idx": idx,
        "answer": answer_letter,
        "task_type": "factual",
        "sub_type": question_payload.get("category"),
        "ability_type": question_payload.get("ability_type", ability_type),
        "mode": question_record["mode"],
    }

    return question_record, answer_record

def save_questions_answers_tsv(
    all_vqa,
    simulation_steps,
    output_path,
    export_format="json",
    image_output="base64",
    number_of_images_max=8,
    root_image_path="",
):
    os.makedirs(output_path, exist_ok=True)
    export_targets = {"json", "tsv"} if export_format == "both" else {export_format}
    normalized_questions = []
    answers = []

    for idx, entry in enumerate(all_vqa):
        question_record = normalize_question_tsv(
            entry,
            idx=idx,
            simulation_steps=simulation_steps,
            image_output=image_output,
            number_of_images_max=number_of_images_max,
            root_image_path=root_image_path,
        )

        normalized_questions.append(question_record) 
    
    fieldnames = _determine_tsv_fieldnames(normalized_questions, answers)
    tsv_path = os.path.join(output_path, "questions.tsv")
    with open(tsv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for idx, record in enumerate(normalized_questions):
            answer = answers[idx] if idx < len(answers) else None
            row = _build_tsv_row(record, answer, fieldnames)
            writer.writerow(row)


def normalize_question_tsv(
    vqa_entry,
    idx,
    simulation_steps,
    image_output="base64",
    number_of_images_max=8,
    root_image_path="",
):
    question_payload = vqa_entry.get("question", {})
    question_text = question_payload.get("question", "").strip()
    labels = vqa_entry.get("labels", [])
    answer_index = vqa_entry.get("answer_index")
    image_indexes = vqa_entry.get("image_indexes", []) or []

    scene_info = simulation_steps.get("scene", {}) if simulation_steps else {}
    scene_name = scene_info.get("scene") or scene_info.get("name") or "simulation_scene"

    objects_metadata = simulation_steps.get("objects", {}) if simulation_steps else {}
    detected_objects = set()
    lowered_question = question_text.lower()
    for obj in objects_metadata.values():
        description = obj.get("description", {}) or {}
        object_name = description.get("object_name") or obj.get("model") or obj.get(
            "name"
        )
        if not object_name:
            continue
        if object_name.lower() in lowered_question:
            detected_objects.add(object_name)

    label_lookup = {
        obj.get("model", "").lower(): obj
        for obj in objects_metadata.values()
        if obj.get("model")
    }
    for label in labels:
        label_key = label.replace(" ", "_").lower()
        if label_key in label_lookup:
            obj = label_lookup[label_key]
            description = obj.get("description", {}) or {}
            object_name = (
                description.get("object_name") or obj.get("model") or obj.get("name")
            )
            if object_name:
                detected_objects.add(object_name)

    file_names = [root_image_path + f"/render/{int(frame_idx):06d}.png" for frame_idx in image_indexes]
    images_base64 = [encode_image_file_to_base64(image_path) for image_path in file_names]

    ability_map = {
        "prediction": "prediction",
        "counting": "counting",
        "estimation": "estimation",
        "attribute": "attribute",
    }
    measurement = question_payload.get("measurement")
    ability_type = ability_map.get(measurement, "general")

    """
    Full row (raw values):
    index: '1000205'
    question: 'Based on the description, how are the people in the image engaging with the game?'
    hint: ''
    A: 'The group of people is engaging with the game by playing a board game.'
    B: 'The group of people is physically engaging with the game by using Nintendo Wii controllers.'
    C: 'The group of people is physically engaging with the game by using traditional gaming controllers.'
    D: 'The group of people is engaging with the game by watching a screen passively.'
    answer: 'B'
    category: 'function_reasoning'
    image: <base64 image omitted>
    source: 'reasoning'
    l2-category: 'attribute_reasoning'
    comment: ''
    split: 'dev'
    """

    question_record = {
        "index": idx,
        "images": images_base64,
        "A": labels[0] if len(labels) > 0 else "",
        "B": labels[1] if len(labels) > 1 else "",
        "C": labels[2] if len(labels) > 2 else "",
        "D": labels[3] if len(labels) > 3 else "",
        "question": question_text,
        "answer": ["A", "B", "C", "D"][answer_index] if answer_index is not None and 0 <= answer_index < 4 else "",
        "category": question_payload.get("category"),
        "source": "simulation",
        "l2-category": question_payload.get("ability_type", ability_type),
    }

    return question_record

# ----- FUNCTION TO GET ANSWER FROM SIMULTAION

resolver_gt = {
    "collision": get_result_by_name_collision,
    "force": get_result_by_name_force,
    "kinematics": get_result_by_name_kinematics,
    "mass": get_result_by_name_mass,
    "material": get_result_by_name_material,
    "meta": get_result_by_name_meta,
    "spatial": get_result_by_name_spatial,
    "temporal": get_result_by_name_temporal,
    "visibility": get_result_by_name_visibility,
    "volume": get_result_by_name_volume,
}

resolver = {
    "collision": get_function_by_name_collision,
    "force": get_function_by_name_force,
    "kinematics": get_function_by_name_kinematics,
    "mass": get_function_by_name_mass,
    "material": get_function_by_name_material,
    "meta": get_function_by_name_meta,
    "spatial": get_function_by_name_spatial,
    "temporal": get_function_by_name_temporal,
    "visibility": get_function_by_name_visibility,
    "volume": get_function_by_name_volume,
}


def get_answer(question_key, question_category, mock=False):
    return resolver[question_category](question_key, mock=mock)


def get_gt(question_key, question_category, mock=False):
    return resolver_gt[question_category](question_key, mock=mock)


# ----- MAIN VQA CREATION LOGIC
def create_vqa(questions, simulation_steps, arg_mock, verbose=False):
    total_correct_per_category = {}

    print("Starting VQA creation...")

    all_vqa = []

    for category_key, category in questions.items():
        # current category dev
        if category_key == "force" or category_key == "deformation" or category_key == "visibility":
            continue

        if verbose:
            print("###" * 10, f"Processing category: {category_key}", "###" * 10)
            print(f"questions: \n{category}")
            print("###" * 20)
        total_questions_in_category = len(category)
        total_correct_answers = 0
        not_implemented = 0
        for question_key, question_data in category.items():
            if verbose:
                print(f"  Question Key: {question_key}")
            fn_to_answer_question = get_answer(
                question_key, category_key, mock=arg_mock
            )
            question, labels, correct_idx, imgs_idx = fn_to_answer_question(
                simulation_steps, question_data
            )
            all_vqa.append(
                {
                    "question": question,
                    "category": category_key,
                    "question_key": question_key,
                    "image_indexes": imgs_idx,
                    "labels": labels,
                    "answer_index": correct_idx,
                })
            
            if verbose:
                print(f"  Question: {question}")
                print(f"  Labels: {labels}")
                print(f"  Correct Index: {correct_idx}")
                print(f"  Images Indexes: {imgs_idx}")

            gt = get_gt(question_key, category_key, mock=arg_mock)
            if verbose:
                print(
                    f"  Answer from function: {labels[correct_idx]}\n  Should match GT: {gt}"
                )

            # Just for development, the rng function given more or less functions will break the integration test
            if str(labels[correct_idx]) != str(gt):
                print("\033[91m  WARNING: Answer does not match Ground Truth!\033[0m")
                # exit(1)
            else:
                if str(labels[correct_idx]) == "not_implemented":
                    not_implemented += 1
                else:
                    total_correct_answers += 1
            if verbose:
                print("===" * 20)
        total_correct_per_category[category_key] = (
            total_correct_answers,
            not_implemented,
            total_questions_in_category,
        )

    if verbose:
        print("Summary of correct answers per category:")
    for category, (
        correct,
        not_implemented,
        total,
    ) in total_correct_per_category.items():
        if verbose:
            print(
                f"Category '{category}': {correct}/{total - not_implemented} correct answers, {not_implemented} not implemented"
            )

    print("Total questions:")
    print(sum(total for _, (_, _, total) in total_correct_per_category.items()))

    return all_vqa

def main(args):
    questions = read_questions(args.vqa_path)
    simulation_steps = read_simulation(
        args.simulation_path + "/simulation_kinematics.json"
    )

    all_vqa = create_vqa(
        questions, simulation_steps, args.mock, verbose=args.verbose
    )
    print(
        f"Saved {len(all_vqa)} questions and answers."
    )

    if args.export_format in ["json", "both"]:
        save_questions_answers_json(
            all_vqa,
            simulation_steps,
            args.output_path,
            export_format=args.export_format,
            image_output=args.image_output,
            number_of_images_max=args.number_of_images_max,
            root_image_path=args.simulation_path,
        )
        print(f"Saved questions and answers to {args.output_path} ({args.export_format})")

    if args.export_format in ["tsv", "both"]:
        save_questions_answers_tsv(
            all_vqa,
            simulation_steps,
            args.output_path,
            export_format=args.export_format,
            image_output=args.image_output,
            number_of_images_max=args.number_of_images_max,
            root_image_path=args.simulation_path,
        )
        print(f"Saved questions and answers to {args.output_path} ({args.export_format})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vqa_path",
        type=str,
        default="../simpler_example.json",
        help="Path to simpler.json file or similar that contain all the vqa templates.",
    )
    parser.add_argument(
        "--simulation_path",
        type=str,
        # default="./sample_simulation_1000_steps_v2_kinematics.json
        default="/Users/sebastiancavada/Desktop/tmp_Paris/vqa/data/output/sims/dl3dv-hf-gso2/3-cg/c-0_no-3_d-3_s-dl3dv-1bef58393fffbf6e34cac11d0b03dd22f65954a1668b7b9dec548f6ad44f29b5_models-hf-gso_MLP-10_smooth_h-10-40_seed-0_dbgsub-1_20251016_013244",
        # default="/Users/sebastiancavada/Desktop/tmp_Paris/vqa/answering_questions/",
        help="Path to the simulation file containing the scenes.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="../output",
        help="Path to save the questions.json and answers.json files.",
    )
    parser.add_argument(
        "--export_format",
        choices=["json", "tsv", "both"],
        default="json",
        help="Output format for generated questions and answers.",
    )
    parser.add_argument(
        "--image_output",
        choices=["base64", "path"],
        default="base64",
        help="Select whether exported questions reference images via base64 or filesystem paths (TSV always uses paths).",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=True,
        help="Use mock implementations for testing.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output for debugging.",
    )
    parser.add_argument(
        "--number_of_images_max",
        type=int,
        default=8,
        help="Maximum number of images to save for VQA.",
    )
    args = parser.parse_args()

    main(args)
