import os
import re
import json
import csv
import string


from utils.encoding_vlm import encode_image_file_to_base64

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
    output_path,
    export_format="json",
    image_output="base64",
    number_of_images_max=8,
):
    os.makedirs(output_path, exist_ok=True)
    export_targets = {"json", "tsv"} if export_format == "both" else {export_format}
    normalized_questions = []
    answers = []

    for idx, entry in enumerate(all_vqa):
        question_record, answer_record = normalize_question_json(
            entry,
            idx=idx,
            image_output=image_output,
            number_of_images_max=number_of_images_max,
        )

        normalized_questions.append(question_record)
        answers.append(answer_record)

    questions_path = os.path.join(output_path, "test.json")
    answers_path = os.path.join(output_path, "val_answer.json")

    with open(questions_path, "w") as f:
        json.dump(normalized_questions, f, indent=4)

    with open(answers_path, "w") as f:
        json.dump(answers, f, indent=4)


def normalize_question_json(
    vqa_entry,
    idx,
    image_output="base64",
    number_of_images_max=8,
):
    question_payload = vqa_entry.get("question", {})
    question_text = question_payload.get("question", "").strip()
    labels = vqa_entry.get("labels", [])
    answer_index = vqa_entry.get("answer_index")
    image_paths = vqa_entry.get("image_paths", []) or []
    letters = list(string.ascii_uppercase)

    # add <image> tags in place of images
    # locking in question images before adding other images in the question
    # slop code, but guess I need to speed up
    formatted_question = question_text
    formatted_question = (
        "".join(["<image>" for _ in image_paths]) + "\n" + formatted_question
    )

    # regex to check if in the label we have an image
    pattern = re.compile(r"^\d{6}$")

    for idx_img, label in enumerate(labels):
        # print("label", label)
        if pattern.match(label):
            # do a smart replacement
            new_image_path = image_paths[0].rsplit("/", 1)[0] + f"/{label}.png"
            image_paths.append(new_image_path)
            labels[idx_img] = "<image>"

    option_letters = [letters[i] for i in range(min(len(labels), len(letters)))]
    option_lines = []
    for letter, label in zip(option_letters, labels):
        option_lines.append(f"{letter}. {label}")

    if option_lines:
        formatted_question = f"{formatted_question}\n" + "\n".join(option_lines)

    # file_names = [root_image_path + f"/render/{int(frame_idx):06d}.png" for frame_idx in image_paths]

    ability_map = {
        "prediction": "prediction",
        "counting": "counting",
        "estimation": "estimation",
        "attribute": "attribute",
    }
    measurement = question_payload.get("measurement")
    ability_type = ability_map.get(measurement, "general")

    answer_letter = None
    if answer_index is not None and 0 <= answer_index < len(option_letters):
        answer_letter = option_letters[answer_index]

    question_record = {
        "scene": vqa_entry["scene"],
        "source": "simulation",
        "simulation_id": vqa_entry.get("simulation_id", ""),
        "file_name": image_paths,
        "description": question_payload.get("description"),
        "question": formatted_question,
        "mode": vqa_entry["mode"],
        "idx": idx,
        "split": question_payload.get("split", "val"),
        "choice_type": question_payload["choice"]
    }

    answer_record = {
        "idx": idx,
        "answer": answer_letter,
        "task_type": "factual",
        "sub_type": question_payload.get("category"),
        "ability_type": question_payload.get("ability_type", ability_type),
        "mode": question_record["mode"],
        "choice_type": question_payload["choice"]
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
    image_paths = vqa_entry.get("image_paths", []) or []

    scene_info = simulation_steps.get("scene", {}) if simulation_steps else {}
    scene_name = scene_info.get("scene") or scene_info.get("name") or "simulation_scene"

    objects_metadata = simulation_steps.get("objects", {}) if simulation_steps else {}
    detected_objects = set()
    lowered_question = question_text.lower()
    for obj in objects_metadata.values():
        description = obj.get("description", {}) or {}
        object_name = (
            description.get("object_name") or obj.get("model") or obj.get("name")
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

    file_names = [
        root_image_path + f"/render/{int(frame_idx):06d}.png"
        for frame_idx in image_paths
    ]
    images_base64 = [
        encode_image_file_to_base64(image_path) for image_path in file_names
    ]

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
        "answer": ["A", "B", "C", "D"][answer_index]
        if answer_index is not None and 0 <= answer_index < 4
        else "",
        "category": question_payload.get("category"),
        "source": "simulation",
        "l2-category": question_payload.get("ability_type", ability_type),
    }

    return question_record
