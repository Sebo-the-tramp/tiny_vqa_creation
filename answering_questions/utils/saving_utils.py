import os
import re
import json
import string

from utils.config import get_config

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


def save_questions_answers_json(
    all_vqa,
    output_path,
    run_name="",
):
    os.makedirs(output_path, exist_ok=True)
    normalized_questions = []
    answers = []

    counter = 0

    for counter, entry in enumerate(all_vqa):
        # THIS IS WEIRD I DON'T KNOW WHY I DID IT BEFORE
        # if idx > 0:
        #     question_id_previous = all_vqa[idx - 1]["question_key"]
        #     question_id_current = entry["question_key"]

        #     if question_id_previous != question_id_current:
        #         counter += 1

        mode = entry["mode"]
        question_idx = f"{counter}_{mode[0]}"

        question_record, answer_record = normalize_question_json(
            entry, idx=question_idx
        )

        normalized_questions.append(question_record)
        answers.append(answer_record)

    config = get_config()
    config_path = os.path.join(
        output_path, f"{run_name}/test_{run_name}_config_used.json"
    )
    answers_path = os.path.join(output_path, f"{run_name}/val_answer_{run_name}.json")
    questions_path = os.path.join(output_path, f"{run_name}/test_{run_name}.json")

    with open(questions_path, "w", encoding="utf-8") as f:
        json.dump(normalized_questions, f, indent=2)

    with open(answers_path, "w", encoding="utf-8") as f:
        json.dump(answers, f, indent=2)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    return questions_path, answers_path


def normalize_question_json(vqa_entry, idx):
    question_payload = vqa_entry.get("question", {})
    question_text = question_payload.get("question", "").strip()
    labels = vqa_entry.get("labels", [])
    answer_index = vqa_entry.get("answer_index")
    image_paths = vqa_entry.get("image_paths", []) or []
    letters = list(string.ascii_uppercase)

    # regex to check if in the label we have an image
    pattern = re.compile(r"^\d{6}$")

    images_in_labels = 0
    for idx_img, label in enumerate(labels):
        if pattern.match(label):
            # do a smart replacement
            # I think there is no need to save the image again, just use the existing one
            # new_image_path = simulation_path.rsplit("/", 1)[0] + f"render/{label}.png"
            # image_paths.append(new_image_path)
            labels[idx_img] = "<image>"
            images_in_labels += 1

    # add <image> tags in place of images
    # locking in question images before adding other images in the question
    # slop code, but guess I need to speed up
    formatted_question = question_text
    formatted_question = (
        "".join(["<image>" for _ in range(len(image_paths) - images_in_labels)])
        + "\n"
        + formatted_question
    )

    option_letters = [letters[i] for i in range(min(len(labels), len(letters)))]
    option_lines = []
    for letter, label in zip(option_letters, labels):
        option_lines.append(f"{letter}. {label}")

    if option_lines:
        formatted_question = f"{formatted_question}\n" + "\n".join(option_lines)

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
        "choice_type": question_payload["choice"],
        "question_id": vqa_entry.get("question_key", ""),
        "category": vqa_entry["category"],
        "sub_category": question_payload.get("sub_category"),
    }

    answer_record = {
        "idx": idx,
        "answer": answer_letter,
        "task_type": "factual",
        "mode": question_record["mode"],
        "choice_type": question_payload["choice"],
    }

    return question_record, answer_record
