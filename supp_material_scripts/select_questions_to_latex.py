#!/usr/bin/env python
import argparse
import json
import os
import random
import re
import shutil
from collections import defaultdict

# Global configuration
N_PER_QUESTION_ID = 2
RANDOM_SEED = 123

DEFAULT_INPUT_JSON = (
    "/data0/sebastian.cavada/compositional-physics/tiny_vqa_creation/output/"
    "run_24_general/test_run_24_general_karo_10K.json"
)
DEFAULT_ANSWER_JSON = (
    "/data0/sebastian.cavada/compositional-physics/tiny_vqa_creation/output/"
    "run_24_general/val_answer_run_24_general.json"
)

OBJ_COUNT_RE = re.compile(r"/random/(\d+)/")


def parse_obj_count(simulation_id: str) -> str | None:
    match = OBJ_COUNT_RE.search(simulation_id)
    if not match:
        return None
    return match.group(1)


def parse_question_and_choices(question_text: str):
    # Strip the <image> marker and split lines
    lines = [line.strip() for line in question_text.splitlines() if line.strip()]
    if lines and lines[0].lower().startswith("<image>"):
        lines = lines[1:]
    if not lines:
        return "", {"A": "", "B": "", "C": "", "D": ""}

    # First non-empty line is question; remaining lines are choices
    question = lines[0]
    choices = {"A": "", "B": "", "C": "", "D": ""}
    for line in lines[1:]:
        if len(line) >= 3 and line[1] == ".":
            key = line[0].upper()
            if key in choices:
                choices[key] = line[2:].strip()
    return question, choices


def load_answers_map(answer_json_path: str):
    with open(answer_json_path, "r") as f:
        answers = json.load(f)
    ans_map = {}
    for item in answers:
        idx = item.get("idx")
        answer = item.get("answer")
        if idx is not None:
            ans_map[idx] = answer
    return ans_map


def select_examples(data, suffix: str, n_per_qid: int):
    grouped = defaultdict(list)
    for item in data:
        idx = item.get("idx", "")
        if not idx.endswith(suffix):
            continue
        grouped[item.get("question_id")].append(item)

    selected = []
    for qid, items in grouped.items():
        random.shuffle(items)
        used_counts = set()
        for item in items:
            obj_count = parse_obj_count(item.get("simulation_id", ""))
            if obj_count is None or obj_count in used_counts:
                continue
            used_counts.add(obj_count)
            selected.append(item)
            if len(used_counts) >= n_per_qid:
                break
    return selected


def copy_images(selected, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    copied = {}
    for item in selected:
        idx = item.get("idx", "unknown")
        img_list = item.get("file_name", []) or []
        if not img_list:
            continue
        copied_paths = []
        for img_path in img_list:
            if not os.path.exists(img_path):
                continue
            base = os.path.basename(img_path)
            dst = os.path.join(out_dir, f"{idx}_{base}")
            shutil.copy2(img_path, dst)
            copied_paths.append(dst)
        if copied_paths:
            copied[idx] = copied_paths
    return copied


def latex_escape(text: str) -> str:
    # Minimal LaTeX escaping for common special characters
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in text)


def build_latex(selected, ans_map, copied_images, images_subdir):
    # Import the macro directly to avoid duplication
    from create_questions_examples import LATEX_MACRO

    lines = [LATEX_MACRO.strip(), ""]
    for item in selected:
        question, choices = parse_question_and_choices(item.get("question", ""))
        idx = item.get("idx", "")
        img_paths = copied_images.get(idx, [])
        if not img_paths:
            continue
        img_path = img_paths[0]
        img_basename = os.path.basename(img_path)
        images_subdir = images_subdir.strip("/").rstrip("/")
        latex_img_path = f"figures/{images_subdir}/{img_basename}"
        lines.append(
            "\\vqaBlock{"
            + latex_escape(latex_img_path)
            + "}{"
            + latex_escape(question)
            + "}{"
            + latex_escape(choices["A"])
            + "}{"
            + latex_escape(choices["B"])
            + "}{"
            + latex_escape(choices["C"])
            + "}{"
            + latex_escape(choices["D"])
            + "}"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Select N examples per question_id with unique object counts and output LaTeX."
    )
    parser.add_argument("--input_json", default=DEFAULT_INPUT_JSON)
    parser.add_argument("--answer_json", default=DEFAULT_ANSWER_JSON)
    parser.add_argument("--suffix", default="_i", choices=["_i", "_g"])
    parser.add_argument("--out_latex", default=None)
    parser.add_argument("--out_images", default=None)
    args = parser.parse_args()

    random.seed(RANDOM_SEED)

    if args.out_latex is None:
        args.out_latex = f"selected_questions{args.suffix}.tex"
    if args.out_images is None:
        args.out_images = f"selected_images{args.suffix}"

    with open(args.input_json, "r") as f:
        data = json.load(f)

    ans_map = load_answers_map(args.answer_json)
    selected = select_examples(data, args.suffix, N_PER_QUESTION_ID)

    copied_images = copy_images(selected, args.out_images)
    latex = build_latex(selected, ans_map, copied_images, args.out_images)

    with open(args.out_latex, "w") as f:
        f.write(latex)

    print(f"Selected {len(selected)} examples with suffix {args.suffix}.")
    print(f"LaTeX saved to: {args.out_latex}")
    print(f"Images copied to: {args.out_images}")


if __name__ == "__main__":
    main()
