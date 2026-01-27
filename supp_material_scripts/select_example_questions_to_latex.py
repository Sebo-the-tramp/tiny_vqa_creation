#!/usr/bin/env python
import argparse
import json
import os
import random
import re
import shutil

# Global configuration
# Set a seed for reproducibility if desired (None = random every run).
RANDOM_SEED = None

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


CATEGORY_QID_RULES = {
    "collision": lambda qid: bool(qid) and qid.startswith("F_COLLISION"),
    "layout": lambda qid: qid == "F_LAYOUT_POSITION_OBJECT_OBJECT",
    "biggest_volume": lambda qid: qid == "F_SIZE_OBJECT_BIGGER",
    "young_modulus": lambda qid: bool(qid) and "YOUNG_MODULUS" in qid,
}


def select_examples(data, suffix: str, target_count: int = 4):
    selected = []
    selected_idxs = set()
    for category, rule in CATEGORY_QID_RULES.items():
        candidates = []
        for item in data:
            idx = item.get("idx", "")
            if not idx.endswith(suffix):
                continue
            qid = item.get("question_id")
            img_list = item.get("file_name", []) or []
            img_path = img_list[0] if img_list else None
            if rule(qid) and img_path and os.path.exists(img_path):
                candidates.append(item)
        if not candidates:
            print(f"Warning: no candidates found for category '{category}'.")
            continue
        pick = random.choice(candidates)
        if pick.get("idx") in selected_idxs:
            continue
        selected.append(pick)
        selected_idxs.add(pick.get("idx"))
        if len(selected) >= target_count:
            return selected
    if len(selected) >= target_count:
        return selected
    # Fill remaining slots with any valid items (existing image), regardless of category
    fallback = []
    for item in data:
        idx = item.get("idx", "")
        if not idx.endswith(suffix) or idx in selected_idxs:
            continue
        img_list = item.get("file_name", []) or []
        img_path = img_list[0] if img_list else None
        if img_path and os.path.exists(img_path):
            fallback.append(item)
    random.shuffle(fallback)
    for item in fallback:
        selected.append(item)
        if len(selected) >= target_count:
            break
    return selected


def copy_images(selected, out_dir):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    copied = {}
    for item in selected:
        idx = item.get("idx", "unknown")
        img_list = item.get("file_name", []) or []
        if not img_list:
            continue
        img_path = img_list[0]
        if not os.path.exists(img_path):
            continue
        base = os.path.basename(img_path)
        dst = os.path.join(out_dir, f"{idx}_{base}")
        shutil.copy2(img_path, dst)
        copied[idx] = dst
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
    lines = []
    for item in selected:
        question, choices = parse_question_and_choices(item.get("question", ""))
        idx = item.get("idx", "")
        img_path = copied_images.get(idx)
        if not img_path:
            continue
        img_basename = os.path.basename(img_path)
        images_subdir = images_subdir.strip("/").rstrip("/")
        latex_img_path = f"{images_subdir}/{img_basename}"

        correct_letter = ans_map.get(idx, "").strip().upper()
        formatted = {}
        for letter in ("A", "B", "C", "D"):
            choice = latex_escape(choices[letter])
            if letter == correct_letter:
                formatted[letter] = "\\correct{" + choice + "}"
            else:
                formatted[letter] = choice

        lines.append(
            "\\includegraphics[width=\\linewidth, height=0.562\\linewidth]{"
            + latex_escape(latex_img_path)
            + "}"
        )
        lines.append(
            "\\vqa{"
            + latex_escape(question)
            + "}{"
            + formatted["A"]
            + "}{"
            + formatted["B"]
            + "}{"
            + formatted["C"]
            + "}{"
            + formatted["D"]
            + "}"
        )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Select one random example for collision, layout, biggest_volume, and "
            "young_modulus and output LaTeX."
        )
    )
    parser.add_argument("--input_json", default=DEFAULT_INPUT_JSON)
    parser.add_argument("--answer_json", default=DEFAULT_ANSWER_JSON)
    parser.add_argument("--suffix", default="_i", choices=["_i", "_g"])
    parser.add_argument("--out_latex", default=None)
    parser.add_argument("--out_images", default=None)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if args.out_latex is None:
        args.out_latex = "example_questions.tex"
    if args.out_images is None:
        args.out_images = "example_images"

    with open(args.input_json, "r") as f:
        data = json.load(f)

    ans_map = load_answers_map(args.answer_json)
    selected = select_examples(data, args.suffix)

    copied_images = copy_images(selected, args.out_images)
    latex = build_latex(selected, ans_map, copied_images, args.out_images)

    with open(args.out_latex, "w") as f:
        f.write(latex)

    print(f"Selected {len(selected)} examples with suffix {args.suffix}.")
    print(f"LaTeX saved to: {args.out_latex}")
    print(f"Images copied to: {args.out_images}")


if __name__ == "__main__":
    main()
