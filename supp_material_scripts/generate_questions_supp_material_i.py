#!/usr/bin/env python3
import json
import shutil
from pathlib import Path

QUESTIONS_IDX_PATH = Path("questions_supp_material_idx.json")
TEST_RUN_PATH = Path("../output/run_24_general/test_run_24_general.json")
ANSWER_PATH = Path("../output/run_24_general/val_answer_run_24_general.json")

OUT_DIR = Path("upload")
OUT_TEX = Path("questions_supp_material_i.tex")


def normalize_text(text: str) -> str:
    repl = {
        "\u2019": "'",
        "\u2018": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2011": "-",
        "\u00a0": " ",
    }
    for k, v in repl.items():
        text = text.replace(k, v)
    return text


def latex_escape(text: str) -> str:
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


def parse_question_and_choices(question_text: str):
    lines = [line.strip() for line in question_text.splitlines() if line.strip()]
    if lines and lines[0].lower().startswith("<image>"):
        lines = lines[1:]
    if not lines:
        return "", {"A": "", "B": "", "C": "", "D": ""}

    question = lines[0]
    choices = {"A": "", "B": "", "C": "", "D": ""}
    for line in lines[1:]:
        if len(line) >= 3 and line[1] == ".":
            key = line[0].upper()
            if key in choices:
                choices[key] = line[2:].strip()
    return question, choices


def main():
    questions_idx = json.loads(QUESTIONS_IDX_PATH.read_text())
    test_data = json.loads(TEST_RUN_PATH.read_text())
    answers = json.loads(ANSWER_PATH.read_text())

    idx_list = []
    seen = set()
    for cat in questions_idx.values():
        if not isinstance(cat, dict):
            continue
        for _, buckets in cat.items():
            if not isinstance(buckets, dict):
                continue
            for idx in buckets.get("i", []) or []:
                if idx not in seen:
                    seen.add(idx)
                    idx_list.append(idx)

    by_idx = {
        item.get("idx"): item
        for item in test_data
        if isinstance(item, dict) and "idx" in item
    }
    ans_map = {
        item.get("idx"): item.get("answer")
        for item in answers
        if isinstance(item, dict)
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    lines = []
    missing = []
    missing_images = []
    missing_answers = []

    for idx in idx_list:
        item = by_idx.get(idx)
        if not item:
            missing.append(idx)
            continue

        question_text = item.get("question", "")
        question, choices = parse_question_and_choices(question_text)
        question = normalize_text(question)
        for k in list(choices.keys()):
            choices[k] = normalize_text(choices[k])

        correct_key = ans_map.get(idx)
        if correct_key not in choices:
            missing_answers.append(idx)
            correct_key = None

        img_list = item.get("file_name") or []
        if not img_list:
            missing_images.append(idx)
            continue
        img_path = Path(img_list[0])
        if not img_path.exists():
            missing_images.append(idx)
            continue

        dst_name = f"{idx}_{img_path.name}"
        dst_path = OUT_DIR / dst_name
        shutil.copy2(img_path, dst_path)

        latex_img_path = f"upload/{dst_name}"

        def fmt_choice(key: str) -> str:
            text = latex_escape(choices[key])
            if correct_key == key:
                return r"\correct{" + text + "}"
            return text

        lines.append(
            "\\vqaRow{"
            + latex_escape(latex_img_path)
            + "}\n"
            + "       {"
            + latex_escape(question)
            + "}\n"
            + "       {"
            + fmt_choice("A")
            + "}"
            + "{"
            + fmt_choice("B")
            + "}"
            + "{"
            + fmt_choice("C")
            + "}"
            + "{"
            + fmt_choice("D")
            + "}"
        )

    OUT_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Collected idx: {len(idx_list)}")
    print(f"Wrote LaTeX: {OUT_TEX}")
    print(f"Images copied to: {OUT_DIR}")
    print(f"Missing idx: {len(missing)}")
    print(f"Missing images: {len(missing_images)}")
    print(f"Missing answers: {len(missing_answers)}")
    if missing[:5]:
        print("Missing idx examples:", missing[:5])
    if missing_images[:5]:
        print("Missing image examples:", missing_images[:5])
    if missing_answers[:5]:
        print("Missing answer examples:", missing_answers[:5])


if __name__ == "__main__":
    main()
