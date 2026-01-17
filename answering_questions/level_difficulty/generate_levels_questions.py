#!/usr/bin/env python3
"""
Utility for generating leveled variants of TinyVQA questions.

Given a question file inside one of the run folders in ../output it looks for
question ids that exist in questions.json (located next to this script) and, for
each of those ids, emits leveled copies of the question targeted at a baseline
and five different reading levels. The new entries reuse the original metadata
but replace `question` and `question_id`.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, Iterable, List

# The levels are ordered to keep the output deterministic.
BASELINE_LEVEL = "baseline"
LEVEL_KEYS = [BASELINE_LEVEL, "child", "teen", "undegrad", "graduate", "expert"]


def load_questions(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    questions = payload.get("questions")
    if not isinstance(questions, dict):
        raise ValueError(f"{path} does not contain a 'questions' mapping.")
    return questions


def compute_output_path(input_path: Path, override: Path | None) -> Path:
    if override:
        return override

    stem = input_path.stem
    suffix = input_path.suffix or ".json"

    general_idx = stem.find("_general")
    if general_idx != -1:
        base_name = stem[: general_idx + len("_general")]
    else:
        base_name = stem
    output_name = f"{base_name}_levels_difficulty{suffix}"
    return input_path.with_name(output_name)


def expand_questions(
    entries: Iterable[dict],
    templates: Dict[str, Dict[str, str]],
    keep_original: bool,
    max_questions: int | None,
) -> List[dict]:
    expanded: List[dict] = []
    missing_templates = 0
    generated_questions = 0
    for entry in entries:
        question_id = entry.get("question_id")
        template = templates.get(question_id)
        if not template:
            if keep_original:
                expanded.append(entry)
            missing_templates += 1
            continue

        if max_questions is not None and generated_questions >= max_questions:
            continue

        generated_questions += 1

        if keep_original:
            expanded.append(entry)

        og_question = template.get("og")

        for level_key in LEVEL_KEYS:
            if level_key == BASELINE_LEVEL:
                level_question = og_question
            else:
                level_question = template.get(level_key)
            if not level_question:
                continue
            new_entry = copy.deepcopy(entry)
            new_entry["question"] = new_entry["question"].replace(
                og_question, level_question
            )
            new_entry["question_id"] = f"{question_id}_level_{level_key}"
            if "idx" in new_entry and new_entry["idx"] is not None:
                new_entry["idx"] = f"{new_entry['idx']}_level_{level_key}"
            new_entry["difficulty_level"] = level_key
            expanded.append(new_entry)
    if missing_templates:
        print(
            f"Skipped {missing_templates} questions without templates "
            f"(use --keep-original to retain them)."
        )
    if max_questions is not None:
        print(f"Generated leveled variants for {generated_questions} questions.")
    return expanded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate leveled questions for a TinyVQA run file."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the test_run_* JSON file to be expanded.",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path(__file__).with_name("questions.json"),
        help="Path to the questions.json template file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for the generated JSON file. "
        "Defaults to {input}_levels_difficulty.json in the same folder.",
    )
    parser.add_argument(
        "--keep-original",
        action="store_true",
        help="Keep the original entries alongside the leveled versions.",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Limit the number of questions expanded (each produces up to 5 entries).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    questions_path = args.questions.expanduser().resolve()

    with input_path.open("r", encoding="utf-8") as handle:
        entries = json.load(handle)

    if not isinstance(entries, list):
        raise ValueError(f"{input_path} does not contain a JSON array.")

    templates = load_questions(questions_path)
    expanded_entries = expand_questions(
        entries, templates, args.keep_original, args.max_questions
    )

    output_path = compute_output_path(input_path, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(expanded_entries, handle, indent=2)

    total_original = len(entries)
    total_expanded = len(expanded_entries)

    print(
        f"Wrote {total_expanded} questions (from {total_original} originals) "
        f"to {output_path}"
    )


if __name__ == "__main__":
    main()
