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
        question_idx = entry.get("idx")
        template = templates.get(question_id)
        if not template:
            if keep_original:
                expanded.append(entry)
            missing_templates += 1
            continue

        #filter non _general questions
        if question_idx is not None and not str(question_idx).endswith("_g"):
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

def _base_question_id(question_id: str) -> str:
    if "_level_" in question_id:
        return question_id.split("_level_", 1)[0]
    return question_id


def _uniform_sample(items: List[dict], target: int) -> List[dict]:
    if len(items) <= target:
        return items
    step = len(items) / target
    indices = [int(i * step) for i in range(target)]
    return [items[i] for i in indices]


def balance_entries(entries: List[dict], templates) -> List[dict]:
    max_per_question = 100
    list_questions = list(templates.keys())

    buckets: Dict[str, List[dict]] = {qid: [] for qid in list_questions}
    for entry in entries:
        question_id = entry.get("question_id")
        if not question_id:
            continue
        base_id = _base_question_id(str(question_id))
        if base_id in buckets:
            buckets[base_id].append(entry)

    counts = {qid: len(items) for qid, items in buckets.items()}
    if not counts:
        print("No entries matched template question ids; skipping balancing.")
        return entries

    min_count = min(counts.values())
    target = min(min_count, max_per_question)
    if target == 0:
        print("No entries available to balance; skipping balancing.")
        return []

    balanced: List[dict] = []
    for question_id in list_questions:
        items = buckets.get(question_id, [])
        print(f"Question ID {question_id} has {len(items)} entries before balancing.")
        to_add = _uniform_sample(items, target)
        balanced.extend(to_add)

    print(f"Balanced to {target} per question id; total {len(balanced)} entries.")
    return balanced


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

    balanced_entries = balance_entries(entries, templates)

    print(len(balanced_entries))

    expanded_entries = expand_questions(
        balanced_entries, templates, args.keep_original, args.max_questions
    )

    balanced_entries = balance_entries(expanded_entries, templates)

    output_path = compute_output_path(input_path, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(balanced_entries, handle, indent=2)

    total_original = len(entries)
    total_expanded = len(balanced_entries)

    print(
        f"Wrote {total_expanded} questions (from {total_original} originals) "
        f"to {output_path}"
    )


if __name__ == "__main__":
    main()
