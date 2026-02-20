#!/usr/bin/env python3
"""Verify that every model prediction has a corresponding entry in val_answer_run_28_general.json."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
import re


IDX_RE = re.compile(r"^(\d+)(?:_(.*))?$")


def parse_suffix(idx: str) -> str:
    m = IDX_RE.match(idx)
    if not m:
        return ""
    return m.group(2) or ""


def load_answer_index(path: Path) -> Tuple[Set[str], Dict[str, Set[str]]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data).__name__}")
    idx_set: Set[str] = set()
    by_suffix: Dict[str, Set[str]] = {}
    missing = 0
    for item in data:
        if not isinstance(item, dict):
            continue
        idx = item.get("idx")
        if idx is None:
            missing += 1
            continue
        idx_set.add(idx)
        suffix = parse_suffix(idx)
        by_suffix.setdefault(suffix, set()).add(idx)
    if missing:
        print(f"Warning: {missing} items in answer file missing 'idx'")
    return idx_set, by_suffix


def list_model_files(results_dir: Path) -> List[Path]:
    return sorted([p for p in results_dir.iterdir() if p.is_file() and p.suffix == ".json"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Check that each model prediction has an answer.")
    parser.add_argument(
        "--answers",
        default="/Users/sebastiancavada/Desktop/tmp_paris/tiny_vqa_creation/output/run_28_general/val_answer_run_28_general.json",
        help="Path to val_answer_run_28_general.json",
    )
    parser.add_argument(
        "--results",
        default="/Users/sebastiancavada/Desktop/tmp_paris/tiny_vqa_creation/output/run_28_general/results_run_28_general-all",
        help="Directory containing merged model result JSON files",
    )
    args = parser.parse_args()

    answers_path = Path(args.answers)
    results_dir = Path(args.results)

    if not answers_path.exists():
        print(f"Answer file not found: {answers_path}")
        return 1
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return 1

    answer_idx, answer_idx_by_suffix = load_answer_index(answers_path)
    model_files = list_model_files(results_dir)

    if not model_files:
        print(f"No model .json files found in {results_dir}")
        return 1

    print(f"Loaded {len(answer_idx)} answer idx values")
    print(f"Checking {len(model_files)} model files in {results_dir}")
    answers_total = len(answer_idx)
    print(
        "Per model: preds_with_answer/preds_total (preds_missing_answer) | "
        "answers_missing_pred_all/answers_total | answers_missing_pred_in_suffix/answers_in_suffix"
    )

    any_missing = False
    missing_subset_models: List[Tuple[str, int, int]] = []

    for model_path in model_files:
        with model_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected list in {model_path}, got {type(data).__name__}")

        missing = []
        total = 0
        model_idx_set: Set[str] = set()
        model_suffix_counts: Dict[str, int] = {}
        for item in data:
            if not isinstance(item, dict):
                continue
            idx = item.get("idx")
            if idx is None:
                continue
            total += 1
            model_idx_set.add(idx)
            suffix = parse_suffix(idx)
            model_suffix_counts[suffix] = model_suffix_counts.get(suffix, 0) + 1
            if idx not in answer_idx:
                missing.append(idx)

        answered = total - len(missing)
        missing_predictions_all = answers_total - len(model_idx_set & answer_idx)
        # Only compare against the answer subset that matches the model's suffixes (e.g., _i vs _g).
        model_suffixes = set(model_suffix_counts.keys())
        answer_subset: Set[str] = set()
        for sfx in model_suffixes:
            answer_subset |= answer_idx_by_suffix.get(sfx, set())
        answers_subset_total = len(answer_subset)
        missing_predictions_subset = answers_subset_total - len(model_idx_set & answer_subset)
        if missing:
            any_missing = True
        print(
            f"- {model_path.name}: {answered}/{total} ({len(missing)} missing) | "
            f"{missing_predictions_all}/{answers_total} | "
            f"{missing_predictions_subset}/{answers_subset_total}"
        )
        if missing:
            print(f"  sample: {missing[:10]}")
        if missing_predictions_all and answers_subset_total == answers_total:
            # show a small sample of answer idx not predicted by this model
            from itertools import islice

            missing_pred_sample = list(islice((idx for idx in answer_idx if idx not in model_idx_set), 10))
            print(f"  missing_pred_sample_all: {missing_pred_sample}")
        if missing_predictions_subset:
            from itertools import islice

            missing_pred_subset_sample = list(
                islice((idx for idx in answer_subset if idx not in model_idx_set), 10)
            )
            print(f"  missing_pred_sample_in_suffix: {missing_pred_subset_sample}")
            missing_subset_models.append(
                (model_path.name, missing_predictions_subset, answers_subset_total)
            )
        if model_suffix_counts:
            suffix_summary = ", ".join(f"{k or 'no_suffix'}={v}" for k, v in sorted(model_suffix_counts.items()))
            print(f"  model_suffix_counts: {suffix_summary}")

    if not any_missing:
        print("All model predictions have corresponding answers.")
    if missing_subset_models:
        print("Models with missing predictions within their suffix subset:")
        for name, miss, total in missing_subset_models:
            print(f"- {name}: {miss}/{total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
