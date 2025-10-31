#!/usr/bin/env python3
"""Utility to subsample a VQA-style JSON file."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly subsample questions from a JSON file, optionally filtering by mode."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("test.json"),
        help="Path to the source JSON file (default: test.json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path where the subsampled JSON will be written.",
    )
    parser.add_argument(
        "--count",
        type=int,
        required=True,
        help=(
            "Number of index groups to sample. Entries sharing the same base index "
            "(e.g., '12_g' and '12_i') are kept together."
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        help="Filter questions by this mode before sampling (e.g., 'general', 'image-only').",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling.",
    )
    return parser.parse_args()


def load_questions(path: Path) -> List[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError as exc:
        raise SystemExit(f"Input file '{path}' does not exist.") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Failed to parse JSON from '{path}': {exc}") from exc

    if not isinstance(data, list):
        raise SystemExit(f"Expected a list of questions in '{path}', found {type(data).__name__}.")

    return data


def group_by_index_suffix(
    questions: Iterable[dict[str, Any]]
) -> Dict[str, List[dict[str, Any]]]:
    grouped: Dict[str, List[dict[str, Any]]] = {}
    for record in questions:
        raw_idx = record.get("idx")
        if raw_idx is None:
            raise SystemExit("Found a question without an 'idx' field; cannot group by index.")

        key = normalise_index(raw_idx)
        grouped.setdefault(key, []).append(record)
    return grouped


def normalise_index(value: Any) -> str:
    if isinstance(value, str):
        base, _, suffix = value.rpartition("_")
        if base and suffix in {"g", "i"}:
            return base
        return value
    return str(value)


def main() -> None:
    args = parse_args()
    questions = load_questions(args.input)

    if args.mode is not None:
        questions = [record for record in questions if record.get("mode") == args.mode]

    grouped_questions = group_by_index_suffix(questions)

    if len(grouped_questions) < args.count:
        raise SystemExit(
            f"Requested {args.count} index groups but only {len(grouped_questions)} available "
            f"after applying filters."
        )

    rng = random.Random(args.seed)
    sampled_keys = rng.sample(list(grouped_questions.keys()), args.count)
    sampled = [item for key in sampled_keys for item in grouped_questions[key]]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(sampled, handle, indent=4)


if __name__ == "__main__":
    main()
