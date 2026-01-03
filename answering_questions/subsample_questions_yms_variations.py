#!/usr/bin/env python3
"""Subsample questions balanced across yms-variations (stiff/medium/soft)."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from subsample_questions_balanced import allocate_evenly, load_questions


VARIATIONS = ("stiff", "medium", "soft")
VARIATION_PATTERN = re.compile(r"/yms-variations/(?P<var>stiff|medium|soft)/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a balanced subsample across yms-variations (stiff/medium/soft) "
            "from a VQA-style JSON file."
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
        help="Where to write the subsampled JSON.",
    )
    parser.add_argument(
        "--total",
        type=int,
        default=10_000,
        help="Total number of records to sample (default: 10000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--drop-missing",
        action="store_true",
        help=(
            "Drop records where the yms-variation cannot be inferred. "
            "By default this is treated as an error."
        ),
    )
    parser.add_argument(
        "--require-all",
        action="store_true",
        help=(
            "Fail if any of stiff/medium/soft are missing from the dataset. "
            "By default the script balances across whichever variations are present."
        ),
    )
    return parser.parse_args()


def extract_variation(record: dict[str, Any]) -> Optional[str]:
    for key in ("yms_variation", "yms-variation", "yms_variations", "variation"):
        value = record.get(key)
        if value in VARIATIONS:
            return value

    paths: List[str] = []
    simulation_id = record.get("simulation_id")
    if isinstance(simulation_id, str):
        paths.append(simulation_id)

    file_name = record.get("file_name")
    if isinstance(file_name, str):
        paths.append(file_name)
    elif isinstance(file_name, list):
        paths.extend([item for item in file_name if isinstance(item, str)])

    for path in paths:
        match = VARIATION_PATTERN.search(path)
        if match:
            return match.group("var")

    return None


def group_by_variation(
    records: Iterable[dict[str, Any]],
    drop_missing: bool,
) -> Dict[str, List[dict[str, Any]]]:
    grouped: Dict[str, List[dict[str, Any]]] = {variation: [] for variation in VARIATIONS}
    missing = 0

    for record in records:
        variation = extract_variation(record)
        if variation is None:
            missing += 1
            if drop_missing:
                continue
            raise SystemExit(
                "Failed to infer yms-variation for at least one record. "
                "Use --drop-missing to skip them."
            )
        grouped[variation].append(record)

    if missing and drop_missing:
        print(f"Dropped {missing} records with unknown yms-variation.")

    return grouped


def main() -> None:
    args = parse_args()
    questions = load_questions(args.input)

    grouped_all = group_by_variation(questions, drop_missing=args.drop_missing)
    grouped = {key: value for key, value in grouped_all.items() if value}

    if not grouped:
        raise SystemExit("No records found with a valid yms-variation.")

    if args.require_all and len(grouped) != len(VARIATIONS):
        missing = [var for var in VARIATIONS if var not in grouped]
        raise SystemExit(
            "Missing yms-variations required for balancing: " + ", ".join(missing)
        )

    rng = random.Random(args.seed)
    allocations = allocate_evenly(grouped, args.total, rng)

    sampled: List[dict[str, Any]] = []
    for variation, records in grouped.items():
        count = allocations.get(variation, 0)
        if count:
            sampled.extend(rng.sample(records, count))

    rng.shuffle(sampled)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(sampled, handle, indent=4)

    print(f"Sampled {len(sampled)} records across {len(grouped)} variations.")
    for variation in VARIATIONS:
        if variation in grouped:
            print(f"  {variation}: {allocations.get(variation, 0)}")
        else:
            print(f"  {variation}: 0 (missing)")


if __name__ == "__main__":
    main()
