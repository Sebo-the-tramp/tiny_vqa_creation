#!/usr/bin/env python3
"""Utility to subsample a VQA-style JSON file with stratified sampling."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Sequence, Tuple

MISSING_TOKEN = "<MISSING>"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stratified subsampling of questions from a JSON file."
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
        help="Total number of records to sample (across all modes).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        help="Filter questions by this mode before sampling (e.g., 'general', 'image-only').",
    )
    parser.add_argument(
        "--balance-on",
        nargs="+",
        default=["sub_category"],
        metavar="FIELD",
        help=(
            "One or more fields to balance on. Sampling is stratified across the unique "
            "combinations for these fields (default: sub_category). Use '-' to disable balancing."
        ),
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


# ----------------------------
# Stratified sampling helpers
# ----------------------------

def make_balance_groups(
    questions: Iterable[dict[str, Any]], fields: Sequence[str]
) -> DefaultDict[Tuple[Any, ...], List[dict[str, Any]]]:
    grouped: DefaultDict[Tuple[Any, ...], List[dict[str, Any]]] = defaultdict(list)

    if not fields or fields == ["-"]:
        grouped[()] = list(questions)
        return grouped

    for record in questions:
        key_components: List[Any] = []
        for field in fields:
            if field not in record:
                raise SystemExit(f"Cannot balance on '{field}' because it is missing from a record.")
            value = record[field]
            key_components.append(MISSING_TOKEN if value in {None, ""} else value)
        grouped[tuple(key_components)].append(record)
    return grouped


def allocate_evenly(
    grouped: Dict[Tuple[Any, ...], List[dict[str, Any]]],
    total: int,
    rng: random.Random,
) -> Dict[Tuple[Any, ...], int]:
    keys = list(grouped.keys())
    num_groups = len(keys)
    if num_groups == 0:
        raise SystemExit("No data available after filtering; nothing to sample.")
    if total < num_groups:
        raise SystemExit(
            f"Requested {total} samples but need at least {num_groups} to cover each balance group once."
        )

    capacities = {key: len(grouped[key]) for key in keys}
    available_total = sum(capacities.values())
    if total > available_total:
        raise SystemExit(
            f"Requested {total} samples but only {available_total} available across balance groups."
        )

    ideal = total / float(num_groups)
    allocations = {key: min(int(ideal), capacities[key]) for key in keys}
    assigned = sum(allocations.values())
    remainder = total - assigned

    if remainder < 0:
        raise AssertionError("Allocation produced more samples than requested.")

    while remainder:
        candidates = [key for key in keys if allocations[key] < capacities[key]]
        if not candidates:
            raise SystemExit("Ran out of capacity while allocating stratified samples.")
        rng.shuffle(candidates)
        candidates.sort(
            key=lambda key: (ideal - allocations[key], capacities[key] - allocations[key]),
            reverse=True,
        )
        chosen = candidates[0]
        allocations[chosen] += 1
        remainder -= 1

    return allocations


def stratified_sample(
    questions: Iterable[dict[str, Any]],
    total: int,
    balance_fields: Sequence[str],
    rng: random.Random,
) -> List[dict[str, Any]]:
    questions = list(questions)
    if total <= 0:
        raise SystemExit("Requested sample size must be positive.")
    if total > len(questions):
        raise SystemExit(
            f"Requested {total} samples but dataset only contains {len(questions)} records."
        )

    grouped = make_balance_groups(questions, balance_fields)
    allocations = allocate_evenly(grouped, total, rng)

    sampled: List[dict[str, Any]] = []
    for key, group in grouped.items():
        required = allocations[key]
        if required == 0:
            continue
        sampled.extend(rng.sample(group, required))

    if len(sampled) != total:
        raise AssertionError(
            f"Expected to collect {total} samples but got {len(sampled)} after allocation."
        )

    rng.shuffle(sampled)
    return sampled


def main() -> None:
    args = parse_args()
    questions = load_questions(args.input)

    if args.mode is not None:
        questions = [record for record in questions if record.get("mode") == args.mode]

    rng = random.Random(args.seed)
    balance_fields: Sequence[str] = args.balance_on

    sampled = stratified_sample(
        questions=questions,
        total=args.count,
        balance_fields=balance_fields,
        rng=rng,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(sampled, handle, indent=4)

    if balance_fields and balance_fields != ["-"]:
        grouped_sample = make_balance_groups(sampled, balance_fields)
        print(
            f"Sampled {len(sampled)} records across {len(grouped_sample)} balance groups "
            f"using fields {', '.join(balance_fields)}."
        )
        total_overall = 0
        for key, group in grouped_sample.items():
            label = ", ".join(str(part) for part in key) if key else "<all>"
            print(f"  {label}: {len(group)}")
            total_overall += len(group)
        print("Total records overall:", total_overall)

    else:
        print(f"Sampled {len(sampled)} records (no balancing applied).")


if __name__ == "__main__":
    main()
