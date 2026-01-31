#!/usr/bin/env python3
"""Subsample questions with uniform coverage over question_id per sub_category."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List

from subsample_questions_balanced import MISSING_TOKEN, load_questions


class AllocationError(RuntimeError):
    """Raised when a requested allocation cannot be satisfied."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a subset where each sub_category contributes at least N samples, "
            "allocating counts as uniformly as possible across question_id."
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
        "--count-per-sub-category",
        "--count_per_sub_category",
        dest="count_per_sub_category",
        type=int,
        default=2000,
        help="Number of samples to take for each sub_category (default: 2000).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        help="Optional filter so only questions with this mode are eligible (e.g., 'general').",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling.",
    )
    return parser.parse_args()


def resolve_question_id(record: dict[str, Any]) -> str:
    for key in ("question_id", "question_key", "idx"):
        value = record.get(key)
        if value not in {None, ""}:
            return str(value)
    return MISSING_TOKEN


def group_by_sub_category(
    questions: Iterable[dict[str, Any]],
) -> DefaultDict[str, List[dict[str, Any]]]:
    grouped: DefaultDict[str, List[dict[str, Any]]] = defaultdict(list)
    for record in questions:
        sub_category = record.get("sub_category")
        if sub_category in {None, ""}:
            sub_category = MISSING_TOKEN
        grouped[str(sub_category)].append(record)
    return grouped


def group_by_question_id(
    records: Iterable[dict[str, Any]],
) -> DefaultDict[str, List[dict[str, Any]]]:
    grouped: DefaultDict[str, List[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[resolve_question_id(record)].append(record)
    return grouped


def allocate_evenly_allow_zero(
    grouped: Dict[str, List[dict[str, Any]]],
    total: int,
    rng: random.Random,
) -> Dict[str, int]:
    keys = list(grouped.keys())
    if not keys:
        raise AllocationError("No groups available for allocation.")
    if total <= 0:
        raise AllocationError("Requested allocation size must be positive.")

    capacities = {key: len(grouped[key]) for key in keys}
    available = sum(capacities.values())
    if total > available:
        raise AllocationError(
            f"Requested {total} samples but only {available} available across groups."
        )

    ideal = total / float(len(keys))
    allocations = {key: min(int(ideal), capacities[key]) for key in keys}
    assigned = sum(allocations.values())
    remainder = total - assigned

    while remainder > 0:
        candidates = [key for key in keys if allocations[key] < capacities[key]]
        if not candidates:
            raise AllocationError("Ran out of capacity while distributing the remainder.")
        rng.shuffle(candidates)
        candidates.sort(
            key=lambda key: (
                ideal - allocations[key],
                capacities[key] - allocations[key],
            ),
            reverse=True,
        )
        allocations[candidates[0]] += 1
        remainder -= 1

    return allocations


def sample_sub_category(
    records: List[dict[str, Any]],
    count_per_sub_category: int,
    rng: random.Random,
) -> tuple[List[dict[str, Any]], Counter[str]]:
    buckets = group_by_question_id(records)
    allocations = allocate_evenly_allow_zero(buckets, count_per_sub_category, rng)

    sampled: List[dict[str, Any]] = []
    question_id_counts: Counter[str] = Counter()
    for question_id, count in allocations.items():
        if count <= 0:
            continue
        group = buckets[question_id]
        if count == len(group):
            chosen = list(group)
        else:
            chosen = rng.sample(group, count)
        sampled.extend(chosen)
        question_id_counts[question_id] += len(chosen)

    rng.shuffle(sampled)
    return sampled, question_id_counts


def print_summary(
    summary: Dict[str, Counter[str]],
    totals: Dict[str, int],
    available_question_ids: Dict[str, int],
) -> None:
    if not summary:
        print("No summary stats available.")
        return

    print("\nSummary by sub_category (uniform over question_id):")
    for sub_category in sorted(summary.keys()):
        counts = summary[sub_category]
        total = totals[sub_category]
        available_ids = available_question_ids[sub_category]
        used_ids = len([qid for qid, count in counts.items() if count > 0])
        min_count = min(counts.values()) if counts else 0
        max_count = max(counts.values()) if counts else 0
        print(f"---- {sub_category.upper()} ----")
        print(f"total: {total}")
        print(f"question_ids available: {available_ids}")
        print(f"question_ids sampled: {used_ids}")
        print(f"question_id count range: {min_count}..{max_count}")


def main() -> None:
    args = parse_args()
    if args.count_per_sub_category <= 0:
        raise SystemExit("Requested count_per_sub_category must be positive.")

    questions = load_questions(args.input)
    if args.mode is not None:
        questions = [record for record in questions if record.get("mode") == args.mode]

    if not questions:
        raise SystemExit("No records left after applying the requested filters.")

    rng = random.Random(args.seed)
    grouped = group_by_sub_category(questions)

    sampled: List[dict[str, Any]] = []
    summary: Dict[str, Counter[str]] = {}
    totals: Dict[str, int] = {}
    available_question_ids: Dict[str, int] = {}

    for sub_category, records in grouped.items():
        if len(records) < args.count_per_sub_category:
            raise SystemExit(
                f"Sub-category '{sub_category}' only has {len(records)} records, "
                f"cannot reach {args.count_per_sub_category}."
            )
        available_question_ids[sub_category] = len(group_by_question_id(records))
        chosen, counts = sample_sub_category(
            records=records,
            count_per_sub_category=args.count_per_sub_category,
            rng=rng,
        )
        sampled.extend(chosen)
        summary[sub_category] = counts
        totals[sub_category] = len(chosen)

    rng.shuffle(sampled)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(sampled, handle, indent=4)

    print_summary(summary, totals, available_question_ids)
    print(f"\nTotal records overall: {len(sampled)}")


if __name__ == "__main__":
    main()
