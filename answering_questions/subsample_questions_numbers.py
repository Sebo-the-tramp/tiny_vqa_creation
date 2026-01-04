#!/usr/bin/env python3
"""Subsample questions with balanced sub-category and object-count coverage."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List

from subsample_questions_balanced import MISSING_TOKEN, load_questions

NUM_OBJECTS_PATTERN = re.compile(r"_no-(\d+)")


class AllocationError(RuntimeError):
    """Raised when a requested allocation cannot be satisfied."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a stratified subset where each sub_category contributes the same number of "
            "questions and each (sub_category, num_objects) pair receives roughly the same share."
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
        "--count",
        type=int,
        required=True,
        help="Total number of questions to keep across all sub_categories.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        help="Optional filter so only questions with this mode are eligible (e.g., 'general').",
    )
    parser.add_argument(
        "--pair-target",
        type=int,
        default=100,
        help=(
            "Ideal number of questions for each (sub_category, num_objects) pair. "
            "If the cap prevents meeting per-category targets it is relaxed automatically."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling.",
    )
    return parser.parse_args()


def derive_num_objects(simulation_id: Any, idx: Any) -> int:
    if not isinstance(simulation_id, str) or not simulation_id:
        raise SystemExit(
            f"Cannot derive 'num_objects' for record idx '{idx}': missing simulation_id field."
        )
    match = NUM_OBJECTS_PATTERN.search(simulation_id)
    if not match:
        raise SystemExit(
            f"Cannot derive 'num_objects' for record idx '{idx}': "
            "simulation_id does not contain '_no-<count>'."
        )
    value = int(match.group(1))
    if value <= 0:
        raise SystemExit(
            f"Derived 'num_objects' for record idx '{idx}' is invalid (value: {value})."
        )
    return value


def resolve_num_objects(record: dict[str, Any]) -> int:
    num_objects = record.get("num_objects")
    if num_objects in {None, ""}:
        num_objects = derive_num_objects(record.get("simulation_id"), record.get("idx"))
        record["num_objects"] = num_objects
    try:
        resolved = int(num_objects)
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            f"Unable to determine 'num_objects' for record idx '{record.get('idx')}': {exc}"
        ) from exc
    record["num_objects"] = resolved
    return resolved


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


def group_by_num_objects(
    records: Iterable[dict[str, Any]],
) -> DefaultDict[int, List[dict[str, Any]]]:
    grouped: DefaultDict[int, List[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[resolve_num_objects(record)].append(record)
    return grouped


def allocate_evenly(
    grouped: Dict[Any, List[dict[str, Any]]],
    total: int,
    rng: random.Random,
    per_key_cap: int | None = None,
) -> Dict[Any, int]:
    keys = list(grouped.keys())
    if not keys:
        raise AllocationError("No groups available for allocation.")
    if total <= 0:
        raise AllocationError("Requested allocation size must be positive.")
    if total < len(keys):
        raise AllocationError(
            f"Need at least {len(keys)} samples to allocate one per group, received {total}."
        )

    maxima: Dict[Any, int] = {}
    for key in keys:
        capacity = len(grouped[key])
        limit = capacity if per_key_cap is None else min(capacity, per_key_cap)
        if limit <= 0:
            raise AllocationError(f"Group '{key}' has no available records.")
        maxima[key] = limit

    available = sum(maxima.values())
    if total > available:
        raise AllocationError(
            f"Requested {total} samples but only {available} available across the selected groups."
        )

    ideal = total / float(len(keys))
    allocations: Dict[Any, int] = {}
    for key in keys:
        limit = maxima[key]
        allocations[key] = min(int(ideal), limit)

    assigned = sum(allocations.values())
    remainder = total - assigned

    while remainder > 0:
        candidates = [key for key in keys if allocations[key] < maxima[key]]
        if not candidates:
            raise AllocationError(
                "Ran out of capacity while distributing the remainder."
            )
        rng.shuffle(candidates)
        candidates.sort(
            key=lambda key: (
                ideal - allocations[key],
                maxima[key] - allocations[key],
            ),
            reverse=True,
        )
        allocations[candidates[0]] += 1
        remainder -= 1

    return allocations


def sample_within_sub_category(
    sub_category: str,
    records: List[dict[str, Any]],
    required: int,
    pair_target: int,
    rng: random.Random,
) -> List[dict[str, Any]]:
    buckets = group_by_num_objects(records)
    cap_value: int | None = None
    if pair_target > 0:
        capped_capacity = sum(
            min(len(group), pair_target) for group in buckets.values()
        )
        if capped_capacity >= required:
            cap_value = pair_target
        else:
            print(
                f"Warning: sub_category '{sub_category}' requires {required} samples but only "
                f"{capped_capacity} fit the per-pair cap of {pair_target}; relaxing the cap."
            )
    allocations = allocate_evenly(buckets, required, rng, per_key_cap=cap_value)

    sampled: List[dict[str, Any]] = []
    for num_objects, count in allocations.items():
        if count <= 0:
            continue
        sampled.extend(rng.sample(buckets[num_objects], count))

    rng.shuffle(sampled)
    return sampled


def stratified_sample(
    questions: List[dict[str, Any]],
    total: int,
    pair_target: int,
    rng: random.Random,
) -> tuple[List[dict[str, Any]], Dict[str, Counter]]:
    for record in questions:
        resolve_num_objects(record)

    grouped = group_by_sub_category(questions)
    if total > len(questions):
        raise SystemExit(
            f"Requested {total} samples but dataset only contains {len(questions)} records "
            "after filtering."
        )

    try:
        sub_allocations = allocate_evenly(grouped, total, rng, per_key_cap=None)
    except AllocationError as exc:
        raise SystemExit(
            f"Unable to allocate {total} questions evenly across {len(grouped)} sub_categories: {exc}"
        ) from exc

    sampled: List[dict[str, Any]] = []
    summary: Dict[str, Counter] = {}

    for sub_category, records in grouped.items():
        required = sub_allocations[sub_category]
        if required <= 0:
            continue
        try:
            chosen = sample_within_sub_category(
                sub_category, records, required, pair_target, rng
            )
        except AllocationError as exc:
            raise SystemExit(
                f"Unable to cover all object-count buckets for sub_category '{sub_category}': {exc}"
            ) from exc

        counters = summary.setdefault(sub_category, Counter())
        for record in chosen:
            counters[resolve_num_objects(record)] += 1
        sampled.extend(chosen)

    rng.shuffle(sampled)
    return sampled, summary


def print_summary(summary: Dict[str, Counter]) -> None:
    total_records = 0
    aggregate_objects: Counter[int] = Counter()

    print("Distribution by sub_category and num_objects:")
    for sub_category in sorted(summary.keys()):
        counters = summary[sub_category]
        sub_total = sum(counters.values())
        total_records += sub_total
        print(f"  {sub_category}: {sub_total}")
        for num_objects in sorted(counters.keys()):
            count = counters[num_objects]
            aggregate_objects[num_objects] += count
            print(f"    num_objects={num_objects}: {count}")

    print("Totals per num_objects across all sub_categories:")
    for num_objects in sorted(aggregate_objects.keys()):
        print(f"  num_objects={num_objects}: {aggregate_objects[num_objects]}")
    print(f"Total sampled questions: {total_records}")


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise SystemExit("Requested sample size must be positive.")

    questions = load_questions(args.input)
    if args.mode is not None:
        questions = [record for record in questions if record.get("mode") == args.mode]

    if not questions:
        raise SystemExit("No records left after applying the requested filters.")

    rng = random.Random(args.seed)
    sampled, summary = stratified_sample(
        questions=questions,
        total=args.count,
        pair_target=args.pair_target,
        rng=rng,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(sampled, handle, indent=4)

    print_summary(summary)


if __name__ == "__main__":
    main()
