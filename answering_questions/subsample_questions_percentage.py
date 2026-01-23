#!/usr/bin/env python3
"""Subsample questions by applying per-balance-key percentage targets."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Mapping, Sequence, Tuple

from subsample_questions_balanced import (
    MISSING_TOKEN,
    load_questions,
    make_balance_groups,
)

NUM_OBJECTS_PATTERN = re.compile(r"_no-(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Subsample a JSON question set by keeping a percentage of each balance group. "
            "Percentages can be supplied inline as JSON or via a JSON file."
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
        "--percentage-map",
        type=str,
        required=True,
        metavar="JSON",
        help=(
            "Mapping from balance keys to percentages. Provide either an inline JSON string "
            '(e.g., \'{"distance": 0.4, "occlusion": 0.6}\') or a path to a JSON file.'
        ),
    )
    parser.add_argument(
        "--default-percentage",
        type=float,
        default=None,
        help=(
            "Fallback percentage (0-1 or 0-100) applied to balance groups missing from the map. "
            "If omitted, encountering an unmapped group raises an error."
        ),
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help=(
            "Optional cap on the total number of sampled questions. When provided, the per-group "
            "percentages are treated as proportions that will be scaled to fit within this limit."
        ),
    )
    parser.add_argument(
        "--balance-on",
        nargs="+",
        default=["sub_category"],
        metavar="FIELD",
        help=(
            "Fields used to define balance groups (default: sub_category). Use '-' to skip grouping "
            "and apply a single percentage to the entire dataset. When 'num_objects' is listed here "
            "it will be inferred from the simulation_id path."
        ),
    )
    parser.add_argument(
        "--objects-per-count",
        type=int,
        default=None,
        metavar="COUNT",
        help=(
            "If provided, enforce that exactly COUNT questions are kept for each num_objects value. "
            "Requires enough questions per object count and may fail if idx grouping prevents an exact fit."
        ),
    )
    parser.add_argument(
        "--soft-objects-per-count",
        action="store_true",
        help=(
            "When enforcing --objects-per-count, keep all available questions for a num_objects "
            "value and emit a warning instead of exiting if the exact quota cannot be met."
        ),
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--group-by-idx",
        dest="group_by_idx",
        action="store_true",
        help=(
            "Ensure questions sharing the same base idx (e.g., '12_g'/'12_i') are kept or discarded "
            "together."
        ),
    )
    group.add_argument(
        "--no-group-by-idx",
        dest="group_by_idx",
        action="store_false",
        help="Disable idx grouping so each question is treated independently.",
    )
    parser.set_defaults(group_by_idx=True)
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        help="Filter questions by this mode before sampling (e.g., 'general').",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed so percentage rounding is reproducible.",
    )
    return parser.parse_args()


def load_percentage_map(raw: str) -> Dict[str, float]:
    path = Path(raw)
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Failed to parse JSON from '{path}': {exc}") from exc
    else:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SystemExit(
                "--percentage-map must be valid JSON (either inline or path)."
            ) from exc

    if not isinstance(payload, Mapping):
        raise SystemExit(
            f"Percentage map must be a JSON object, received {type(payload).__name__}."
        )

    parsed: Dict[str, float] = {}
    for key, value in payload.items():
        if not isinstance(value, (int, float)):
            raise SystemExit(
                f"Percentage for '{key}' must be numeric, received {type(value).__name__}."
            )
        parsed[str(key)] = normalise_percentage(
            float(value), context=f"map entry '{key}'"
        )

    if not parsed:
        raise SystemExit("Percentage map is empty; nothing to subsample toward.")
    return parsed


def normalise_percentage(value: float, context: str = "value") -> float:
    if value < 0:
        raise SystemExit(f"Percentage for {context} must be non-negative.")
    if value <= 1:
        return value
    if value <= 100:
        return value / 100.0
    raise SystemExit(f"Percentage for {context} exceeds 100%: {value}.")


def format_group_key(key: Tuple[Any, ...]) -> str:
    if not key:
        return "<all>"
    if len(key) == 1:
        return str(key[0])
    return "|".join(str(part) for part in key)


def extract_balance_key(
    record: dict[str, Any], fields: Sequence[str]
) -> Tuple[Any, ...]:
    if not fields or fields == ["-"]:
        return ()
    components: List[Any] = []
    for field in fields:
        if field not in record:
            raise SystemExit(
                f"Cannot balance on '{field}' because it is missing from a record while grouping by idx."
            )
        value = record[field]
        components.append(MISSING_TOKEN if value in {None, ""} else value)
    return tuple(components)


def normalise_index(value: Any) -> str:
    if isinstance(value, str):
        base, _, suffix = value.rpartition("_")
        if base and suffix in {"g", "i"}:
            return base
        return value
    return str(value)


def group_by_index_suffix(
    questions: Sequence[dict[str, Any]],
) -> Dict[str, List[dict[str, Any]]]:
    grouped: Dict[str, List[dict[str, Any]]] = defaultdict(list)
    for record in questions:
        raw_idx = record.get("idx")
        if raw_idx is None:
            raise SystemExit(
                "Found a question without an 'idx' field; cannot enforce idx grouping."
            )
        key = normalise_index(raw_idx)
        grouped[key].append(record)
    return grouped


def allocate_from_percentages(
    grouped: Dict[Tuple[Any, ...], List[dict[str, Any]]],
    percentages: Dict[str, float],
    default_percentage: float | None,
    rng: random.Random,
    max_total: int | None,
) -> Tuple[List[dict[str, Any]], Dict[Tuple[Any, ...], float]]:
    keys = list(grouped.keys())
    raw_targets: Dict[Tuple[Any, ...], float] = {key: 0.0 for key in keys}

    if max_total is not None and max_total <= 0:
        return [], raw_targets

    fractions: Dict[Tuple[Any, ...], float] = {}
    takes: Dict[Tuple[Any, ...], int] = {}

    def resolve_percentage(label: str) -> float:
        pct = percentages.get(label)
        if pct is None:
            if default_percentage is None:
                raise SystemExit(
                    f"No percentage specified for balance group '{label}' and no default provided."
                )
            pct = default_percentage
        return normalise_percentage(pct, context=f"group '{label}'")

    total_capacity = sum(len(grouped[key]) for key in keys)
    if max_total is None:
        for key in keys:
            label = format_group_key(key)
            pct = resolve_percentage(label)
            raw = pct * len(grouped[key])
            raw_targets[key] = min(float(len(grouped[key])), raw)
        desired_total = sum(raw_targets.values())
        if desired_total == 0:
            return [], raw_targets
    else:
        target_total = min(max_total, total_capacity)
        if target_total == 0:
            return [], raw_targets

        weights: Dict[Tuple[Any, ...], float] = {}
        for key in keys:
            label = format_group_key(key)
            weights[key] = max(0.0, resolve_percentage(label))

        active = [key for key in keys if weights[key] > 0 and len(grouped[key]) > 0]
        if not active:
            return [], raw_targets

        remaining_total = float(target_total)
        capacities = {key: float(len(grouped[key])) for key in keys}
        while remaining_total > 1e-9 and active:
            weight_sum = sum(weights[key] for key in active)
            if weight_sum <= 0:
                break
            progress = 0.0
            shares: Dict[Tuple[Any, ...], float] = {}
            for key in active:
                shares[key] = remaining_total * (weights[key] / weight_sum)
            for key in list(active):
                space = capacities[key] - raw_targets[key]
                if space <= 1e-9:
                    active.remove(key)
                    continue
                take = min(shares[key], space)
                raw_targets[key] += take
                progress += take
                if space - take <= 1e-9:
                    active.remove(key)
            if progress == 0.0:
                break
            remaining_total -= progress

        desired_total = sum(raw_targets.values())
        if desired_total == 0:
            return [], raw_targets

    sampled: List[dict[str, Any]] = []
    for key in keys:
        raw_value = raw_targets[key]
        base = int(math.floor(raw_value))
        if base > len(grouped[key]):
            base = len(grouped[key])
        takes[key] = base
        fractions[key] = max(0.0, raw_value - base)

    total_taken = sum(takes.values())
    if max_total is None:
        for key in keys:
            if takes[key] >= len(grouped[key]):
                continue
            if rng.random() < fractions[key]:
                takes[key] += 1
    else:
        target_total = int(round(desired_total))
        target_total = min(target_total, max_total)
        remaining = target_total - total_taken
        if remaining < 0:
            remaining = 0
        if remaining > 0:
            order = keys[:]
            rng.shuffle(order)
            order.sort(key=lambda key: fractions[key], reverse=True)
            for key in order:
                while (
                    remaining and takes[key] < len(grouped[key]) and fractions[key] > 0
                ):
                    takes[key] += 1
                    remaining -= 1
            if remaining:
                order = keys[:]
                rng.shuffle(order)
                for key in order:
                    while remaining and takes[key] < len(grouped[key]):
                        takes[key] += 1
                        remaining -= 1

    for key in keys:
        take = takes[key]
        if take <= 0:
            continue
        sampled.extend(rng.sample(grouped[key], take))

    rng.shuffle(sampled)
    return sampled, raw_targets


def _derive_num_objects(simulation_id: Any, idx: Any) -> int:
    if not isinstance(simulation_id, str) or not simulation_id:
        raise SystemExit(
            f"Cannot derive 'num_objects' for record idx '{idx}': missing simulation_id field."
        )
    match = NUM_OBJECTS_PATTERN.search(simulation_id)
    if not match:
        raise SystemExit(
            f"Cannot derive 'num_objects' for record idx '{idx}': simulation_id does not contain '_no-<count>'."
        )
    value = int(match.group(1))
    if value <= 0:
        raise SystemExit(
            f"Derived 'num_objects' for record idx '{idx}' is invalid (value: {value})."
        )
    return value


def _resolve_num_objects(record: dict[str, Any]) -> int:
    num_objects = record.get("num_objects")
    if num_objects in {None, ""}:
        num_objects = _derive_num_objects(
            record.get("simulation_id"), record.get("idx")
        )
    try:
        return int(num_objects)
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            f"Unable to determine 'num_objects' for record idx '{record.get('idx')}': {exc}"
        ) from exc


def ensure_derived_balance_fields(
    questions: List[dict[str, Any]], balance_fields: Sequence[str]
) -> None:
    if not balance_fields or balance_fields == ["-"]:
        return
    needs_num_objects = "num_objects" in balance_fields
    if not needs_num_objects:
        return
    for record in questions:
        if record.get("num_objects") not in {None, ""}:
            continue
        record["num_objects"] = _derive_num_objects(
            record.get("simulation_id"), record.get("idx")
        )


def enforce_idx_grouping(
    sampled: List[dict[str, Any]],
    idx_lookup: Dict[str, List[dict[str, Any]]],
    max_total: int | None,
    rng: random.Random,
    balance_fields: Sequence[str],
    balance_targets: Dict[Tuple[Any, ...], float],
) -> List[dict[str, Any]]:
    if not sampled:
        return []

    ordered_keys: List[str] = []
    seen: set[str] = set()
    groups_meta: List[Tuple[str, List[dict[str, Any]], Tuple[Any, ...]]] = []
    for record in sampled:
        raw_idx = record.get("idx")
        if raw_idx is None:
            raise SystemExit(
                "Cannot group by idx because a sampled record is missing 'idx'."
            )
        key = normalise_index(raw_idx)
        if key not in idx_lookup:
            raise SystemExit(
                f"Idx '{key}' present in sampled data but missing from the lookup table."
            )
        if key not in seen:
            ordered_keys.append(key)
            seen.add(key)
            balance_key = extract_balance_key(idx_lookup[key][0], balance_fields)
            groups_meta.append((key, idx_lookup[key], balance_key))

    expanded: List[dict[str, Any]] = []
    for _, group, _ in groups_meta:
        expanded.extend(group)

    if max_total is None or len(expanded) <= max_total:
        return expanded

    actual_counts: DefaultDict[Tuple[Any, ...], int] = defaultdict(int)
    for _, group, balance_key in groups_meta:
        actual_counts[balance_key] += len(group)

    kept = groups_meta[:]
    excess = len(expanded) - max_total
    while excess > 0 and kept:
        candidates = []
        for idx_key, group, balance_key in kept:
            over = max(
                actual_counts[balance_key] - balance_targets.get(balance_key, 0.0), 0.0
            )
            candidates.append((over, len(group), rng.random(), idx_key))

        candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        selected_idx = candidates[0][3]

        for i, (idx_key, group, balance_key) in enumerate(kept):
            if idx_key == selected_idx:
                actual_counts[balance_key] -= len(group)
                excess -= len(group)
                kept.pop(i)
                break

    flattened: List[dict[str, Any]] = []
    for _, group, _ in kept:
        flattened.extend(group)
    return flattened


def _select_idx_groups_for_quota(
    groups: Sequence[Tuple[str, List[dict[str, Any]]]],
    target: int,
    rng: random.Random,
) -> List[Tuple[str, List[dict[str, Any]]]]:
    if target <= 0:
        return []
    ordered = list(groups)
    rng.shuffle(ordered)
    prev_sum = [-1] * (target + 1)
    choice = [-1] * (target + 1)
    prev_sum[0] = 0

    for idx, (_, records) in enumerate(ordered):
        size = len(records)
        if size > target:
            continue
        for total in range(target, size - 1, -1):
            if prev_sum[total] != -1 or prev_sum[total - size] == -1:
                continue
            prev_sum[total] = total - size
            choice[total] = idx

    if prev_sum[target] == -1:
        return []

    selected: List[Tuple[str, List[dict[str, Any]]]] = []
    remaining = target
    used: set[int] = set()
    while remaining > 0:
        idx = choice[remaining]
        if idx == -1 or idx in used:
            return []
        used.add(idx)
        selected.append(ordered[idx])
        remaining = prev_sum[remaining]
        if remaining == -1:
            return []
    return selected


def enforce_object_quota(
    sampled: List[dict[str, Any]],
    target_per_count: int,
    rng: random.Random,
    respect_idx_groups: bool,
    soft: bool,
) -> List[dict[str, Any]]:
    if target_per_count <= 0:
        raise SystemExit(
            "--objects-per-count must be a positive integer when provided."
        )
    if not sampled:
        return []

    if respect_idx_groups:
        idx_groups = group_by_index_suffix(sampled)
        grouped_by_objects: DefaultDict[int, List[Tuple[str, List[dict[str, Any]]]]] = (
            defaultdict(list)
        )
        for idx_key, group in idx_groups.items():
            grouped_by_objects[_resolve_num_objects(group[0])].append((idx_key, group))

        trimmed: List[dict[str, Any]] = []
        for num_objects, groups in grouped_by_objects.items():
            available = sum(len(records) for _, records in groups)
            if available < target_per_count:
                if soft:
                    print(
                        f"Warning: cannot enforce {target_per_count} questions for num_objects={num_objects}: "
                        f"only {available} available while preserving idx grouping. "
                        "Keeping all available for this num_objects value.",
                        file=sys.stderr,
                    )
                    for _, records in groups:
                        trimmed.extend(records)
                    continue
                raise SystemExit(
                    f"Cannot enforce {target_per_count} questions for num_objects={num_objects}: "
                    f"only {available} available while preserving idx grouping."
                )
            selected = _select_idx_groups_for_quota(groups, target_per_count, rng)
            if not selected:
                if soft:
                    print(
                        f"Warning: unable to allocate exactly {target_per_count} questions for num_objects={num_objects} "
                        "while preserving idx grouping. Keeping all available for this num_objects value.",
                        file=sys.stderr,
                    )
                    for _, records in groups:
                        trimmed.extend(records)
                    continue
                raise SystemExit(
                    f"Unable to allocate exactly {target_per_count} questions for num_objects={num_objects} "
                    "while preserving idx grouping. Consider --no-group-by-idx or a different target."
                )
            for _, records in selected:
                trimmed.extend(records)
        rng.shuffle(trimmed)
        return trimmed

    buckets: DefaultDict[int, List[dict[str, Any]]] = defaultdict(list)
    for record in sampled:
        buckets[_resolve_num_objects(record)].append(record)

    trimmed: List[dict[str, Any]] = []
    for num_objects, records in buckets.items():
        if len(records) < target_per_count:
            if soft:
                print(
                    f"Warning: cannot enforce {target_per_count} questions for num_objects={num_objects}: "
                    f"only {len(records)} available. Keeping all available for this num_objects value.",
                    file=sys.stderr,
                )
                trimmed.extend(records)
                continue
            raise SystemExit(
                f"Cannot enforce {target_per_count} questions for num_objects={num_objects}: "
                f"only {len(records)} available."
            )
        trimmed.extend(rng.sample(records, target_per_count))
    rng.shuffle(trimmed)
    return trimmed


def main() -> None:
    args = parse_args()
    questions = load_questions(args.input)
    if args.mode is not None:
        questions = [record for record in questions if record.get("mode") == args.mode]
    ensure_derived_balance_fields(questions, args.balance_on)

    percentages = load_percentage_map(args.percentage_map)
    default_percentage = (
        None
        if args.default_percentage is None
        else normalise_percentage(
            args.default_percentage, context="--default-percentage"
        )
    )

    rng = random.Random(args.seed)
    balance_fields: Sequence[str] = args.balance_on
    grouped = make_balance_groups(questions, balance_fields)
    sampled, targets = allocate_from_percentages(
        grouped,
        percentages,
        default_percentage,
        rng,
        args.count,
    )

    if args.group_by_idx:
        idx_lookup = group_by_index_suffix(questions)
        sampled = enforce_idx_grouping(
            sampled,
            idx_lookup,
            args.count,
            rng,
            balance_fields,
            targets,
        )

    if args.objects_per_count is not None:
        sampled = enforce_object_quota(
            sampled,
            args.objects_per_count,
            rng,
            args.group_by_idx,
            args.soft_objects_per_count,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(sampled, handle, indent=4)

    grouped_sample = make_balance_groups(sampled, balance_fields)
    print(
        f"Sampled {len(sampled)} records across {len(grouped_sample)} balance groups "
        f"using fields {', '.join(balance_fields)}."
    )
    for key, group in grouped_sample.items():
        print(f"  {format_group_key(key)}: {len(group)}")

    object_counts: DefaultDict[int, int] = defaultdict(int)
    for record in sampled:
        object_counts[_resolve_num_objects(record)] += 1

    if object_counts:
        print("Counts per object total:")
        for count in sorted(object_counts):
            print(f"  {count}: {object_counts[count]}")


if __name__ == "__main__":
    main()


# usage
# python answering_questions/subsample_questions_percentage.py \
#     --input answering_questions/data/tiny_vqa_all_questions.json \
#     --output answering_questions/data/tiny_vqa_subsampled_50pct_distance_occlusion.json \
#     --percentage-map answering_questions/balancing_sub_categories.json \
#     --seed 42
