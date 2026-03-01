#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two JSON files containing lists of records. "
            "Comparison is order-independent and keyed by idx."
        )
    )
    parser.add_argument("file_a", help="First JSON file path.")
    parser.add_argument("file_b", help="Second JSON file path.")
    parser.add_argument(
        "--idx-key",
        default="idx",
        help="Record key used as unique identifier (default: idx).",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=10,
        help="Maximum number of sample mismatches to print (default: 10).",
    )
    parser.add_argument(
        "--strict-order",
        action="store_true",
        help="Also require list order to match exactly.",
    )
    return parser.parse_args()


def load_list(path: Path) -> List[Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list.")
    return data


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def index_by_key(
    rows: List[Any], idx_key: str
) -> Tuple[Dict[Any, List[dict]], List[int], List[int]]:
    idx_map: Dict[Any, List[dict]] = defaultdict(list)
    missing_key_rows: List[int] = []
    non_dict_rows: List[int] = []

    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            non_dict_rows.append(i)
            continue
        if idx_key not in row:
            missing_key_rows.append(i)
            continue
        idx_map[row[idx_key]].append(row)

    return idx_map, missing_key_rows, non_dict_rows


def sample(values: List[Any], limit: int) -> str:
    if not values:
        return "-"
    shown = values[:limit]
    suffix = "" if len(values) <= limit else f" ... (+{len(values) - limit} more)"
    return ", ".join(str(v) for v in shown) + suffix


def diff_keys_for_dicts(a: dict, b: dict) -> List[str]:
    keys = set(a.keys()) | set(b.keys())
    return sorted(k for k in keys if a.get(k) != b.get(k))


def first_order_mismatch(rows_a: List[Any], rows_b: List[Any]) -> str:
    limit = min(len(rows_a), len(rows_b))
    for i in range(limit):
        if rows_a[i] != rows_b[i]:
            a = rows_a[i]
            b = rows_b[i]
            if isinstance(a, dict) and isinstance(b, dict):
                changed = diff_keys_for_dicts(a, b)
                return f"index={i}, changed_keys={changed[:15]}"
            return f"index={i}, row types: A={type(a).__name__}, B={type(b).__name__}"
    if len(rows_a) != len(rows_b):
        return f"length differs: A={len(rows_a)}, B={len(rows_b)}"
    return "no mismatch found"


def main() -> None:
    args = parse_args()
    file_a = Path(args.file_a).expanduser().resolve()
    file_b = Path(args.file_b).expanduser().resolve()

    rows_a = load_list(file_a)
    rows_b = load_list(file_b)

    idx_map_a, missing_a, non_dict_a = index_by_key(rows_a, args.idx_key)
    idx_map_b, missing_b, non_dict_b = index_by_key(rows_b, args.idx_key)

    keys_a = set(idx_map_a.keys())
    keys_b = set(idx_map_b.keys())

    only_a = sorted(keys_a - keys_b, key=str)
    only_b = sorted(keys_b - keys_a, key=str)
    common = sorted(keys_a & keys_b, key=str)

    duplicate_idxs_a = sorted([k for k, v in idx_map_a.items() if len(v) > 1], key=str)
    duplicate_idxs_b = sorted([k for k, v in idx_map_b.items() if len(v) > 1], key=str)

    content_mismatch_idxs: List[Any] = []
    question_mismatch_count = 0
    detailed_samples: List[str] = []

    for idx in common:
        rows_for_idx_a = idx_map_a[idx]
        rows_for_idx_b = idx_map_b[idx]

        if len(rows_for_idx_a) == 1 and len(rows_for_idx_b) == 1:
            row_a = rows_for_idx_a[0]
            row_b = rows_for_idx_b[0]
            if row_a != row_b:
                content_mismatch_idxs.append(idx)
                if row_a.get("question") != row_b.get("question"):
                    question_mismatch_count += 1
                if len(detailed_samples) < args.show:
                    keys_changed = diff_keys_for_dicts(row_a, row_b)
                    detailed_samples.append(
                        f"idx={idx} changed_keys={keys_changed[:15]}"
                    )
            continue

        counter_a = Counter(canonical_json(x) for x in rows_for_idx_a)
        counter_b = Counter(canonical_json(x) for x in rows_for_idx_b)
        if counter_a != counter_b:
            content_mismatch_idxs.append(idx)
            if len(detailed_samples) < args.show:
                detailed_samples.append(
                    f"idx={idx} duplicate-count/content differs "
                    f"(A has {len(rows_for_idx_a)} rows, B has {len(rows_for_idx_b)} rows)"
                )

    total_rows_a = len(rows_a)
    total_rows_b = len(rows_b)
    idx_entries_a = sum(len(v) for v in idx_map_a.values())
    idx_entries_b = sum(len(v) for v in idx_map_b.values())
    distinct_idx_a = len(idx_map_a)
    distinct_idx_b = len(idx_map_b)

    print(f"A: {file_a}")
    print(f"B: {file_b}")
    print()
    print(f"Rows: A={total_rows_a}, B={total_rows_b}")
    print(f"idx entries: A={idx_entries_a}, B={idx_entries_b}")
    print(f"Distinct idx: A={distinct_idx_a}, B={distinct_idx_b}")
    print(
        f"Rows missing '{args.idx_key}': A={len(missing_a)}, B={len(missing_b)}"
    )
    print(f"Non-dict rows: A={len(non_dict_a)}, B={len(non_dict_b)}")
    print(f"Duplicate idx values: A={len(duplicate_idxs_a)}, B={len(duplicate_idxs_b)}")
    print(f"idx only in A: {len(only_a)}")
    print(f"idx only in B: {len(only_b)}")
    print(f"Common idx with different content: {len(content_mismatch_idxs)}")
    print(f"Question mismatches within differing idx: {question_mismatch_count}")
    order_equal = rows_a == rows_b
    print(f"Exact row order equal: {order_equal}")
    print()

    is_match = (
        not missing_a
        and not missing_b
        and not non_dict_a
        and not non_dict_b
        and not only_a
        and not only_b
        and not content_mismatch_idxs
        and (order_equal if args.strict_order else True)
    )

    if is_match:
        if args.strict_order:
            print("MATCH: files are exactly the same with strict row order.")
        else:
            print("MATCH: files are exactly the same by idx (ignoring row order).")
        return

    print("MISMATCH: files differ.")
    print(f"Sample idx only in A: {sample(only_a, args.show)}")
    print(f"Sample idx only in B: {sample(only_b, args.show)}")
    print(f"Sample duplicate idx in A: {sample(duplicate_idxs_a, args.show)}")
    print(f"Sample duplicate idx in B: {sample(duplicate_idxs_b, args.show)}")
    print(f"Sample differing idx: {sample(content_mismatch_idxs, args.show)}")
    if args.strict_order and not order_equal:
        print(f"First strict-order mismatch: {first_order_mismatch(rows_a, rows_b)}")
    if detailed_samples:
        print("Detailed examples:")
        for line in detailed_samples:
            print(f"  - {line}")


if __name__ == "__main__":
    main()
