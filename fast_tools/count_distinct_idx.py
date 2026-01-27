#!/usr/bin/env python3
import argparse
import json
from collections import deque

def _collect_from_candidates(candidates):
    idx_values = []
    for item in candidates:
        if isinstance(item, dict) and "idx" in item:
            idx_values.append(item["idx"])
    return idx_values

def _fallback_recursive_collect(data):
    idx_values = []
    queue = deque([data])
    while queue:
        current = queue.popleft()
        if isinstance(current, dict):
            if "idx" in current:
                idx_values.append(current["idx"])
            for value in current.values():
                queue.append(value)
        elif isinstance(current, list):
            queue.extend(current)
    return idx_values

def count_distinct_idx(data):
    candidates = None
    if isinstance(data, list):
        candidates = data
    elif isinstance(data, dict):
        for key in ("questions", "data", "items"):
            if isinstance(data.get(key), list):
                candidates = data[key]
                break
        if candidates is None and data and all(isinstance(v, dict) for v in data.values()):
            candidates = data.values()

    idx_values = []
    if candidates is not None:
        idx_values = _collect_from_candidates(candidates)

    if not idx_values:
        idx_values = _fallback_recursive_collect(data)

    return len(set(idx_values))


def main():
    parser = argparse.ArgumentParser(description="Count distinct 'idx' values in a JSON file.")
    parser.add_argument(
        "path",
        nargs="?",
        default="/data0/sebastian.cavada/compositional-physics/tiny_vqa_creation/output/run_24_general/test_run_24_general.json",
        help="Path to JSON file (defaults to run_24_general test file).",
    )
    args = parser.parse_args()

    with open(args.path, "r", encoding="utf-8") as f:
        data = json.load(f)

    distinct_count = count_distinct_idx(data)
    print(distinct_count)


if __name__ == "__main__":
    main()
