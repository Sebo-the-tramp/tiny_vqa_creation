#!/usr/bin/env python3
"""Count questions by idx suffix.

Reads a JSON file containing a list of question objects (or a dict wrapping one).
Prints total count, count of idx ending with _g (multi-image), and _i (single-image).
"""

import argparse
import json
import sys
from typing import Any, List


def _extract_list(data: Any) -> List[dict]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # Common container keys
        for key in ("questions", "data", "items", "records"):
            value = data.get(key)
            if isinstance(value, list):
                return value
        # Fallback: first list value
        for value in data.values():
            if isinstance(value, list):
                return value
    raise TypeError("JSON must be a list of objects or a dict containing a list")


def main() -> int:
    parser = argparse.ArgumentParser(description="Count questions by idx suffix")
    parser.add_argument("path", help="Path to JSON file")
    args = parser.parse_args()

    try:
        with open(args.path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {args.path}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"Invalid JSON: {exc}", file=sys.stderr)
        return 2

    questions = _extract_list(data)

    total = len(questions)
    multi_g = 0
    single_i = 0
    missing_idx = 0

    for item in questions:
        if not isinstance(item, dict):
            continue
        idx = item.get("idx")
        if not isinstance(idx, str):
            missing_idx += 1
            continue
        if idx.endswith("_g"):
            multi_g += 1
        if idx.endswith("_i"):
            single_i += 1

    print(f"total_questions: {total}")
    print(f"idx_endswith__g: {multi_g}")
    print(f"idx_endswith__i: {single_i}")
    if missing_idx:
        print(f"missing_or_nonstring_idx: {missing_idx}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
