#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from typing import Iterable


def iter_records(obj: object) -> Iterable[dict]:
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                yield item
    elif isinstance(obj, dict):
        # If the JSON is a dict of records, try values.
        for item in obj.values():
            if isinstance(item, dict):
                yield item


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check for duplicate keys (default: idx) in a JSON file."
    )
    parser.add_argument("path", help="Path to JSON file.")
    parser.add_argument(
        "--key",
        default="idx",
        help="Record key to check for duplicates (default: idx).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Max number of duplicate keys to print (default: 50).",
    )
    args = parser.parse_args()

    with open(args.path, "r", encoding="utf-8") as f:
        data = json.load(f)

    counts = Counter()
    missing = 0
    total = 0
    for rec in iter_records(data):
        total += 1
        if args.key not in rec:
            missing += 1
            continue
        counts[str(rec[args.key])] += 1

    dupes = [(k, c) for k, c in counts.items() if c > 1]
    dupes.sort(key=lambda x: (-x[1], x[0]))

    print(f"records_with_key: {total - missing}")
    if missing:
        print(f"records_missing_key: {missing}")
    print(f"unique_{args.key}: {len(counts)}")
    print(f"duplicate_{args.key}: {len(dupes)}")

    if dupes:
        print(f"\nTop duplicate {args.key} values:")
        for k, c in dupes[: args.limit]:
            print(f"{k}\t{c}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
