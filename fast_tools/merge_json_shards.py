#!/usr/bin/env python3
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

SHARD_RE = re.compile(r"^(?P<prefix>.+)-(?P<part>\d+)(?P<suffix>.*)\.json$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge sharded JSON files in a directory (e.g. file-1.json, file-2.json) "
            "into file-all.json and report idx statistics."
        )
    )
    parser.add_argument(
        "directory",
        help="Directory containing shard JSON files.",
    )
    parser.add_argument(
        "--pattern",
        default="*.json",
        help="Glob pattern used inside directory (default: *.json).",
    )
    parser.add_argument(
        "--group",
        default=None,
        help=(
            "Optional prefix filter before the shard number. "
            "Example: test_run_28_general for test_run_28_general-1_karo_10K.json"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output -all files if they already exist.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Write merged output with indentation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be merged without writing files.",
    )
    return parser.parse_args()


def collect_shard_groups(
    directory: Path, pattern: str, group_filter: str | None
) -> Dict[Tuple[str, str], List[Tuple[int, Path]]]:
    groups: Dict[Tuple[str, str], List[Tuple[int, Path]]] = defaultdict(list)

    for path in sorted(directory.glob(pattern)):
        if not path.is_file():
            continue

        match = SHARD_RE.match(path.name)
        if not match:
            continue

        prefix = match.group("prefix")
        part = int(match.group("part"))
        suffix = match.group("suffix")

        if group_filter is not None and prefix != group_filter:
            continue

        output_name = f"{prefix}-all{suffix}.json"
        if path.name == output_name:
            continue

        groups[(prefix, suffix)].append((part, path))

    return {
        key: sorted(entries, key=lambda x: x[0])
        for key, entries in groups.items()
        if len(entries) >= 2
    }


def extract_items(data, source_path: Path):
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ("questions", "data", "items"):
            value = data.get(key)
            if isinstance(value, list):
                return value

    raise ValueError(
        f"{source_path} must be a JSON list or contain list key in "
        "'questions', 'data', or 'items'."
    )


def merge_group(
    directory: Path,
    prefix: str,
    suffix: str,
    shards: List[Tuple[int, Path]],
    overwrite: bool,
    pretty: bool,
    dry_run: bool,
) -> tuple[Path, int, int, int, int, int]:
    output_path = directory / f"{prefix}-all{suffix}.json"

    if output_path.exists() and not overwrite and not dry_run:
        raise FileExistsError(
            f"{output_path} already exists. Use --overwrite to replace it."
        )

    merged_items = []
    for _, shard_path in shards:
        with shard_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        merged_items.extend(extract_items(data, shard_path))

    idx_values = [
        item["idx"]
        for item in merged_items
        if isinstance(item, dict) and "idx" in item
    ]
    total_rows = len(merged_items)
    total_idx_entries = len(idx_values)
    distinct_idx = len(set(idx_values))
    duplicate_idx_values = total_idx_entries - distinct_idx

    if not dry_run:
        with output_path.open("w", encoding="utf-8") as f:
            if pretty:
                json.dump(merged_items, f, indent=2, ensure_ascii=False)
                f.write("\n")
            else:
                json.dump(merged_items, f, ensure_ascii=False)

    missing_idx = total_rows - total_idx_entries
    return (
        output_path,
        total_rows,
        total_idx_entries,
        distinct_idx,
        duplicate_idx_values,
        missing_idx,
    )


def main() -> None:
    args = parse_args()
    directory = Path(args.directory).expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"{directory} is not a directory.")

    groups = collect_shard_groups(directory, args.pattern, args.group)
    if not groups:
        print("No shard groups found to merge.")
        return

    merged_group_count = 0
    for (prefix, suffix), shards in sorted(groups.items()):
        shard_names = ", ".join(path.name for _, path in shards)
        print(f"\nGroup: {prefix}-[{len(shards)} shards]{suffix}.json")
        print(f"Shards: {shard_names}")

        output_path, total_rows, idx_entries, distinct_idx, duplicate_idx, missing_idx = merge_group(
            directory=directory,
            prefix=prefix,
            suffix=suffix,
            shards=shards,
            overwrite=args.overwrite,
            pretty=args.pretty,
            dry_run=args.dry_run,
        )

        if args.dry_run:
            print(f"Would write: {output_path}")
        else:
            print(f"Wrote: {output_path}")

        print(f"Rows merged: {total_rows}")
        print(f"idx entries: {idx_entries}")
        print(f"Distinct idx: {distinct_idx}")
        print(f"Duplicate idx values: {duplicate_idx}")
        print(f"Rows missing idx: {missing_idx}")
        merged_group_count += 1

    print(f"\nMerged groups: {merged_group_count}")


if __name__ == "__main__":
    main()
