#!/usr/bin/env python3
"""
Merge model result JSON files across results_run_28_general-<number> folders.

By default it scans:
  /Users/sebastiancavada/Desktop/tmp_paris/tiny_vqa_creation/output/run_28_general
and writes merged files to:
  .../results_run_28_general-all
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


RESULTS_PREFIX = "results_run_28_general-"
OUTPUT_DIRNAME = "results_run_28_general-all"


def find_result_dirs(base: Path) -> List[Tuple[int, Path]]:
    dirs: List[Tuple[int, Path]] = []
    for p in base.iterdir():
        if not p.is_dir():
            continue
        if not p.name.startswith(RESULTS_PREFIX):
            continue
        suffix = p.name[len(RESULTS_PREFIX) :]
        if suffix.isdigit():
            dirs.append((int(suffix), p))
    dirs.sort(key=lambda x: x[0])
    return dirs


def list_model_files(dirs: List[Path]) -> List[str]:
    filenames = set()
    for d in dirs:
        for f in d.iterdir():
            if f.is_file() and f.suffix == ".json":
                filenames.add(f.name)
    return sorted(filenames)


def merge_lists(items: List[dict]) -> Tuple[List[dict], int, int]:
    """Merge a list of dicts by idx when present.

    Returns (merged, deduped_count, missing_idx_count)
    """
    seen = set()
    merged = []
    deduped = 0
    missing_idx = 0
    for item in items:
        if not isinstance(item, dict):
            merged.append(item)
            continue
        idx = item.get("idx")
        if idx is None:
            missing_idx += 1
            merged.append(item)
            continue
        if idx in seen:
            deduped += 1
            continue
        seen.add(idx)
        merged.append(item)
    return merged, deduped, missing_idx


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge run_28_general result JSON files.")
    parser.add_argument(
        "--base",
        default="/Users/sebastiancavada/Desktop/tmp_paris/tiny_vqa_creation/output/run_28_general",
        help="Base directory containing results_run_28_general-<number> folders.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory. Defaults to <base>/results_run_28_general-all.",
    )
    args = parser.parse_args()

    base = Path(args.base)
    if not base.exists():
        print(f"Base directory does not exist: {base}")
        return 1

    numbered_dirs = find_result_dirs(base)
    if not numbered_dirs:
        print(f"No folders found matching {RESULTS_PREFIX}<number> in {base}")
        return 1

    dirs = [p for _, p in numbered_dirs]
    out_dir = Path(args.output) if args.output else base / OUTPUT_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)

    models = list_model_files(dirs)
    if not models:
        print(f"No .json files found in {', '.join(d.name for d in dirs)}")
        return 1

    print(f"Merging {len(models)} model files from {len(dirs)} folders into {out_dir}")

    for model in models:
        merged_items: List[dict] = []
        missing_dirs: List[str] = []

        for d in dirs:
            path = d / model
            if not path.exists():
                missing_dirs.append(d.name)
                continue
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"Expected list in {path}, got {type(data).__name__}")
            merged_items.extend(data)

        merged_items, deduped, missing_idx = merge_lists(merged_items)

        out_path = out_dir / model
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(merged_items, f, indent=4, ensure_ascii=False)
            f.write("\n")

        if missing_dirs:
            print(f"- {model}: missing in {len(missing_dirs)} folders")
        if deduped:
            print(f"- {model}: removed {deduped} duplicate idx entries")
        if missing_idx:
            print(f"- {model}: {missing_idx} items missing idx")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
