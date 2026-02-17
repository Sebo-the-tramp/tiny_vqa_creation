#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _sorted_input_dirs(dirs: Iterable[Path], prefix: str) -> List[Path]:
    pattern = re.compile(rf"^{re.escape(prefix)}-(\d+)$")

    def key(p: Path):
        m = pattern.match(p.name)
        if m:
            return (0, int(m.group(1)))
        return (1, p.name)

    return sorted(dirs, key=key)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _all_items_have_id(data: List[Any], id_key: str) -> bool:
    for item in data:
        if not isinstance(item, dict) or id_key not in item:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge results across multiple numbered result folders into a single folder, grouped by model file name."
        )
    )
    parser.add_argument(
        "parent",
        type=Path,
        help="Parent directory containing result folders (e.g., /.../output/run_28_general)",
    )
    parser.add_argument(
        "prefix",
        help="Folder prefix to match (e.g., results_run_28_general)",
    )
    parser.add_argument(
        "--output-suffix",
        default="all",
        help="Suffix for the merged output folder (default: all)",
    )
    parser.add_argument(
        "--include-base",
        action="store_true",
        help="Include the base folder without a numeric suffix if it exists.",
    )
    parser.add_argument(
        "--glob",
        default="*.json",
        help="Glob pattern for files to merge within each folder (default: *.json)",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Dedupe list entries by id key (requires list of dicts with the id key).",
    )
    parser.add_argument(
        "--id-key",
        default="idx",
        help="ID key used for duplicate detection/deduping (default: idx)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting files in the output folder if it already exists.",
    )

    args = parser.parse_args()

    parent: Path = args.parent
    prefix: str = args.prefix
    output_suffix: str = args.output_suffix
    include_base: bool = args.include_base
    glob_pattern: str = args.glob
    dedupe: bool = args.dedupe
    id_key: str = args.id_key
    overwrite: bool = args.overwrite

    if not parent.exists():
        raise SystemExit(f"Parent directory not found: {parent}")

    output_dir = parent / f"{prefix}-{output_suffix}"

    input_dirs = []
    for child in parent.iterdir():
        if not child.is_dir():
            continue
        if child.name == output_dir.name:
            continue
        if child.name.startswith(prefix + "-"):
            input_dirs.append(child)

    if include_base:
        base_dir = parent / prefix
        if base_dir.is_dir():
            input_dirs.append(base_dir)

    input_dirs = _sorted_input_dirs(input_dirs, prefix)

    if not input_dirs:
        raise SystemExit(f"No input folders found for prefix '{prefix}' in {parent}")

    if output_dir.exists() and not overwrite:
        existing = list(output_dir.glob("*") if output_dir.is_dir() else [])
        if existing:
            raise SystemExit(
                f"Output directory {output_dir} exists and is not empty. Use --overwrite to replace files."
            )

    output_dir.mkdir(parents=True, exist_ok=True)

    files_by_dir: Dict[Path, List[Path]] = {}
    all_filenames = set()

    for d in input_dirs:
        files = [p for p in d.glob(glob_pattern) if p.is_file()]
        files_by_dir[d] = files
        all_filenames.update(p.name for p in files)

    if not all_filenames:
        raise SystemExit("No files matched the glob pattern in the input folders.")

    total_files = 0
    warnings: List[str] = []

    for filename in sorted(all_filenames):
        parts: List[Tuple[Path, Any]] = []
        types = set()
        all_have_ids = True

        for d in input_dirs:
            path = d / filename
            if not path.exists():
                continue
            data = _load_json(path)
            parts.append((d, data))
            types.add(type(data))
            if isinstance(data, list):
                if not _all_items_have_id(data, id_key):
                    all_have_ids = False

        if not parts:
            continue

        if len(types) > 1:
            raise SystemExit(f"Type mismatch for {filename}: found {', '.join(t.__name__ for t in types)}")

        data_type = types.pop()
        out_path = output_dir / filename

        if data_type is list:
            combined: List[Any] = []
            seen_ids = set()
            dupes = 0

            for _, data in parts:
                for item in data:
                    if all_have_ids:
                        item_id = item[id_key]
                        if item_id in seen_ids:
                            dupes += 1
                            if dedupe:
                                continue
                        else:
                            seen_ids.add(item_id)
                    combined.append(item)

            if dedupe and not all_have_ids:
                warnings.append(
                    f"{filename}: --dedupe requested but items missing '{id_key}'; concatenated without dedupe."
                )
            if dupes and all_have_ids:
                warnings.append(f"{filename}: detected {dupes} duplicate '{id_key}' values")

            with out_path.open("w", encoding="utf-8") as f:
                json.dump(combined, f, ensure_ascii=False, indent=4)

        elif data_type is dict:
            combined_dict: Dict[Any, Any] = {}
            collisions = 0
            for _, data in parts:
                for k, v in data.items():
                    if k in combined_dict and combined_dict[k] != v:
                        collisions += 1
                        # Keep first value by default
                        continue
                    combined_dict.setdefault(k, v)

            if collisions:
                warnings.append(f"{filename}: {collisions} key collisions (kept first value)")

            with out_path.open("w", encoding="utf-8") as f:
                json.dump(combined_dict, f, ensure_ascii=False, indent=4)

        else:
            raise SystemExit(f"Unsupported JSON type in {filename}: {data_type.__name__}")

        total_files += 1

    print(f"Merged {total_files} files from {len(input_dirs)} folders into {output_dir}")
    for w in warnings:
        print(f"WARN: {w}")


if __name__ == "__main__":
    main()


# python ../answering_questions/merge_results_by_model.py ../output/run_28_general_30K results_run_28_general_30K
