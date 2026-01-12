#!/usr/bin/env python3
import argparse
import glob
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterable, List, Optional


DEFAULT_DROP_PATHS = [
    # Explicitly strip collision point clouds.
    "simulation.*.objects.*.collisions.*.points",
    "simulation.*.objects.*.collisions.*.normals",
]


def natural_key(path: str) -> List[object]:
    """Natural sort helper that keeps numeric fragments in order."""
    return [
        int(txt) if txt.isdigit() else txt.lower() for txt in re.split(r"(\d+)", path)
    ]


def find_simulation_files(simulation_root: str) -> List[str]:
    """Return all simulation.json paths under the provided root."""
    abs_root = os.path.abspath(simulation_root)
    if os.path.isfile(abs_root):
        return [abs_root]

    pattern = os.path.join(abs_root, "**", "simulation.json")
    print(f"Searching for simulation files with pattern: {pattern}")
    matches = glob.glob(pattern, recursive=True)
    matches.sort(key=natural_key)
    return matches


def parse_drop_paths(path: Optional[str]) -> List[str]:
    if not path:
        return DEFAULT_DROP_PATHS[:]
    drop_paths = []
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            drop_paths.append(line)
    return drop_paths


def drop_by_paths(data: dict, drop_paths: List[str]) -> None:
    for path in drop_paths:
        parts = path.split(".")
        _drop_path(data, parts)


def _drop_path(source, parts: List[str]) -> None:
    if not parts:
        return

    head, *tail = parts

    if head == "*":
        if not isinstance(source, dict):
            return
        for value in source.values():
            _drop_path(value, tail)
        return

    if not isinstance(source, dict) or head not in source:
        return

    if not tail:
        source.pop(head, None)
        return

    _drop_path(source[head], tail)


def minify_single_simulation(
    in_path: str,
    out_path: Optional[str] = None,
    drop_paths: Optional[List[str]] = None,
) -> str:
    before_size = os.path.getsize(in_path)
    with open(in_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    drop_paths = drop_paths or DEFAULT_DROP_PATHS
    if drop_paths:
        drop_by_paths(data, drop_paths)

    destination = out_path or in_path.replace("simulation.json", "simulation_min.json")
    with open(destination, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=4, ensure_ascii=True)
    after_size = os.path.getsize(destination)
    factor = (before_size / after_size) if after_size else 0.0
    print(
        f"[{in_path}] size {before_size} -> {after_size} bytes "
        f"(x{factor:.2f})"
    )
    print(f"[{in_path}] wrote minified simulation to {destination}")
    return destination


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create a minimal JSON snapshot from simulation.json files."
    )
    parser.add_argument(
        "simulation_path",
        help="Path to a simulation.json file or a directory tree to search.",
    )
    parser.add_argument(
        "--out-path",
        default=None,
        help="Optional output path (only valid when a single simulation file is processed).",
    )
    parser.add_argument(
        "--drop-paths-file",
        default=None,
        help="Optional file with dotpaths to drop (one per line, # for comments).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=0,
        help="Number of worker processes. 0 uses CPU count, 1 forces sequential execution.",
    )
    args = parser.parse_args(argv)

    simulation_files = find_simulation_files(args.simulation_path)
    if not simulation_files:
        print("No simulation.json files found.", file=sys.stderr)
        return 1

    if args.out_path and len(simulation_files) > 1:
        print(
            "--out-path can only be used when processing a single simulation.",
            file=sys.stderr,
        )
        return 2

    drop_paths = parse_drop_paths(args.drop_paths_file)

    if len(simulation_files) == 1:
        sim_file = simulation_files[0]
        try:
            minify_single_simulation(
                sim_file,
                out_path=args.out_path,
                drop_paths=drop_paths,
            )
        except Exception as exc:
            print(f"Failed to process {sim_file}: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.max_workers < 0:
        print("--max-workers must be >= 0.", file=sys.stderr)
        return 2
    if args.max_workers in (0, None):
        workers = os.cpu_count() or 1
    else:
        workers = args.max_workers

    if workers <= 1:
        for sim_file in simulation_files:
            try:
                minify_single_simulation(
                    sim_file,
                    drop_paths=drop_paths,
                )
            except Exception as exc:
                print(f"Failed to process {sim_file}: {exc}", file=sys.stderr)
                return 1
        return 0

    status = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(
                minify_single_simulation,
                sim_path,
                None,
                drop_paths,
            ): sim_path
            for sim_path in simulation_files
        }
        for future in as_completed(future_map):
            sim_file = future_map[future]
            try:
                future.result()
            except Exception as exc:
                status = 1
                print(f"Failed to process {sim_file}: {exc}", file=sys.stderr)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
