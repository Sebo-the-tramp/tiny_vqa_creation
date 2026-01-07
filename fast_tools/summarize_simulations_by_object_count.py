#!/usr/bin/env python3

import argparse
import glob
import os
import re
import time
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple


def natural_key(value: str) -> List[object]:
    return [
        int(chunk) if chunk.isdigit() else chunk.lower()
        for chunk in re.split(r"(\d+)", value)
    ]


def collect_simulation_files(simulation_roots: Iterable[str]) -> List[Tuple[str, str]]:
    list_simulations: List[Tuple[str, str]] = []
    for simulation_root in simulation_roots:
        pattern = os.path.join(simulation_root, "**", "simulation.json")
        print(f"Searching for simulation files with pattern: {pattern}")
        for sim_file in glob.glob(pattern, recursive=True):
            list_simulations.append((sim_file, simulation_root))
    list_simulations.sort(key=lambda entry: natural_key(entry[0]))
    return list_simulations


def extract_object_count(sim_path: str, sim_root: str) -> Optional[int]:
    abs_root = os.path.normpath(os.path.abspath(sim_root))
    abs_path = os.path.normpath(os.path.abspath(sim_path))
    try:
        rel_path = os.path.relpath(abs_path, abs_root)
    except ValueError:
        rel_path = abs_path
    parts = rel_path.split(os.sep)
    if parts and parts[0].isdigit():
        return int(parts[0])
    normalized = abs_path.replace("\\", "/")
    match = re.search(r"/random/(\d+)/", normalized)
    if match:
        return int(match.group(1))
    return None


def build_bar(count: int, max_count: int, bar_width: int) -> Tuple[str, int]:
    if max_count <= 0:
        return "", max(bar_width, 0)
    if bar_width <= 0:
        return "#" * count, max_count
    ratio = count / max_count if max_count else 0
    bar_len = int(round(ratio * bar_width))
    if count > 0 and bar_len == 0:
        bar_len = 1
    return "#" * bar_len, bar_width


def print_summary(
    counts: Dict[int, int],
    total: int,
    unknown_count: int,
    bar_width: int,
) -> None:
    if not counts:
        print("No simulations counted.")
        return
    max_count = max(counts.values())
    max_obj = max(counts.keys())
    max_obj_len = len(str(max_obj))
    max_count_len = len(str(max_count))
    _, effective_width = build_bar(max_count, max_count, bar_width)
    if bar_width <= 0:
        print(f"Bar width: {effective_width} (auto = max count)")
    else:
        print(f"Bar width: {effective_width}")

    print("\nSummary by number of objects:")
    for obj_count in sorted(counts.keys()):
        count = counts[obj_count]
        bar, effective_width = build_bar(count, max_count, bar_width)
        pct = (count / total) if total else 0.0
        count_field = str(count).rjust(max_count_len)
        obj_field = str(obj_count).rjust(max_obj_len)
        print(
            f"{bar:<{effective_width}}  "
            f"N={obj_field}  "
            f"count={count_field}  "
            f"pct={pct:.1%}"
        )
    print("-" * 12)
    print(
        "RUN SUMMARY:\t"
        f"sims={total}\t"
        f"object_counts={len(counts)}\t"
        f"unknown={unknown_count}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize how many simulations exist for each object count "
            "using only the path structure."
        )
    )
    parser.add_argument(
        "--simulation-paths",
        nargs="+",
        default=["/data0/sebastian.cavada/datasets/simulations_v4/dl3dv/random"],
        help="Root folders that contain simulation runs (random folder for now).",
    )
    parser.add_argument(
        "--n-scenes",
        type=int,
        default=0,
        help="Optional limit on the number of simulations to scan (0 = all).",
    )
    parser.add_argument(
        "--bar-width",
        type=int,
        default=100,
        help=(
            "Width of the bar plot. Use 0 or a negative value to scale to the "
            "max count."
        ),
    )
    parser.add_argument(
        "--exclude-simulations-file",
        type=str,
        default="../answering_questions/problematic_paths.txt",
        help="Optional path to a txt file listing simulation.json paths to skip.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging for skipped or malformed paths.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.perf_counter()
    list_simulations = collect_simulation_files(args.simulation_paths)
    print(f"Found {len(list_simulations)} simulation files.")

    if not list_simulations:
        print("No simulation files found.")
        return

    if args.n_scenes > 0:
        list_simulations = list_simulations[: args.n_scenes]

    counts: Dict[int, int] = defaultdict(int)
    unknown_count = 0
    excluded_simulations = set()
    excluded_count = 0

    if args.exclude_simulations_file:
        exclude_path = os.path.expanduser(args.exclude_simulations_file)
        try:
            with open(exclude_path, "r") as handle:
                for line in handle:
                    path = line.strip()
                    if not path or path.startswith("#"):
                        continue
                    normalized = os.path.normpath(os.path.abspath(path))
                    excluded_simulations.add(normalized)
            print(
                f"Loaded {len(excluded_simulations)} simulations to skip from {exclude_path}"
            )
        except FileNotFoundError:
            print(
                f"Exclude file {exclude_path} not found. Continuing without exclusions."
            )

    for sim_file, root in list_simulations:
        normalized_sim_path = os.path.normpath(os.path.abspath(sim_file)).replace(
            f"{os.sep}simulation.json", ""
        )
        if normalized_sim_path in excluded_simulations:
            excluded_count += 1
            if args.verbose:
                print(f"Skipping excluded simulation: {sim_file}")
            continue
        obj_count = extract_object_count(sim_file, root)
        if obj_count is None:
            unknown_count += 1
            if args.verbose:
                print(f"Could not infer object count from path: {sim_file}")
            continue
        counts[obj_count] += 1

    if excluded_count:
        print(f"Skipped {excluded_count} simulations listed in exclude file.")
    print_summary(counts, sum(counts.values()), unknown_count, args.bar_width)
    elapsed = time.perf_counter() - start_time
    print(f"Done in {elapsed:.2f}s.")


if __name__ == "__main__":
    main()
