#!/usr/bin/env python3
import argparse
import glob
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterable, List, Optional, Tuple


def natural_key(path: str) -> List[object]:
    """Natural sort helper that keeps numeric fragments in order."""
    return [int(txt) if txt.isdigit() else txt.lower() for txt in re.split(r"(\d+)", path)]


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


def get_center(obj: dict) -> Optional[Tuple[float, float, float]]:
    """Extract the OBB center from an object record."""
    try:
        center = obj["obb"]["center"]
        if isinstance(center, list) and len(center) >= 3:
            return float(center[0]), float(center[1]), float(center[2])
    except Exception:
        pass
    return None


def augment_single_simulation(in_path: str, out_path: Optional[str] = None) -> str:
    """Augment a single simulation.json and write the kinematics JSON."""
    with open(in_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    sim = data["simulation"]
    times = sorted(sim.keys(), key=lambda s: float(s))
    last_state: dict[str, Tuple[float, float, float, float, Optional[float], Optional[float], Optional[float]]] = {}

    for t_str in times:
        t = float(t_str)
        frame = sim[t_str]
        for oid, obj in frame.get("objects", {}).items():
            center = get_center(obj)
            if center is None:
                continue
            vx = vy = vz = speed = ax = ay = az = accel = None
            if oid in last_state:
                (
                    t_prev,
                    px_prev,
                    py_prev,
                    pz_prev,
                    vx_prev,
                    vy_prev,
                    vz_prev,
                ) = last_state[oid]
                dt = t - t_prev
                if dt != 0:
                    vx = (center[0] - px_prev) / dt
                    vy = (center[1] - py_prev) / dt
                    vz = (center[2] - pz_prev) / dt
                    speed = (vx * vx + vy * vy + vz * vz) ** 0.5
                    if vx_prev is not None:
                        ax = (vx - vx_prev) / dt
                        ay = (vy - vy_prev) / dt
                        az = (vz - vz_prev) / dt
                        accel = (ax * ax + ay * ay + az * az) ** 0.5
            obj["kinematics"] = {
                "v": [vx, vy, vz],
                "speed": speed,
                "a": [ax, ay, az],
                "accel": accel,
            }
            last_state[oid] = (t, center[0], center[1], center[2], vx, vy, vz)

    destination = out_path or os.path.join(os.path.dirname(in_path), "simulation_kinematics.json")
    with open(destination, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    print(f"[{in_path}] wrote kinematics to {destination}")
    return destination


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Augment simulations with simple kinematics for object OBB centers."
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
        print("--out-path can only be used when processing a single simulation.", file=sys.stderr)
        return 2

    if len(simulation_files) == 1:
        sim_file = simulation_files[0]
        try:
            augment_single_simulation(sim_file, out_path=args.out_path)
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
                augment_single_simulation(sim_file)
            except Exception as exc:
                print(f"Failed to process {sim_file}: {exc}", file=sys.stderr)
                return 1
        return 0

    status = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(augment_single_simulation, sim_path, None): sim_path for sim_path in simulation_files}
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
