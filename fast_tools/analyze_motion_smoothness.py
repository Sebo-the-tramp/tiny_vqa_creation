#!/usr/bin/env python3

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import statistics
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

warnings.filterwarnings("ignore", message="CUDA initialization:", category=UserWarning)

from answering_questions.utils import seed_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect temporal slowdowns in simulation.json files.")
    parser.add_argument(
        "--simulation-paths",
        nargs="+",
        required=True,
        help="Root folders that contain simulation runs.",
    )
    parser.add_argument("--output-path", default="./analysis_outputs", help="Directory where reports are written.")
    parser.add_argument("--run-name", default="motion_smoothness", help="Sub-folder used inside the output path.")
    parser.add_argument("--n-scenes", type=int, default=0, help="Optional limit for the number of simulations to scan.")
    parser.add_argument("--seed", type=int, default=0, help="Seed used to make the run deterministic.")
    parser.add_argument(
        "--dt-relative-tolerance",
        type=float,
        default=0.15,
        help="Tolerated relative error for the timestep before declaring a slowdown.",
    )
    parser.add_argument(
        "--max-logged-dt-intervals",
        type=int,
        default=10,
        help="Maximum number of timestep anomalies stored per simulation.",
    )
    parser.add_argument(
        "--speed-jump-ratio",
        type=float,
        default=6.0,
        help="Minimum ratio between consecutive speeds for a jump to be suspicious.",
    )
    parser.add_argument(
        "--speed-drop-ratio",
        type=float,
        default=6.0,
        help="Minimum ratio between consecutive speeds to treat a slowdown as suspicious.",
    )
    parser.add_argument(
        "--speed-jump-min-delta",
        type=float,
        default=0.8,
        help="Absolute delta in m/s required for a speed change anomaly.",
    )
    parser.add_argument(
        "--min-speed-for-ratio",
        type=float,
        default=0.2,
        help="Minimum speed that must be reached before ratio based checks kick in.",
    )
    parser.add_argument(
        "--jerk-threshold",
        type=float,
        default=200.0,
        help="Minimum jerk (delta speed divided by delta time) for a spike.",
    )
    parser.add_argument(
        "--freeze-speed-threshold",
        type=float,
        default=0.02,
        help="Speed below which an object is considered frozen.",
    )
    parser.add_argument(
        "--freeze-duration",
        type=int,
        default=5,
        help="Number of consecutive frames required to call out a freeze.",
    )
    parser.add_argument(
        "--max-speed-threshold",
        type=float,
        default=80.0,
        help="Speed limit in m/s that is considered implausible for the simulation setup.",
    )
    parser.add_argument(
        "--summary-limit",
        type=int,
        default=50,
        help="Maximum number of flagged simulations included in the text summary.",
    )
    return parser.parse_args()


def natural_key(value: str) -> List[Any]:
    return [int(chunk) if chunk.isdigit() else chunk.lower() for chunk in re.split(r"(\d+)", value)]


def collect_simulation_files(simulation_roots: Iterable[str]) -> List[str]:
    list_simulations: List[str] = []
    for simulation_root in simulation_roots:
        pattern = os.path.join(simulation_root, "**", "simulation.json")
        print(f"Searching for simulation files with pattern: {pattern}")
        for sim_file in glob.glob(pattern, recursive=True):
            list_simulations.append(sim_file)
    list_simulations.sort(key=natural_key)
    return list_simulations


def extract_frames(simulation_blob: Dict[str, Any]) -> List[Tuple[float, Dict[str, Any]]]:
    frames: List[Tuple[float, Dict[str, Any]]] = []
    for ts_str, frame in simulation_blob.items():
        try:
            ts_val = float(ts_str)
        except ValueError:
            continue
        frames.append((ts_val, frame))
    frames.sort(key=lambda entry: entry[0])
    return frames


def compute_time_stats(
    frames: List[Tuple[float, Dict[str, Any]]],
    expected_dt: float | None,
    rel_tol: float,
    max_logged: int,
) -> Dict[str, Any]:
    if len(frames) < 2:
        return {
            "expected_dt": expected_dt,
            "median_dt": None,
            "min_dt": None,
            "max_dt": None,
            "bad_interval_count": 0,
            "max_ratio": 0.0,
            "issue_score": 0.0,
            "bad_intervals": [],
            "slow_fast_pairs": [],
        }
    deltas: List[float] = []
    for idx in range(len(frames) - 1):
        dt = frames[idx + 1][0] - frames[idx][0]
        if dt > 0:
            deltas.append(dt)
    if not deltas:
        return {
            "expected_dt": expected_dt,
            "median_dt": None,
            "min_dt": None,
            "max_dt": None,
            "bad_interval_count": 0,
            "max_ratio": 0.0,
            "issue_score": 0.0,
            "bad_intervals": [],
            "slow_fast_pairs": [],
        }
    median_dt = statistics.median(deltas)
    target_dt = expected_dt or median_dt
    if not target_dt:
        target_dt = median_dt or 1.0
    bad_intervals = []
    max_ratio = 0.0
    slow_fast_pairs = []
    for idx, dt in enumerate(deltas):
        ratio = abs(dt - target_dt) / target_dt
        if ratio > rel_tol:
            bad_intervals.append({"interval_index": idx, "dt": dt, "ratio": ratio})
            max_ratio = max(max_ratio, ratio)
        if idx < len(deltas) - 1:
            next_dt = deltas[idx + 1]
            slow_condition = dt > target_dt * (1 + rel_tol)
            fast_condition = next_dt < target_dt * (1 - rel_tol)
            if slow_condition and fast_condition:
                slow_fast_pairs.append(
                    {
                        "slow_interval_index": idx,
                        "slow_dt": dt,
                        "fast_interval_index": idx + 1,
                        "fast_dt": next_dt,
                    }
                )
    bad_interval_count = len(bad_intervals)
    issue_score = bad_interval_count / len(deltas)
    return {
        "expected_dt": expected_dt,
        "median_dt": median_dt,
        "min_dt": min(deltas),
        "max_dt": max(deltas),
        "bad_interval_count": bad_interval_count,
        "max_ratio": max_ratio,
        "issue_score": issue_score,
        "bad_intervals": bad_intervals[:max_logged],
        "slow_fast_pairs": slow_fast_pairs[:max_logged],
    }


def build_tracks(frames: List[Tuple[float, Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    tracks: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for time_val, frame in frames:
        objects = frame.get("objects", {})
        frame_idx = frame.get("frame_idx")
        for obj_id, obj_state in objects.items():
            center = obj_state.get("obb", {}).get("center")
            if not center or len(center) != 3:
                continue
            tracks[obj_id].append(
                {
                    "time": time_val,
                    "frame_idx": frame_idx,
                    "center": center,
                }
            )
    return tracks


def vector_length(v: Iterable[float]) -> float:
    return math.sqrt(sum(component * component for component in v))


def collect_velocity_samples(track: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    velocities: List[Dict[str, Any]] = []
    for prev, curr in zip(track, track[1:]):
        dt = curr["time"] - prev["time"]
        if dt <= 0:
            continue
        disp = [
            curr["center"][axis] - prev["center"][axis]
            for axis in range(3)
        ]
        distance = vector_length(disp)
        speed = distance / dt
        velocities.append(
            {
                "frame_idx": curr["frame_idx"],
                "time": curr["time"],
                "speed": speed,
                "dt": dt,
                "disp": disp,
            }
        )
    return velocities


def detect_motion_anomalies(
    velocities: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    anomalies: List[Dict[str, Any]] = []
    if not velocities:
        return anomalies

    for prev, curr in zip(velocities, velocities[1:]):
        prev_speed = prev["speed"]
        curr_speed = curr["speed"]
        base_speed = min(prev_speed, curr_speed)
        top_speed = max(prev_speed, curr_speed)
        ratio = top_speed / (base_speed + 1e-9)
        delta_speed = abs(curr_speed - prev_speed)
        dt_avg = max(1e-9, 0.5 * (prev["dt"] + curr["dt"]))
        jerk = delta_speed / dt_avg

        if (
            base_speed > args.min_speed_for_ratio
            and ratio > args.speed_jump_ratio
            and delta_speed > args.speed_jump_min_delta
        ):
            anomalies.append(
                {
                    "type": "speed_jump",
                    "frame_idx": curr["frame_idx"],
                    "time": curr["time"],
                    "prev_speed": prev_speed,
                    "curr_speed": curr_speed,
                    "ratio": ratio,
                }
            )

        if (
            prev_speed < args.freeze_speed_threshold
            and curr_speed > args.min_speed_for_ratio
            and (curr_speed - prev_speed) > args.speed_jump_min_delta
        ):
            anomalies.append(
                {
                    "type": "stall_to_jump",
                    "frame_idx": curr["frame_idx"],
                    "time": curr["time"],
                    "prev_speed": prev_speed,
                    "curr_speed": curr_speed,
                }
            )

        if (
            prev_speed > args.min_speed_for_ratio
            and curr_speed < max(args.freeze_speed_threshold, prev_speed / args.speed_drop_ratio)
            and (prev_speed - curr_speed) > args.speed_jump_min_delta
        ):
            anomalies.append(
                {
                    "type": "sudden_slowdown",
                    "frame_idx": curr["frame_idx"],
                    "time": curr["time"],
                    "prev_speed": prev_speed,
                    "curr_speed": curr_speed,
                    "ratio": prev_speed / (curr_speed + 1e-9),
                }
            )

        if jerk > args.jerk_threshold:
            anomalies.append(
                {
                    "type": "jerk_spike",
                    "frame_idx": curr["frame_idx"],
                    "time": curr["time"],
                    "jerk": jerk,
                    "delta_speed": delta_speed,
                }
            )

    freeze_run = 0
    freeze_start_idx = None
    for sample in velocities:
        if sample["speed"] < args.freeze_speed_threshold:
            freeze_run += 1
            if freeze_start_idx is None:
                freeze_start_idx = sample["frame_idx"]
        else:
            if freeze_run >= args.freeze_duration:
                anomalies.append(
                    {
                        "type": "freeze",
                        "frame_idx": sample["frame_idx"],
                        "time": sample["time"],
                        "duration": freeze_run,
                    }
                )
            freeze_run = 0
            freeze_start_idx = None
    if freeze_run >= args.freeze_duration and velocities:
        anomalies.append(
            {
                "type": "freeze",
                "frame_idx": velocities[-1]["frame_idx"],
                "time": velocities[-1]["time"],
                "duration": freeze_run,
            }
        )

    for sample in velocities:
        if sample["speed"] > args.max_speed_threshold:
            anomalies.append(
                {
                    "type": "speed_limit",
                    "frame_idx": sample["frame_idx"],
                    "time": sample["time"],
                    "speed": sample["speed"],
                }
            )
            break

    return anomalies


def analyze_object_tracks(
    tracks: Dict[str, List[Dict[str, Any]]],
    object_meta: Dict[str, Any],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    object_summaries: List[Dict[str, Any]] = []
    for obj_id, track in tracks.items():
        velocities = collect_velocity_samples(track)
        speeds = [sample["speed"] for sample in velocities]
        median_speed = statistics.median(speeds) if speeds else 0.0
        mean_speed = statistics.mean(speeds) if speeds else 0.0
        speed_deltas = [abs(cur - prev) for prev, cur in zip(speeds, speeds[1:])]
        smoothness = 0.0
        if speed_deltas:
            smoothness = statistics.mean(speed_deltas) / (mean_speed + 1e-6)

        anomalies = detect_motion_anomalies(velocities, args)
        issue_score = len(anomalies) / max(1, len(velocities))
        summary = {
            "object_id": obj_id,
            "name": object_meta.get(obj_id, {})
            .get("description", {})
            .get("object_name"),
            "num_samples": len(track),
            "num_velocity_samples": len(velocities),
            "max_speed": max(speeds) if speeds else 0.0,
            "median_speed": median_speed,
            "smoothness": smoothness,
            "issue_score": issue_score,
        }
        if anomalies:
            summary["anomalies"] = anomalies
        object_summaries.append(summary)
    return object_summaries


def analyze_simulation(sim_path: str, args: argparse.Namespace) -> Dict[str, Any]:
    with open(sim_path, "r") as handle:
        payload = json.load(handle)
    frames = extract_frames(payload.get("simulation", {}))
    expected_dt = (
        payload.get("config", {})
        .get("render", {})
        .get("renderstep")
    )
    time_stats = compute_time_stats(frames, expected_dt, args.dt_relative_tolerance, args.max_logged_dt_intervals)
    tracks = build_tracks(frames)
    object_summaries = analyze_object_tracks(tracks, payload.get("objects", {}), args)

    objects_with_issues = [entry for entry in object_summaries if entry.get("anomalies")]
    dominant_issue = []
    if time_stats["bad_interval_count"] > 0:
        dominant_issue.append("time_jitter")
    if objects_with_issues:
        dominant_issue.append("motion_spikes")

    result = {
        "simulation_path": sim_path,
        "num_objects": len(payload.get("objects", {})),
        "num_frames": len(frames),
        "time_stats": time_stats,
        "objects_with_issues": objects_with_issues,
        "flagged": bool(dominant_issue),
        "dominant_issue": dominant_issue,
    }
    return result


def write_reports(
    results: List[Dict[str, Any]],
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    report_path = output_dir / "motion_analysis.json"
    report_payload = {
        "parameters": {
            key: value
            for key, value in vars(args).items()
            if key not in {"seed"}
        },
        "num_simulations": len(results),
        "results": results,
    }
    with open(report_path, "w") as handle:
        json.dump(report_payload, handle, indent=2)

    flagged = [entry for entry in results if entry.get("flagged")]
    summary_path = output_dir / "motion_analysis.txt"
    with open(summary_path, "w") as handle:
        handle.write(f"Total simulations analyzed: {len(results)}\n")
        handle.write(f"Flagged simulations: {len(flagged)}\n")
        handle.write(
            f"Parameters: dt_tol={args.dt_relative_tolerance}, speed_jump_ratio={args.speed_jump_ratio}, jerk={args.jerk_threshold}\n\n"
        )
        for entry in flagged[: args.summary_limit]:
            handle.write(f"{entry['simulation_path']}\n")
            t_stats = entry.get("time_stats", {})
            if t_stats.get("bad_interval_count"):
                handle.write(
                    f"  Time anomalies: {t_stats['bad_interval_count']} (max ratio {t_stats['max_ratio']:.3f})\n"
                )
            for obj in entry.get("objects_with_issues", []):
                handle.write(
                    f"  Object {obj.get('object_id')} ({obj.get('name')}): issue_score={obj.get('issue_score'):.3f}\n"
                )
                for issue in obj.get("anomalies", [])[:5]:
                    handle.write(
                        f"    - {issue['type']} @ frame {issue.get('frame_idx')} time {issue.get('time')}\n"
                    )
            handle.write("\n")

    buggy_path = output_dir / "buggy_simulations.txt"
    with open(buggy_path, "w") as handle:
        if not flagged:
            handle.write("No simulations with flagged objects or time anomalies were detected.\n")
        for entry in flagged:
            handle.write(f"{entry['simulation_path']}\n")
            t_stats = entry.get("time_stats", {})
            if t_stats.get("bad_interval_count"):
                handle.write(
                    f"  Time anomalies: {t_stats['bad_interval_count']} intervals deviating (max ratio {t_stats['max_ratio']:.3f})\n"
                )
            if t_stats.get("slow_fast_pairs"):
                handle.write("  Slow-then-fast timestep pairs detected:\n")
                for pair in t_stats["slow_fast_pairs"]:
                    handle.write(
                        f"    slow interval {pair['slow_interval_index']} dt={pair['slow_dt']:.6f} -> fast interval {pair['fast_interval_index']} dt={pair['fast_dt']:.6f}\n"
                    )
            for obj in entry.get("objects_with_issues", []):
                anomalies = obj.get("anomalies", [])
                handle.write(
                    f"  Object {obj.get('object_id')} ({obj.get('name')}): {len(anomalies)} anomalies, max_speed={obj.get('max_speed'):.2f} m/s\n"
                )
                for issue in anomalies:
                    extra = ""
                    if issue["type"] in {"speed_jump", "sudden_slowdown"}:
                        extra = f" ratio={issue.get('ratio'):.2f}"
                    elif issue["type"] == "jerk_spike":
                        extra = f" jerk={issue.get('jerk'):.2f}"
                    elif issue["type"] == "stall_to_jump":
                        extra = f" prev={issue.get('prev_speed'):.3f} curr={issue.get('curr_speed'):.3f}"
                    elif issue["type"] == "freeze":
                        extra = f" duration={issue.get('duration')}"
                    elif issue["type"] == "speed_limit":
                        extra = f" speed={issue.get('speed'):.2f}"
                    handle.write(
                        f"    - {issue['type']} @ frame {issue.get('frame_idx')} time {issue.get('time')} {extra}\n"
                    )
            handle.write("\n")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_path) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_utils.seed_everything(args.seed)

    all_vqa: List[Any] = []
    simulation_roots = args.simulation_paths
    list_simulations = collect_simulation_files(simulation_roots)
    if args.n_scenes and args.n_scenes > 0:
        list_simulations = list_simulations[: args.n_scenes]

    results: List[Dict[str, Any]] = []
    for idx, sim_file in enumerate(list_simulations, 1):
        print(f"[{idx}/{len(list_simulations)}] Analyzing {sim_file}")
        try:
            analysis = analyze_simulation(sim_file, args)
        except Exception as exc:  # pragma: no cover - defensive logging
            analysis = {
                "simulation_path": sim_file,
                "flagged": True,
                "error": str(exc),
            }
        results.append(analysis)

    write_reports(results, output_dir, args)
    print(f"Analysis stored in {output_dir} (motion_analysis.json/.txt and buggy_simulations.txt)")
    _ = all_vqa  # placeholder to mirror the workflow shown in the instructions


if __name__ == "__main__":
    main()
