#!/usr/bin/env python3
"""Summarize question counts by YMS variation and category from a QA JSON dump."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt


VARIATION_PATTERN = re.compile(r"/yms-variations/([^/]+)/([^/]+)/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Report counts by YMS variation and sub-category for a QA JSON file."
        )
    )
    parser.add_argument(
        "json_path",
        type=Path,
        help="Path to the JSON file containing a list of QA samples.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/yms_variation_distribution"),
        help="Directory where the figures and CSVs will be written.",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save bar plots to the output directory.",
    )
    return parser.parse_args()


def load_entries(json_path: Path) -> Iterable[dict]:
    with json_path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON root must be a list of QA entries.")
    return data


def parse_variation(simulation_path: str | None) -> Tuple[str | None, str | None]:
    if not simulation_path:
        return None, None
    match = VARIATION_PATTERN.search(simulation_path)
    if not match:
        return None, None
    return match.group(1), match.group(2)


def plot_bar(counts: Iterable[Tuple[str, int]], output_path: Path, title: str) -> None:
    labels, values = zip(*counts)
    fig_height = max(6, 0.3 * len(labels))
    fig, ax = plt.subplots(figsize=(12, fig_height))
    y_pos = range(len(labels))
    ax.barh(y_pos, values, color="#4C72B0")
    ax.set_yticks(y_pos, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Number of samples")
    ax.set_title(title)
    for idx, value in enumerate(values):
        ax.text(value, idx, f"{value:,}", va="center", ha="left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Wrote {output_path}")


def write_counter_csv(
    counter: Counter, output_path: Path, header: Tuple[str, str]
) -> None:
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow((*header,))
        for label, count in counter.most_common():
            writer.writerow((label, count))
    print(f"Wrote {output_path}")


def write_variation_yms_csv(counter: Counter, output_path: Path) -> None:
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("variation", "yms_category", "count"))
        for (variation, yms_category), count in sorted(counter.items()):
            writer.writerow((variation, yms_category, count))
    print(f"Wrote {output_path}")


def write_variation_yms_sub_csv(counter: Counter, output_path: Path) -> None:
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("variation", "yms_category", "sub_category", "count"))
        for (variation, yms_category, sub_category), count in sorted(counter.items()):
            writer.writerow((variation, yms_category, sub_category, count))
    print(f"Wrote {output_path}")


def render_count_bar(counts: Counter, width: int = 30) -> str:
    if not counts:
        return ""
    max_count = max(counts.values())
    lines = []
    for label, count in counts.most_common():
        bar_len = int(round((count / max_count) * width)) if max_count else 0
        bar = "#" * bar_len
        lines.append(f"{label:>8} | {bar:<{width}} {count}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    entries = load_entries(args.json_path)

    variation_counts = Counter()
    yms_counts = Counter()
    variation_yms_counts = Counter()
    variation_yms_sub_counts = Counter()
    missing_variation = 0

    for entry in entries:
        variation, yms_category = parse_variation(entry.get("simulation_id", ""))
        if not variation or not yms_category:
            missing_variation += 1
            continue
        sub_category = entry.get("sub_category") or "unknown"
        variation_counts[variation] += 1
        yms_counts[yms_category] += 1
        variation_yms_counts[(variation, yms_category)] += 1
        variation_yms_sub_counts[(variation, yms_category, sub_category)] += 1

    total = sum(variation_counts.values())
    print(f"Loaded {len(entries)} samples; parsed {total} with YMS variation metadata.")
    if missing_variation:
        print(f"Missing variation metadata for {missing_variation} samples.")

    print("Variation counts:")
    print(render_count_bar(variation_counts))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    write_counter_csv(
        variation_counts,
        args.output_dir / "variation_counts.csv",
        ("variation", "count"),
    )
    write_counter_csv(
        yms_counts,
        args.output_dir / "yms_category_counts.csv",
        ("yms_category", "count"),
    )
    write_variation_yms_csv(
        variation_yms_counts, args.output_dir / "variation_by_yms_category.csv"
    )
    write_variation_yms_sub_csv(
        variation_yms_sub_counts,
        args.output_dir / "variation_by_yms_sub_category.csv",
    )

    if args.save_images:
        plot_bar(
            variation_counts.most_common(),
            args.output_dir / "variation_counts.png",
            "YMS variation counts",
        )
        plot_bar(
            yms_counts.most_common(),
            args.output_dir / "yms_category_counts.png",
            "YMS category counts",
        )


if __name__ == "__main__":
    main()
