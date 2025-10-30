#!/usr/bin/env python3
"""
Create a grid video from simulation renders.

The script searches ``../data/simulations`` recursively for files ending in
``_fps-25_render.mp4``. A random selection of ``grid_size^2`` matches is
stitched into a single video using FFmpeg's ``xstack`` filter and written next
to this script. Export either MP4 (default) or GIF by toggling the ``--format``
flag.

Example usage (2x2 grid with the default pattern)::

    python stitch_simulations.py --grid-size 2
"""

from __future__ import annotations

import argparse
import random
import subprocess
from pathlib import Path
from typing import List, Sequence, Tuple

DEFAULT_ROOT = Path("..") / "data" / "simulations"
DEFAULT_SUFFIX = "_fps-25_render.mp4"


def discover_videos(root: Path, suffix: str) -> List[Path]:
    """Return all videos ending with ``suffix`` under ``root`` sorted by path."""
    if not root.exists():
        raise FileNotFoundError(f"Source directory not found: {root}")
    matches = sorted(
        path for path in root.rglob("*") if path.is_file() and path.name.endswith(suffix)
    )
    if not matches:
        raise FileNotFoundError(f"No videos ending with {suffix!r} found under {root}")
    return matches


def probe_resolution(video: Path) -> Tuple[int, int]:
    """Return ``(width, height)`` for the first video stream using ffprobe."""
    cmd: Sequence[str] = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0:s=x",
        str(video),
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )
    try:
        width_str, height_str = result.stdout.strip().split("x")
        width, height = int(width_str), int(height_str)
    except ValueError as exc:
        raise RuntimeError(f"Could not parse resolution from ffprobe output: {result.stdout!r}") from exc
    return width, height


def build_filter_graph(count: int, tile_w: int, tile_h: int, grid_size: int) -> str:
    """Return an ``xstack`` filter graph that tiles ``count`` videos."""
    parts: List[str] = []
    labels: List[str] = []
    for index in range(count):
        label = f"v{index}"
        filters = (
            f"[{index}:v]"
            f"setpts=PTS-STARTPTS,"
            f"scale={tile_w}:{tile_h}:force_original_aspect_ratio=decrease,"
            f"pad={tile_w}:{tile_h}:(ow-iw)/2:(oh-ih)/2:black"
            f"[{label}]"
        )
        parts.append(filters)
        labels.append(f"[{label}]")

    layout_entries: List[str] = []
    for index in range(count):
        row, col = divmod(index, grid_size)
        layout_entries.append(f"{col * tile_w}_{row * tile_h}")

    layout = "|".join(layout_entries)
    stacked = "".join(labels) + f"xstack=inputs={count}:layout={layout}:fill=black[out]"
    parts.append(stacked)
    return ";".join(parts)


def run_ffmpeg(
    inputs: Sequence[Path],
    filter_graph: str,
    output_label: str,
    output: Path,
    fmt: str,
    crf: int,
    preset: str,
) -> None:
    """Invoke ffmpeg with the provided filter graph."""
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    for video in inputs:
        command.extend(["-i", str(video)])
    command.extend(
        [
            "-filter_complex",
            filter_graph,
        ]
    )
    command.extend(["-map", output_label])

    if fmt == "gif":
        command.extend(
            [
                "-loop",
                "0",
                "-c:v",
                "gif",
            ]
        )
    else:
        command.extend(
            [
                "-c:v",
                "libx264",
                "-crf",
                str(crf),
                "-preset",
                preset,
                "-pix_fmt",
                "yuv420p",
            ]
        )

    command.append(str(output))

    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Directory containing simulation runs (default: {DEFAULT_ROOT}).",
    )
    parser.add_argument(
        "--suffix",
        default=DEFAULT_SUFFIX,
        help=f"File name suffix to match (default: {DEFAULT_SUFFIX}).",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=2,
        help="Number of tiles per row/column (default: 2 -> 2x2 grid).",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=20,
        help="Constant rate factor for libx264 (lower is higher quality).",
    )
    parser.add_argument(
        "--preset",
        default="medium",
        help="libx264 encoding preset (ultrafast, superfast, ..., placebo).",
    )
    parser.add_argument(
        "--format",
        choices=("mp4", "gif"),
        default="mp4",
        help="Output container/codec to use.",
    )
    parser.add_argument(
        "--tile-width",
        type=int,
        default=None,
        help="Target width for each tile (defaults to source width).",
    )
    parser.add_argument(
        "--tile-height",
        type=int,
        default=None,
        help="Target height for each tile (defaults to source height).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path (defaults to stitched_simulations.<format>).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for deterministic sampling of the input videos.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    grid_size = args.grid_size
    if grid_size < 1:
        raise ValueError("grid-size must be at least 1")

    videos = discover_videos(args.root.resolve(), args.suffix)
    required = grid_size * grid_size
    if len(videos) < required:
        raise RuntimeError(
            f"Found {len(videos)} videos, but {required} are required for a {grid_size}x{grid_size} grid."
        )

    rng = random.Random(args.seed)
    selected = rng.sample(videos, required)
    width, height = probe_resolution(selected[0])
    tile_w = args.tile_width or width
    tile_h = args.tile_height or height

    # Make sure dimensions are divisible by two for H.264.
    if tile_w % 2:
        tile_w += 1
    if tile_h % 2:
        tile_h += 1

    base_graph = build_filter_graph(len(selected), tile_w, tile_h, grid_size)
    if args.format == "gif":
        filter_graph = (
            base_graph
            + ";[out]split[out0][out1];[out0]palettegen[p];[out1][p]paletteuse[outgif]"
        )
        output_label = "[outgif]"
    else:
        filter_graph = base_graph
        output_label = "[out]"

    output_path = (
        args.output
        if args.output is not None
        else Path(__file__).resolve().with_name(f"stitched_simulations.{args.format}")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    run_ffmpeg(
        selected,
        filter_graph,
        output_label,
        output_path,
        args.format,
        args.crf,
        args.preset,
    )
    print(f"Wrote grid video to {output_path}")


if __name__ == "__main__":
    main()
