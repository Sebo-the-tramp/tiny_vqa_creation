from __future__ import annotations

import argparse
from fileinput import filename
from pathlib import Path
import matplotlib.pyplot as plt

from utils import (
    utils_mapping, 
    utils_read
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-path",
        default="../output",
    )
    parser.add_argument("--run-name", default="run_28_general")
    parser.add_argument(
        "--mode",
        choices=["all", "general", "image-only", "mixed"],
        default="all",
        help="Filter by model mode; all keeps all models.",
    )
    parser.add_argument(
        "--vqa-set",
        default="150K",
        help="VQA set to use (e.g., 10K, 30K, karo_5K).",
    )
    args = parser.parse_args()

    output_dir_models = Path("output") / args.run_name / "markers" / "models"
    output_dir_families = Path("output") / args.run_name / "markers" / "families"
    output_dir_models.mkdir(parents=True, exist_ok=True)
    output_dir_families.mkdir(parents=True, exist_ok=True)

    eval_df = utils_read.build_eval_df(args.run_name, args.base_path, vqa_set=args.vqa_set)
    metadata_path = "utils/metadata.json"
    
    # Desired output size in pixels
    width_px = 64
    height_px = 64
    dpi = 300

    fig = plt.figure(figsize=(width_px/dpi, height_px/dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])  # fill the whole canvas

    model_style, family_map = utils_mapping._build_model_style(
        metadata_path,
        group_by="model_id"
    )
    for model in eval_df["model_id"].unique():
        ax.clear()
        color, marker, size, edge = model_style[model]

        # Plot a single marker
        ax.scatter(0.5, 0.5, 
                color=color,
                s=size**2,
                alpha=0.8,
                edgecolor=edge,
                linewidth=1,
                marker=marker,
                )

        # Fix limits so marker position is stable
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Remove axes
        ax.axis('off')

        # Save PNG
        plt.savefig(output_dir_models / f"{model.lower()}.png", dpi=dpi, transparent=True)

    
    model_style, family_map = utils_mapping._build_model_style(
        metadata_path,
        group_by="model_family"
    )
    for family in eval_df["model_family"].unique():
        ax.clear()
        color, marker, size, edge = model_style[family]

        # Plot a single marker
        ax.scatter(0.5, 0.5, 
                color=color,
                s=size**2,
                alpha=0.8,
                edgecolor=edge,
                linewidth=1,
                marker=marker,
                )

        # Fix limits so marker position is stable
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Remove axes
        ax.axis('off')

        # Save PNG
        plt.savefig(output_dir_families / f"{family.lower()}.png", dpi=dpi, transparent=True)

    plt.close(fig)
        


if __name__ == "__main__":
    main()
