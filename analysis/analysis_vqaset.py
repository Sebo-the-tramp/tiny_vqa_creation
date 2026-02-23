from __future__ import annotations

import argparse
from fileinput import filename
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import tqdm

import utils.utils_read

if not Path("/scratch/project/eu-25-92/composite_physics/dataset/simulation_v4/").exists():
    utils.utils_read.sim_path_fct = lambda x: x.replace("/scratch/project/eu-25-92/composite_physics/dataset/simulation_v4/", "/nfs/data/workspaces/rdechare/codes/physics-sim/output/sims/v4/")

# from utils.utils_read import load_results, _sanitize_answer
import utils.utils_graph as utils_graph
import utils.utils_graph_correlation as utils_graph_correlation
from utils.utils_graph_correlation import (
    create_category_accuracy,
    create_num_objects_violin_grid
)


def plot_row_counts_by_column(
    df: pd.DataFrame,
    column: str,
    *,
    top_n: int | None = 20,
    dropna: bool = False,
    sort_desc: bool = True,
    figsize: tuple[float, float] | None = None,
    show_values: bool = True,
) -> tuple[plt.Figure, plt.Axes, pd.Series]:
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in dataframe.")

    counts = df[column].value_counts(dropna=dropna)
    if top_n is not None:
        counts = counts.head(top_n)
    counts = counts.sort_values(ascending=not sort_desc)

    if counts.empty:
        raise ValueError(f"No rows to plot for column '{column}'.")

    y_labels = counts.index.astype(str).tolist()
    x_vals = counts.values.tolist()

    if figsize is None:
        figsize = (10, max(4, min(0.45 * len(counts) + 2, 20)))

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(y_labels, x_vals, alpha=0.85)

    if show_values:
        max_x = max(x_vals)
        x_pad = max(1, int(round(0.01 * max_x)))
        for bar, value in zip(bars, x_vals):
            ax.text(
                value + x_pad,
                bar.get_y() + bar.get_height() / 2,
                str(value),
                va="center",
                fontsize=9,
            )

    ax.set_xlabel("Row count")
    ax.set_ylabel(column)
    ax.set_title(f"{column} ({len(counts)})")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()

    return fig, ax, counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-path",
        default="../output",
    )
    parser.add_argument("--run-name", default="run_26_general")
    parser.add_argument(
        "--mode",
        choices=["all", "general", "image-only", "mixed"],
        default="all",
        help="Filter by model mode; all keeps all models.",
    )
    parser.add_argument(
        "--vqa-set",
        default="30K",
        help="VQA set to use (e.g., 10K, 30K, karo_5K).",
    )
    args = parser.parse_args()

    utils_graph.RUN_NAME = args.run_name
    utils_graph_correlation.RUN_NAME = args.run_name

    output_dir = Path("output") / args.run_name / args.vqa_set / "vqa"
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_df = utils.utils_read.build_eval_df(args.base_path, vqa_set=args.vqa_set)

    for mode_label, mode_df in utils.utils_read.select_eval_df(
        eval_df, mode=args.mode
    ):
        cur_output_dir = output_dir / mode_label
        cur_output_dir.mkdir(parents=True, exist_ok=True)
        
        for col in ["model_id", "category", "sub_category", "question_id"]:
            fig, _, _ = plot_row_counts_by_column(
                mode_df,
                col,
                top_n=1000,
            )
            fig.savefig(cur_output_dir / f"hist_{col}.png", dpi=300, bbox_inches="tight")
            plt.close(fig)


if __name__ == "__main__":
    main()
