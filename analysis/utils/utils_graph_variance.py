from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import matplotlib.markers as mmarkers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from utils import (
    utils_graph,
    utils_mapping,
    utils_read
)

warnings.filterwarnings("ignore", message=".*edgecolor.*unfilled marker.*")

def _safe_filename(label: str) -> str:
    return label.replace("/", "_").replace("\\", "_").replace(" ", "_")


def _format_thousands_tick(value: float, _pos: int) -> str:
    if abs(value) >= 1000:
        scaled = value / 1000
        if float(scaled).is_integer():
            return f"{int(scaled)}k"
        return f"{scaled:g}k"
    if float(value).is_integer():
        return f"{int(value)}"
    return f"{value:g}"


def create_variance_curve(
    eval_df: pd.DataFrame,
    *,
    metadata_path: str | Path | None = "utils/metadata.json",
    show: bool = False,
    output_dir: str | Path | None = None,
    filename: str = "vqa_set_variance.png",
    y_pad: float = 0.05,
    y_limit_mode: str = "zero_to_max",
    legend_loc: str = "best",
    group_by: str = "model_family",

    category_column: str = "category",
    category: str = "all",
) -> plt.Figure:
    """
    Plot accuracy (in %) vs num_objects, one curve per category (6 total).
    """
    # Filter category if needed
    if category != "all":
        eval_df = eval_df[eval_df[category_column] == category]

    # Convert accuracy to percentage
    plot_df = eval_df.copy()
    plot_df["accuracy"] = plot_df["accuracy"] * 100
    plot_df = utils_read.macro_accuracy(plot_df, level="model_id", group_by=["vqa_set", "vqa_set_count"])

    # Compute group stats (mean and std) across vqa_set_count
    stats_df = (
        plot_df.groupby(["vqa_set_count", group_by], observed=True)["accuracy"]
        .agg(['mean', 'std'])
        .reset_index()
    )

    model_style, family_map = utils_mapping._build_model_style(
        metadata_path,
        group_by=group_by,
        family_marker_mode="distinct",
    )

    # Prepare plot
    fig, ax = plt.subplots(figsize=(7, 3.5))

    groups = list(stats_df[group_by].unique())
    for i, group in enumerate(groups):
        stats_group = stats_df[stats_df[group_by] == group]
        
        x = stats_group["vqa_set_count"].values
        mean = stats_group["mean"].values
        std = stats_group["std"].values
        # print(f"Group: {group}, x (vqa_set_count): {x}, mean accuracy: {mean}, std: {std}")

        color, marker, size, edge = model_style[group]
        group_label = group

        ax.plot(x, 
                mean, 
                marker=marker, 
                markersize=size, 
                markeredgecolor=edge,
                color=color, 
                linewidth=2, 
                alpha=0.85, 
                label=group_label)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)

    ax.set_xlabel("VQA set", fontsize=12, fontweight="bold")
    if category == "all":
        ylabel = "Overall accuracy (%)"
        ylabel_color = "black"
    else:
        assert eval_df[category_column].nunique() == 1, f"Expected exactly one unique category in eval_df for category='{category}'"
        cat = eval_df[category_column].iloc[0]

        ylabel = utils_mapping.mapping_cat_short.get(cat)
        ylabel_color = utils_mapping.mapping_cat_colors.get(cat)+"CC"
    
    ax.set_ylabel(ylabel.capitalize(), color=ylabel_color)
    # ax.axhline(y=25, color="gray", linestyle="--", linewidth=1)

    # Set y limits
    y_min = stats_df["mean"].min()
    y_max = stats_df["mean"].max()
    if y_limit_mode == "zero_to_max":
        ax.set_ylim(0, y_max + y_pad * max(1e-6, y_max))
    elif y_limit_mode == "fit":
        pad = y_pad * max(1e-6, y_max - y_min)
        ax.set_ylim(y_min - pad, y_max + pad)
    elif y_limit_mode == "fixed":
        ax.set_ylim(15, 55)

    thousands_formatter = FuncFormatter(_format_thousands_tick)
    ax.xaxis.set_major_formatter(thousands_formatter)

    utils_graph.paperformat(ax, grid=["y"], minor=True, figsize=None)

    # Save
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        f_out = out_dir / filename
        fig.savefig(f_out, 
                dpi=300,
                bbox_inches="tight")
        print(f"Plot saved to: {f_out}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig

