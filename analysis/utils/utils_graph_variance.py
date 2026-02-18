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
import utils.utils_mapping

warnings.filterwarnings("ignore", message=".*edgecolor.*unfilled marker.*")

def paperformat(ax, figsize=(4, 3.1), ylim=None, ticks_step=10, grid=["x", "y"], minor=True):
    fig = ax.get_figure()
    if figsize is not None:
        fig.set_size_inches(*figsize)

    ax.set_title("")
    for label in ax.get_xticklabels():
        label.set_fontsize(13)
        label.set_ha('center')
        label.set_fontweight('bold')  # or 'normal', 'light', etc.
    
    for label in ax.get_yticklabels():
        label.set_fontsize(13)
        label.set_fontweight('bold')  # or 'normal', 'light', etc.

    for label in [ax.xaxis.label, ax.yaxis.label]:
        label.set_fontsize(14)
        label.set_fontweight("bold")

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if ylim is not None:
        ax.set_ylim(ylim)
    
    import matplotlib.ticker as mticker
    ax.yaxis.set_major_locator(mticker.MultipleLocator(ticks_step))
    if minor:
        ax.yaxis.set_minor_locator(mticker.MultipleLocator(ticks_step//2))

    ax.grid(False)
    if grid:
        for axis in grid:
            ax.grid(axis=axis, which="major", linestyle="-", alpha=0.5)
            ax.grid(axis=axis, which="minor", linestyle="-", alpha=0.1)
        

def _safe_filename(label: str) -> str:
    return label.replace("/", "_").replace("\\", "_").replace(" ", "_")


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
    if category_column not in eval_df.columns or "model_id" not in eval_df.columns:
        raise KeyError(f"eval_df must include '{category_column}' and 'model_id' columns.")
    
    # Filter category if needed
    if category != "all":
        eval_df = eval_df[eval_df[category_column] == category]

    # Compute cat mean accuracy
    cat_acc_df = (
        eval_df.groupby(["vqa_set_count", "vqa_set", "category", "model_family", "model_id"], observed=True)["accuracy"]
        .mean()
        .reset_index()
    )

    # Compute model mean accuracy
    model_acc_df = (
        cat_acc_df.groupby(["vqa_set_count", "vqa_set", "model_family", "model_id"], observed=True)["accuracy"]
        .mean()
        .reset_index()
    )

    # Compute group stats (mean and std) across vqa_set_count
    stats_df = (
        model_acc_df.groupby(["vqa_set_count", group_by], observed=True)["accuracy"]
        .agg(['mean', 'std'])
        .reset_index()
    )

    stats_df["mean"] *= 100
    stats_df["std"] *= 100

    model_style, family_map = utils.utils_mapping._build_model_style(
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

        color, marker, size = model_style[group]
        group_label = group

        ax.plot(x, 
                mean, 
                marker=marker, 
                markersize=size, 
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

        ylabel = utils.utils_mapping.mapping_cat_short.get(cat)
        ylabel_color = utils.utils_mapping.mapping_cat_colors.get(cat)+"CC"

    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold", color=ylabel_color)
    
    # legend = ax.legend(  loc=legend_loc, 
    #             fontsize=9.5, 
    #             ncol=2, 
    #             markerscale=0.5,
    #             handletextpad=0.2,
    #             columnspacing=0.5,
    #             borderpad=0.2,
    #             frameon=True,
    #             handlelength=1.3 )
    # for text, handle in zip(legend.get_texts(), legend.legend_handles):
    #     if hasattr(handle, "get_color"):
    #         text.set_color(handle.get_color())
    #         text.set_fontweight("bold")
    #     elif hasattr(handle, "get_facecolor"):
    #         text.set_color(handle.get_facecolor()[0])
    ax.grid(axis="y", alpha=0.3)

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

    paperformat(ax, grid=["y"], minor=True, figsize=None)

    # ax.set_xticks(range(1, 11))
    # ax.set_xticklabels([str(v) for v in range(1, 11)])


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

