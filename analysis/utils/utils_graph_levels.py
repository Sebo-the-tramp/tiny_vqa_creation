from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.markers as mmarkers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils import (
    utils_graph,
    utils_mapping,
    utils_read
)

warnings.filterwarnings("ignore", message=".*edgecolor.*unfilled marker.*")

# /data0/sebastian.cavada/compositional-physics/tiny_vqa_deterministic/output/run_11_general_levels

def _safe_filename(label: str) -> str:
    return label.replace("/", "_").replace("\\", "_").replace(" ", "_")


def create_levels_plot(
    eval_df: pd.DataFrame,
    output_dir: str | Path,
    *,
    accuracy_mode: str = "absolute",  # absolute, baseline_change
    group_by: str = "model_id",
    metadata_path: str | Path | None = "utils/metadata.json",
    levels: list[str] | None = None,
    show: bool = True,
    filename: str = "levels_by_family.png",
) -> plt.Figure:
    
    levels = levels or [
        "baseline",
        "child",
        "teen",
        "undergrad",
        "graduate",
        "expert",
    ]

    if "level" not in eval_df.columns:
        raise KeyError("eval_df must include 'level'.")

    question_col = (
        "question_id_base" if "question_id_base" in eval_df.columns else "question_id"
    )

    plot_df = eval_df.copy()
    plot_df["accuracy"] *= 100

    plot_df = utils_read.macro_accuracy(plot_df, level=group_by, group_by=["level"])

    level_map = {lvl: i for i, lvl in enumerate(levels)}
    plot_df["level_idx"] = plot_df["level"].map(level_map)

    # Compute baseline change
    plot_df["accuracy_change"] = plot_df.apply(
        lambda row: row["accuracy"] - plot_df[
            (plot_df[group_by] == row[group_by]) & (plot_df["level_idx"] == 0)
        ]["accuracy"].values[0],
        axis=1,
    )

    model_style, family_map = utils_mapping._build_model_style(
        metadata_path=metadata_path,
        group_by=group_by,
    )
    
    if accuracy_mode == "absolute":
        accuracy_col = "accuracy"
    elif accuracy_mode == "baseline_change":
        accuracy_col = "accuracy_change"

    show_baseline = accuracy_mode == "absolute"
    if not show_baseline:
        plot_df = plot_df[plot_df["level_idx"] > 0]
    
    # line_styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 3))]
    rng = np.random.default_rng(0)

    output_name = filename

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.violinplot(
        data=plot_df,
        x="level_idx",
        y=accuracy_col,
        ax=ax,
        color="0.90",
        inner=None,
        linewidth=0.5,
        cut=0,
        order=list(range(len(levels))),
        zorder=3
    )

    for i, (group, group_df) in enumerate(plot_df.groupby(group_by)):
        group_df = group_df.sort_values("level_idx")
        if group_df.empty:
            continue
        
        jitter = rng.uniform(-0.15, 0.15, size=group_df["level_idx"].values.size)
        x_jitter = group_df["level_idx"].values + jitter

        y = group_df[accuracy_col].values

        color, marker, _size, edge = model_style[group]
        # ls = line_styles[i % len(line_styles)]
        # ax.plot(
        #     x, y, color=color, linestyle=ls, linewidth=2, alpha=0.85, label=family
        # )
        ax.scatter(
            x_jitter, 
            y, 
            color=color, 
            marker=marker, 
            s=_size**2, 
            edgecolor=edge, 
            linewidth=1, 
            zorder=4
        )
        # if np.isfinite(y_err).any():
        #     ax.fill_between(x, y - y_err, y + y_err, color=color, alpha=0.12)
    
    # n = 10
    # y_min, y_max = ax.get_ylim()
    # for i in range(n):
    #     ax.axhspan(y_min*(i+1)/n, y_min*i/n, facecolor="#ffcccc", alpha=0.5*i/n, zorder=1)
    # for i in range(n):
    #     ax.axhspan(y_max*i/n, y_max*(i+1)/n, facecolor="#ccffcc", alpha=0.5*i/n, zorder=1)
    # ax.set_ylim(y_min, y_max)
    # ax.axhline(0, color="#555555", alpha=1, zorder=3, linewidth=2)

    ax.set_xlabel("")

    # Levels xticks labels
    levels_idx = sorted(plot_df["level_idx"].unique())
    levels_vis = [levels[i] for i in levels_idx]
    ax.set_xticks(list(levels_idx))

    nice_labels = [level.capitalize() for level in levels_vis]
    ax.set_xticklabels(nice_labels, fontsize=11, fontweight="bold", rotation=30)
    ax.tick_params(axis='x', pad=-2)

    fig.tight_layout()

    assert eval_df["category"].nunique() == 1
    cat = eval_df["category"].unique()[0]
    ylabel = utils_mapping.mapping_cat_short[cat]
    ylabel_color = utils_mapping.mapping_cat_colors[cat]+"CC"

    if accuracy_mode == "baseline_change":
        ticks_step = 2
        # ylabel += " (change)"
        ax.axhline(0, color="#000", alpha=0.8, zorder=2, linewidth=1)
    elif accuracy_mode == "absolute":
        ticks_step = 5
        ylabel += " (%)"

    ax.set_ylabel(ylabel, color=ylabel_color)

    utils_graph.paperformat(ax, figsize=(4, 3.5), grid=["y"], ticks_step=ticks_step)
    
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels([f"{'+' if i>0 else ''}{int(i)}" for i in ax.get_yticks()])
    colors = ["black" if y==0 else ("green" if y > 0 else "red") for y in ax.get_yticks()]
    for ticklabel, color in zip(ax.get_yticklabels(), colors):
        ticklabel.set_color(color)

    legend_handles, legend_labels, legend_groups, title_str = utils_graph._build_group_legend_items(
        eval_df,
        group_by=group_by,
        metadata_path=metadata_path
    )
    ax.legend(  legend_handles, 
                legend_labels, 
                title=title_str, 
                bbox_to_anchor=(1.05, 1), 
                loc='upper left', 
                fontsize=8, 
                title_fontsize=9, 
                markerscale=0.9)

    fpath = Path(output_dir) / output_name
    fpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        fpath, 
        dpi=300, 
        bbox_inches="tight", 
        pad_inches=0.05
    )
    print("Plot saved to:", fpath)

    if show:
        plt.show()
    else:
        plt.close(fig)

    # def _plot_baseline_improvement(
    #     plot_df: pd.DataFrame, title_suffix: str, output_name: str
    # ) -> plt.Figure:
    #     q_scores = (
    #         plot_df.groupby(
    #             ["family", question_col, "level", "sub_category"], observed=True
    #         )["accuracy"]
    #         .mean()
    #         .reset_index()
    #         .dropna()
    #     )

    #     final_plot_df = (
    #         q_scores.groupby(["family", "level"], observed=True)["accuracy"]
    #         .agg(mean_accuracy="mean", band_width="sem")
    #         .reset_index()
    #         .dropna()
    #     )

    #     level_map = {lvl: i for i, lvl in enumerate(levels)}
    #     final_plot_df["level_idx"] = final_plot_df["level"].map(level_map)

    #     fig, ax = plt.subplots(figsize=(12, 6))
    #     # plot_df_model = (
    #     #     plot_df.groupby(
    #     #         ["model_id", question_col, "level", "sub_category"], observed=True
    #     #     )["accuracy"]
    #     #     .mean()
    #     #     .reset_index()
    #     #     .dropna()
    #     # )
    #     # final_plot_df_model = (
    #     #     plot_df_model.groupby(["model_id", "level"], observed=True)["accuracy"]
    #     #     .agg(mean_accuracy="mean", band_width="sem")
    #     #     .reset_index()
    #     #     .dropna()
    #     # )
    #     # final_plot_df_model["level_idx"] = final_plot_df_model["level"].map(level_map)
    #     # sns.violinplot(
    #     #     data=final_plot_df_model,
    #     #     x="level_idx",
    #     #     y="mean_accuracy",
    #     #     ax=ax,
    #     #     color="0.90",
    #     #     # inner="box",
    #     #     inner=None,
    #     #     cut=0,
    #     #     width=1.0,
    #     #     linewidth=0.5,
    #     #     order=list(range(len(levels_sorted))),
    #     # )

    #     for i, (family, fam_data) in enumerate(final_plot_df.groupby("family")):
    #         fam_data = fam_data.sort_values("level_idx")
    #         if fam_data.empty:
    #             continue
            
    #         jitter = rng.uniform(-0.15, 0.15, size=fam_data["level_idx"].values[1:].size)
    #         x = fam_data["level_idx"].values[1:] + jitter
    #         y = fam_data["mean_accuracy"].values[1:]
    #         baseline_acc =fam_data[fam_data["level_idx"] == 0]["mean_accuracy"].values
    #         y = y - baseline_acc

    #         y_err = fam_data["band_width"].values
    #         color, marker, _size = model_style.get(family, ("black", "o", 0.5))
    #         ls = line_styles[i % len(line_styles)]
    #         # ax.plot(
    #         #     x, y, color=color, linestyle=ls, linewidth=2, alpha=0.85, label=family
    #         # )
    #         ax.scatter(
    #             x, y, color=color, marker=marker, s=_size**2, edgecolor="white", linewidth=1, zorder=4, label=family
    #         )
    #         # if np.isfinite(y_err).any():
    #         #     ax.fill_between(x, y - y_err, y + y_err, color=color, alpha=0.12)
        
    #     n = 10
    #     y_min, y_max = ax.get_ylim()
    #     for i in range(n):
    #         ax.axhspan(y_min*(i+1)/n, y_min*i/n, facecolor="#ffcccc", alpha=0.5*i/n, zorder=1)
    #     for i in range(n):
    #         ax.axhspan(y_max*i/n, y_max*(i+1)/n, facecolor="#ccffcc", alpha=0.5*i/n, zorder=1)
    #     ax.set_ylim(y_min, y_max)
    #     ax.axhline(0, color="#555555", alpha=1, zorder=3, linewidth=2)

    #     ax.set_xticks(list(range(len(levels)))[1:])
    #     nice_labels = [level.capitalize() for level in levels]
    #     ax.set_xticklabels(nice_labels[1:], fontsize=11, fontweight="bold", rotation=30)
    #     ax.tick_params(axis='x', pad=-2)
    #     # ax.set_xlabel("Difficulty Level", fontsize=12)
    #     ax.set_ylabel("Change in accuracy", fontsize=12)
    #     ax.set_title(f"Performance by Family{title_suffix}", fontsize=14)
    #     ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    #     ax.grid(True, linestyle="--", alpha=0.5)

    #     fig.tight_layout()
    #     utils_graph.paperformat(ax, figsize=(4.5, 3), grid=["y"], minor=False)

    #     yticks = np.arange(-2, 10, 2)
    #     ax.set_yticks(yticks)
    #     ax.set_yticklabels([f"{'+' if i>0 else ''}{int(i)}" for i in yticks])
    #     colors = ["black" if y==0 else ("green" if y > 0 else "red") for y in yticks]
    #     for ticklabel, color in zip(ax.get_yticklabels(), colors):
    #         ticklabel.set_color(color)

    #     run = run_name or globals().get("RUN_NAME", "default")
    #     out_dir = Path(output_dir) if output_dir is not None else Path("output") / run
    #     out_dir.mkdir(parents=True, exist_ok=True)
    #     fig.savefig(out_dir / output_name, dpi=300, bbox_inches="tight", pad_inches=0.05)

    #     if show:
    #         plt.show()
    #     else:
    #         plt.close(fig)

    #     return fig
    

    # return _plot_subset(eval_df, "", filename)
