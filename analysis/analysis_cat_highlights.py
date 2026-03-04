from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from utils import (
    utils_read,
    utils_graph,
    utils_mapping,
    utils_graph_correlation
)

def plot_highlight_violin(
    eval_df: pd.DataFrame,
    *,
    top_category: str,
    output_path: Path,
    metadata_path: str | Path = "utils/metadata.json",
    group_by: str = "model_id",
    exclude_categories: list[str] | None = None,
    seed: int = 0,
    title: str | None = None,
) -> None:
    plot_df = eval_df.copy()
    plot_df["accuracy"] *= 100

    highlight_categories = ["material_understanding", "mechanics"]
    exclude_set = exclude_categories
    plot_df = plot_df[~plot_df["category"].isin(exclude_set)]

    plot_subcat_df = utils_read.macro_accuracy(plot_df, level="sub_category", group_by=group_by)
    plot_cat_df = utils_read.macro_accuracy(plot_df, level="category", group_by=group_by)

    cat_to_subcats_visible = {cat: [sub for sub, c in utils_mapping.subcat_to_cat.items() if c == cat and sub in plot_df["sub_category"].unique().tolist()] for cat in highlight_categories}

    subcat_order = []
    for key in highlight_categories:
        subcat_order = subcat_order + cat_to_subcats_visible[key]

    cat_order = utils_mapping.sort_categories()
    cat_order = [cat for cat in cat_order if cat in plot_df["category"].unique().tolist() and cat not in highlight_categories and cat in plot_cat_df["category"].unique().tolist()]

    group_keys = subcat_order + cat_order
    group_types = ["sub_category"] * len(subcat_order) + ["category"] * len(cat_order)
    group_labels = [utils_mapping.subcategories[sub] for sub in subcat_order] + [utils_mapping.categories[cat] for cat in cat_order]

    sns.set_style("white")
    label_fontsize = 14
    tick_fontsize = 12
    fig_width = 7.0 * 0.84 * 1.05 * 0.85
    fig_height = max(4.0, 0.55 * len(group_keys) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    groups = []
    groups.append(plot_subcat_df[plot_subcat_df["sub_category"].isin(subcat_order)])
    groups.append(plot_subcat_df[plot_subcat_df["category"].isin(cat_order)])
    
    groups_map = {cat: len(group_keys) - 1 - i for i, cat in enumerate(group_keys)}
    
    assert len(groups_map) == len(group_keys)

    groups_df = pd.concat(groups, ignore_index=True)
    groups_df["group_label"] = groups_df.apply(
        lambda row: utils_mapping.subcategories[row["sub_category"]] if row["sub_category"] in subcat_order else utils_mapping.categories[row["category"]],
        axis=1,
    ).astype(str)
    groups_df["group_idx"] = groups_df.apply(
        lambda row: groups_map[row["sub_category" if (row["sub_category"] in subcat_order) else "category"]],
        axis=1,
    )

    sns.violinplot(
        data=groups_df,
        y="group_label",
        x="accuracy",
        order=group_labels[::-1],
        color="0.85",
        inner=None,
        cut=0,
        linewidth=1.0,
        ax=ax,
        zorder=2
    )

    model_style, family_map = utils_mapping._build_model_style(
        metadata_path,
        group_by=group_by
    )

    rng = np.random.default_rng(seed)
    for group_name, df_m in groups_df.groupby(group_by):
        y_pos = df_m["group_idx"]
        if y_pos is None:
            continue
        
        jitter = rng.uniform(-0.2, 0.2)
        color, marker, size, edge = model_style[group_name]
        ax.scatter(
            df_m["accuracy"],
            y_pos + jitter,
            color=color,
            s=size**2,
            alpha=0.85,
            edgecolor=edge,
            linewidth=0.7,
            marker=marker,
            zorder=3,
        )

    ax.axvline(25, color="#d62728", linestyle="--", linewidth=1.2, zorder=-1)
    ax.set_xlabel("")
    ax.set_ylabel("")
    y_min = -1.0
    y_max = len(group_keys) - 0.5
    ax.set_xlim(0.0, 100.0)
    ax.set_xticks([4, 25, 50, 75, 94])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], ha="center")
    ax.tick_params(axis="x", direction="in", pad=-13, length=5, width=1.0, colors="black")
    ax.set_ylim(y_min, y_max)
    ax.tick_params(axis="both", labelsize=tick_fontsize, colors="black")
    plt.setp(ax.get_xticklabels(), fontweight="bold")
    plt.setp(ax.get_yticklabels(), fontweight="bold")
    for spine in ["left", "bottom", "right", "top"]:
        ax.spines[spine].set_color("black")
        ax.spines[spine].set_linewidth(1.0)
        ax.spines[spine].set_zorder(5)
        ax.spines[spine].set_visible(True)
    ax.set_yticks(range(len(group_keys)))
    ax.set_yticklabels([""] * len(group_keys))

    max_by_group = (
        groups_df.groupby("group_label", observed=True)["accuracy"].max().to_dict()
    )
    y_offset = -0.3
    for idx, (key, label, g_type) in enumerate(zip(group_keys, group_labels, group_types)):
        display_label = label
        x_pos = max_by_group[label]
        x_pos = min(x_pos + 3, 103)
        
        label_color = utils_mapping.get_cat_color(key, g_type)
        ax.text(
            x_pos,
            len(group_keys) - 1 - idx + y_offset,
            display_label.replace(" ", "\n"),
            va="bottom",
            ha="left",
            fontsize=tick_fontsize,
            color=label_color,
            fontweight="bold",
        )
    ax.grid(False)
    ax.text(
        20,
        y_max - 1.49,
        "Random",
        ha="left",
        va="bottom",
        fontsize=tick_fontsize,
        color="#d62728",
        rotation=90,
    )
    top_end = len(group_keys) - 0.5
    label_x = 0.04
    for cat in highlight_categories:
        label = utils_mapping.categories[cat]
        if cat == "material_understanding":
            sub_label = "\nLow-level physics"
        elif cat == "mechanics":
            sub_label = "\nHigh-level physics"
        
        color = utils_mapping.get_cat_color(cat, "category")
        
        bottom = top_end - len(cat_to_subcats_visible[cat])
        ax.text(
            label_x,
            (top_end + bottom) / 2,
            label,
            rotation=90,
            va="center",
            ha="center",
            fontsize=label_fontsize,
            color=color,
            transform=ax.get_yaxis_transform(),
            fontweight="bold",
        )

        ax.text(
            label_x+0.03,
            (top_end + bottom) / 2,
            sub_label,
            rotation=90,
            va="center",
            ha="center",
            fontsize=label_fontsize-4,
            color="gray",
            transform=ax.get_yaxis_transform(),
        )
        ax.axhspan(bottom, top_end, color=color, alpha=0.08, zorder=-2)
        top_end = bottom

    if title:
        ax.set_title(title)
    fig.subplots_adjust(bottom=0.02)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, 
                dpi=300, 
                bbox_inches="tight", 
                pad_inches=0.02)
    print("Saved plot to:", output_path)
    plt.close(fig)

def plot_commonsense_violin(
    eval_df: pd.DataFrame,
    *,
    output_path: Path,
    metadata_path: str | Path = "utils/metadata.json",
    seed: int = 0,
    group_by: str = "model_id",
    title: str | None = None,
) -> None:
    plot_df = eval_df.copy()

    assert group_by == "model_id"

    models_df = plot_df[[group_by]].drop_duplicates().reset_index(drop=True)

    benchmark_field = "Avg. Score"
    model_cs_mapping, cs_accuracy = utils_graph._build_commonsense_mapping(eval_df, benchmark_field)

    models_df["cs_mapping"] = models_df["model_id"].map(utils_graph._standardize_model_label)
    models_df["cs_accuracy"] = None
    for model_id in models_df["model_id"].unique():
        if model_id not in cs_accuracy:
            continue
        models_df.loc[models_df["model_id"] == model_id, "cs_accuracy"] = cs_accuracy[model_id]

    models_df = models_df[models_df["cs_accuracy"].notnull()]

    group_keys = ["commonsense"]
    group_labels = ["Common Sense"]
    group_types = ["cs"]

    models_df["group_label"] = group_labels[0]
    models_df["group_idx"] = 0

    sns.set_style("white")
    label_fontsize = 14
    tick_fontsize = 12
    fig_width = 7.0 * 0.84 * 1.05 * 0.85
    fig_height = 1.4
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    sns.violinplot(
        data=models_df,
        y="group_label",
        x="cs_accuracy",
        order=group_labels,
        color="0.85",
        inner=None,
        cut=0,
        linewidth=1.0,
        ax=ax,
        zorder=2,
        width=0.7
    )

    model_style, family_map = utils_mapping._build_model_style(
        metadata_path,
        group_by=group_by
    )

    rng = np.random.default_rng(seed)
    for group_name, df_m in models_df.groupby(group_by):
        y_pos = df_m["group_idx"]
        if y_pos is None:
            continue
        
        jitter = rng.uniform(-0.2, 0.2)
        color, marker, size, edge = model_style[group_name]
        ax.scatter(
            df_m["cs_accuracy"],
            y_pos + jitter,
            color=color,
            s=size**2,
            alpha=0.85,
            edgecolor=edge,
            linewidth=0.7,
            marker=marker,
            zorder=3,
        )

    ax.axvline(25, color="#d62728", linestyle="--", linewidth=1.2, zorder=-1)
    ax.set_xlabel("")
    ax.set_ylabel("")
    y_min = -0.5
    y_max = 1 - 0.5
    ax.set_xlim(0.0, 100.0)
    # ax.set_xticks([4, 25, 50, 75, 94])
    # ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], ha="center")
    ax.set_xticks([])
    ax.tick_params(axis="x", direction="in", pad=-13, length=5, width=1.0, colors="black")
    ax.set_ylim(y_min, y_max)
    ax.tick_params(axis="both", labelsize=tick_fontsize, colors="black")
    plt.setp(ax.get_xticklabels(), fontweight="bold")
    plt.setp(ax.get_yticklabels(), fontweight="bold")
    for spine in ["left", "bottom", "right", "top"]:
        ax.spines[spine].set_color("black")
        ax.spines[spine].set_linewidth(1.0)
        ax.spines[spine].set_zorder(5)
        ax.spines[spine].set_visible(True)
    ax.set_yticks(range(1))
    ax.set_yticklabels([""] * 1)

    # max_by_group = (
    #     models_df.groupby("group_label", observed=True)["cs_accuracy"].max().to_dict()
    # )
    # y_offset = -0.3
    # for idx, (key, label, g_type) in enumerate(zip(group_keys, group_labels, group_types)):
    #     display_label = label
    #     x_pos = max_by_group[label]
    #     x_pos = min(x_pos + 3, 103)
        
    #     label_color = utils_mapping.get_cat_color(key, g_type)
    #     ax.text(
    #         x_pos,
    #         len(group_keys) - 1 - idx + y_offset,
    #         display_label.replace(" ", "\n"),
    #         va="bottom",
    #         ha="left",
    #         fontsize=tick_fontsize,
    #         color=label_color,
    #         fontweight="bold",
    #     )
    ax.grid(False)
    # ax.text(
    #     20,
    #     y_max - 1.49,
    #     "Random",
    #     ha="left",
    #     va="bottom",
    #     fontsize=tick_fontsize,
    #     color="#d62728",
    #     rotation=90,
    # )
    top_end = len(group_keys) - 0.5
    label_x = 0.06
    for cat in group_keys:
        label = "Common\nSense"
        sub_label = "\n(8 benchmarks)"
        
        color = "#1F1BEE"
        
        bottom = top_end - 1
        ax.text(
            label_x,
            (top_end + bottom) / 2,
            label,
            rotation=90,
            va="center",
            ha="center",
            fontsize=label_fontsize,
            color=color,
            transform=ax.get_yaxis_transform(),
            fontweight="bold",
        )

        ax.text(
            label_x+0.05,
            (top_end + bottom) / 2,
            sub_label,
            rotation=90,
            va="center",
            ha="center",
            fontsize=label_fontsize-4,
            color="gray",
            transform=ax.get_yaxis_transform(),
        )
        ax.axhspan(bottom, top_end, color=color, alpha=0.08, zorder=-2)
        top_end = bottom

    if title:
        ax.set_title(title)
    fig.subplots_adjust(bottom=0.02)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, 
                dpi=300, 
                bbox_inches="tight", 
                pad_inches=0.02)
    print("Saved plot to:", output_path)
    plt.close(fig)



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-path",
        default="../output/",
    )
    parser.add_argument("--run-name", default="run_24_general_yms_variations")
    parser.add_argument(
        "--mode",
        choices=["all", "general", "image-only", "mixed"],
        default="all",
        help="Filter by model mode; all keeps all models.",
    )
    parser.add_argument(
        "--vqa-set",
        default="10K",
        help="VQA set to use (e.g., 10K, 30K, karo_5K).",
    )
    args = parser.parse_args()
    parser.add_argument(
        "--top-category",
        default="material_understanding",
        help="Top-level category to plot.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path("output") / args.run_name / args.vqa_set / "cat_highlights"

    eval_df = utils_read.build_eval_df(args.run_name, args.base_path, vqa_set=args.vqa_set)

    for mode_label, mode_df in utils_read.select_eval_df(
        eval_df, mode=args.mode
    ):
        cur_output_dir = output_dir / mode_label

        safe_category = str(args.top_category).replace("/", "_").replace(" ", "_")
        output_path = cur_output_dir 

        plot_highlight_violin(
            mode_df,
            top_category=args.top_category,
            output_path=output_path / f"highlight_model.png",
            group_by="model_id",
            exclude_categories=[c for c in mode_df["category"].unique().tolist() if c not in ["material_understanding", "mechanics"]],
            seed=args.seed,
            # title=args.title,
        )
        plot_commonsense_violin(
            mode_df,
            output_path=output_path / f"commonsense_model.png",
            group_by="model_id",
            seed=args.seed,
            # title=args.title,
        )


if __name__ == "__main__":
    main()
