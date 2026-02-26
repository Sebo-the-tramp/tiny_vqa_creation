from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import textwrap

from utils.utils_read import load_results, _sanitize_answer
import utils.utils_graph as utils_graph
import utils.utils_graph_correlation as utils_graph_correlation
from utils.utils_graph_correlation import _build_model_style
from utils.utils_mapping import subcategories, categories


def add_model_mode(
    eval_df: pd.DataFrame, metadata_path: str | Path = "utils/metadata.json"
) -> pd.DataFrame:
    path = Path(metadata_path)
    if not path.exists():
        eval_df["model_mode"] = pd.NA
        return eval_df

    metadata_df = pd.read_json(path)
    if "id" in metadata_df.columns and "model_id" not in metadata_df.columns:
        metadata_df = metadata_df.rename(columns={"id": "model_id"})
    mode_map = (
        metadata_df.dropna(subset=["model_id"])
        .set_index("model_id")["mode"]
        .to_dict()
    )
    eval_df["model_mode"] = eval_df["model_id"].map(mode_map).fillna("unknown")
    return eval_df


def build_eval_df(base_path: str | Path) -> pd.DataFrame:
    base = Path(base_path)

    run_folder = Path(utils_graph.RUN_NAME)

    df = load_results(
        base,
        run_folder=run_folder,
        merge_model_answers=True,
        model_answers_wide=True,
        cache=True,
        add_sim_metadata=True,
    )

    results_dir = base / run_folder / f"results_{run_folder}"
    model_cols = sorted(
        p.stem.replace("_val", "") for p in results_dir.glob("*_val.json")
    )
    model_cols = [c for c in model_cols if c in df.columns]
    if not model_cols:
        raise ValueError(f"No model answer columns found in {results_dir}")

    df["answer"] = df["answer"].apply(
        lambda a: _sanitize_answer(a, max_prefix_chars=None)
    )

    id_cols = [
        c
        for c in [
            "idx",
            "question_id",
            "category",
            "sub_category",
            "num_objects",
            "object_count",
            "answer",
            "mode_test",
            "mode_val",
            "mode",
        ]
        if c in df.columns
    ]

    eval_df = df.melt(
        id_vars=id_cols,
        value_vars=model_cols,
        var_name="model_id",
        value_name="model_answer",
    )

    valid = eval_df["model_answer"].notna() & eval_df["answer"].notna()
    eval_df["is_correct"] = pd.NA
    eval_df.loc[valid, "is_correct"] = (
        eval_df.loc[valid, "model_answer"] == eval_df.loc[valid, "answer"]
    )

    if "mode_val" in eval_df.columns:
        eval_df["mode_y"] = eval_df["mode_val"]
    elif "mode_test" in eval_df.columns:
        eval_df["mode_y"] = eval_df["mode_test"]
    elif "mode" in eval_df.columns:
        eval_df["mode_y"] = eval_df["mode"]

    return eval_df


def macro_avg_by_question(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if "question_id" not in df.columns:
        return df.groupby(group_cols, observed=True)["accuracy"].mean().reset_index()
    q_acc = (
        df.groupby(group_cols + ["question_id"], observed=True)["accuracy"]
        .mean()
        .reset_index()
    )
    return q_acc.groupby(group_cols, observed=True)["accuracy"].mean().reset_index()


def normalize_category(
    eval_df: pd.DataFrame, top_category: str
) -> tuple[pd.DataFrame, str]:
    if "category" not in eval_df.columns:
        raise KeyError("eval_df must include 'category'.")
    categories = pd.unique(eval_df["category"].dropna())
    if top_category in categories:
        return eval_df[eval_df["category"] == top_category], top_category
    lower_map = {str(c).lower(): c for c in categories}
    fallback = lower_map.get(top_category.lower())
    if fallback is not None:
        return eval_df[eval_df["category"] == fallback], str(fallback)
    raise ValueError(
        "Top category not found. Available categories: "
        f"{sorted(str(c) for c in categories)}"
    )


def plot_subcategory_violin(
    eval_df: pd.DataFrame,
    *,
    top_category: str,
    output_path: Path,
    metadata_path: str | Path = "utils/metadata.json",
    group_by: str = "family",
    exclude_categories: list[str] | None = None,
    seed: int = 0,
    title: str | None = None,
) -> None:
    if (
        "sub_category" not in eval_df.columns
        or "model_id" not in eval_df.columns
        or "category" not in eval_df.columns
    ):
        raise KeyError(
            "eval_df must include 'category', 'sub_category', and 'model_id'."
        )

    plot_df = eval_df.copy()
    sub_label_map = subcategories.copy()
    plot_df["accuracy"] = pd.to_numeric(plot_df["is_correct"], errors="coerce")
    plot_df = plot_df.dropna(subset=["accuracy", "sub_category", "model_id"])

    material_category = normalize_category(eval_df, top_category)[1]
    material_subcats = [
        "density",
        "mass",
        "material_identification",
        "poisson_ratio",
        "young_modulus",
    ]
    exclude_set = {c.strip() for c in (exclude_categories or []) if c.strip()}
    plot_df = plot_df[~plot_df["category"].isin(exclude_set)]

    plot_df["group_label"] = np.where(
        (plot_df["category"] == material_category)
        & (plot_df["sub_category"].isin(material_subcats)),
        plot_df["sub_category"].astype(str),
        plot_df["category"].astype(str),
    )

    model_style, family_map = _build_model_style(
        plot_df,
        metadata_path,
        group_by="family" if group_by == "family" else "model_id",
    )

    agg_df = macro_avg_by_question(plot_df, ["model_id", "group_label"])
    if agg_df.empty:
        raise ValueError("No aggregated rows available after filtering.")

    agg_df["family"] = agg_df["model_id"].map(family_map).fillna("Other")
    if group_by == "family":
        agg_df = (
            agg_df.groupby(["family", "group_label"], observed=True)["accuracy"]
            .mean()
            .reset_index()
        )
    group_values = set(agg_df["group_label"])
    top_order = sorted(
        cat
        for cat in pd.unique(plot_df["category"])
        if cat != material_category and cat in group_values
    )
    material_order = [sub for sub in material_subcats if sub in group_values]
    group_order = top_order + material_order
    if not group_order:
        group_order = sorted(pd.unique(agg_df["group_label"]))

    sns.set_style("white")
    label_fontsize = 18
    tick_fontsize = 15
    fig_width = 7.0 * 0.84 * 1.05 * 0.95
    fig_height = max(4.0, 0.55 * len(group_order) + 1.2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    sns.violinplot(
        data=agg_df,
        y="group_label",
        x="accuracy",
        order=group_order,
        color="0.85",
        inner=None,
        cut=0,
        linewidth=1.0,
        ax=ax,
    )

    rng = np.random.default_rng(seed)
    y_map = {cat: i for i, cat in enumerate(group_order)}
    for _, row in agg_df.iterrows():
        y_pos = y_map.get(row["group_label"])
        if y_pos is None:
            continue
        jitter = rng.uniform(-0.2, 0.2)
        if group_by == "family":
            style_key = str(row["family"])
        else:
            style_key = str(row["model_id"])
        color, marker, size = model_style[style_key]
        ax.scatter(
            row["accuracy"],
            y_pos + jitter,
            color=color,
            s=size**2,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.7,
            marker=marker,
        )

    ax.axvline(0.25, color="#d62728", linestyle="--", linewidth=1.2, zorder=-1)
    ax.axvline(0.78, color="#8ecae6", linestyle="--", linewidth=1.2, zorder=-1)
    ax.set_xlabel("")
    ax.set_ylabel("")
    y_min = -1.0
    y_max = len(group_order) - 0.5
    ax.set_xlim(0.0, 1.05)
    ax.set_xticks([0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis="x", direction="in", pad=-18)
    ax.set_ylim(y_min, y_max)
    ax.tick_params(axis="both", labelsize=tick_fontsize, colors="black")
    for spine in ["left", "bottom", "right", "top"]:
        ax.spines[spine].set_color("black")
        ax.spines[spine].set_linewidth(1.0)
        ax.spines[spine].set_zorder(5)
        ax.spines[spine].set_visible(True)
    ax.set_yticks(range(len(group_order)))
    ax.set_yticklabels([""] * len(group_order))
    max_by_group = (
        agg_df.groupby("group_label", observed=True)["accuracy"].max().to_dict()
    )
    y_offset = 0.0
    for idx, cat in enumerate(group_order):
        raw_label = categories.get(str(cat), None)
        if raw_label is None:
            raw_label = sub_label_map.get(str(cat), str(cat))
        display_label = raw_label
        x_pos = max_by_group.get(cat, 0.0)
        x_pos = min(x_pos + 0.03, 1.03)
        ax.text(
            x_pos,
            idx + y_offset,
            display_label,
            va="bottom",
            ha="left",
            fontsize=tick_fontsize,
            color="black",
        )
    ax.grid(False)
    ax.text(
        0.11,
        y_max - 0.35,
        "Random",
        ha="left",
        va="bottom",
        fontsize=tick_fontsize - 2,
        color="#d62728",
    )
    ax.text(
        0.77,
        y_max - 0.75,
        "Common sense\nmean accuracy",
        ha="left",
        va="bottom",
        fontsize=tick_fontsize - 2,
        color="#3d7ea6",
    )
    if top_order and material_order:
        top_end = len(top_order) - 0.5
        ax.axhspan(y_min, top_end, color="#e9f1ff", zorder=-2)
        ax.axhspan(top_end, y_max, color="#efe6ff", zorder=-2)
        label_x = 0.07
        ax.text(
            label_x,
            (top_end - 0.5) / 2,
            "High-Level Physics",
            rotation=90,
            va="center",
            ha="center",
            fontsize=tick_fontsize,
            color="black",
            transform=ax.get_yaxis_transform(),
        )
        ax.text(
            label_x,
            (top_end + len(group_order) - 0.5) / 2,
            "Low-Level Physics",
            rotation=90,
            va="center",
            ha="center",
            fontsize=tick_fontsize,
            color="black",
            transform=ax.get_yaxis_transform(),
        )
    if title:
        ax.set_title(title)
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-path",
        default="/data0/sebastian.cavada/compositional-physics/tiny_vqa_creation/output/",
    )
    parser.add_argument("--run-name", default="run_23_general_obj_num")
    parser.add_argument(
        "--mode",
        choices=["mixed", "general", "image-only"],
        default="mixed",
        help="Filter by model mode; mixed keeps all models.",
    )
    parser.add_argument(
        "--top-category",
        default="material_understandgin",
        help="Top-level category to plot.",
    )
    parser.add_argument(
        "--group-by",
        choices=["family", "model_id"],
        default="family",
        help="Color/marker grouping for scatter points.",
    )
    parser.add_argument(
        "--exclude-top-categories",
        default="",
        help="Comma-separated list of top categories to exclude (e.g., temporal).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--title",
        default=None,
        help="Optional plot title; omit for no title.",
    )
    args = parser.parse_args()

    utils_graph.RUN_NAME = args.run_name
    utils_graph_correlation.RUN_NAME = args.run_name

    output_dir = Path("output") / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_df = build_eval_df(args.base_path)
    eval_df = add_model_mode(eval_df)
    if args.mode != "mixed":
        eval_df = eval_df[eval_df["model_mode"] == args.mode]

    safe_category = str(args.top_category).replace("/", "_").replace(" ", "_")
    output_path = output_dir / f"{safe_category}_subcategory_violin_{args.mode}.png"

    plot_subcategory_violin(
        eval_df,
        top_category=args.top_category,
        output_path=output_path,
        group_by=args.group_by,
        exclude_categories=[
            c for c in args.exclude_top_categories.split(",") if c.strip()
        ],
        seed=args.seed,
        title=args.title,
    )


if __name__ == "__main__":
    main()
