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

warnings.filterwarnings("ignore", message=".*edgecolor.*unfilled marker.*")


_DEFAULT_MARKERS = [
    "o",
    "s",
    "^",
    "v",
    "<",
    ">",
    "p",
    "*",
    "h",
    "H",
    "D",
    "d",
    ".",
    "1",
    "2",
    "3",
    "4",
    "8",
    "P",
    "X",
]


def _safe_filename(label: str) -> str:
    return label.replace("/", "_").replace("\\", "_").replace(" ", "_")


def _macro_avg_by_question(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if "question_id" not in df.columns:
        return df.groupby(group_cols, observed=True)["accuracy"].mean().reset_index()
    q_acc = (
        df.groupby(group_cols + ["question_id"], observed=True)["accuracy"]
        .mean()
        .reset_index()
    )
    return q_acc.groupby(group_cols, observed=True)["accuracy"].mean().reset_index()


def _load_model_metadata(metadata_path: str | Path) -> pd.DataFrame:
    path = Path(metadata_path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    if "id" in df.columns:
        df = df.rename(columns={"id": "model_id"})
    return df


def _build_model_style(
    eval_df: pd.DataFrame,
    metadata_path: str | Path | None,
    *,
    group_by: str = "model_id",
) -> tuple[dict[str, tuple[str, str, float]], dict[str, str]]:
    group_ids = []
    palette = []

    metadata_df = None
    if metadata_path is not None and Path(metadata_path).exists():
        metadata_df = _load_model_metadata(metadata_path)

    families = []
    params = []
    family_map = {}
    for model_id in pd.unique(eval_df["model_id"]):
        if metadata_df is not None and model_id in set(metadata_df["model_id"]):
            row = metadata_df[metadata_df["model_id"] == model_id].iloc[0]
            family = str(row.get("family", "Other"))
            params_b = pd.to_numeric(row.get("params_b", np.nan), errors="coerce")
        else:
            family = "Other"
            params_b = np.nan
        family_map[str(model_id)] = family
        if group_by == "model_id":
            families.append(family)
            params.append(params_b)

    if group_by == "family":
        group_ids = pd.unique(pd.Series(list(family_map.values())))
        families = list(group_ids)
        params = [np.nan] * len(group_ids)
    else:
        group_ids = pd.unique(eval_df["model_id"])

    unique_families = list(dict.fromkeys(families)) if families else ["Other"]
    palette = sns.color_palette("tab20", len(group_ids))
    markers = list(_DEFAULT_MARKERS)
    for marker in list(mmarkers.MarkerStyle.markers.keys()):
        if marker not in markers:
            markers.append(marker)

    family_markers = {fam: markers[i % len(markers)] for i, fam in enumerate(unique_families)}

    params = np.array(params, dtype=float)
    valid_params = params[~np.isnan(params)]
    fallback = float(np.nanmedian(valid_params)) if valid_params.size else 5.0
    params = np.where(np.isnan(params), fallback, params)
    params = np.clip(params, 2.0, 15.0)
    sizes = (params - 2.0) / (15.0 - 2.0)

    model_style = {}
    for group_id, color, fam, size in zip(group_ids, palette, families, sizes):
        model_style[str(group_id)] = (color, family_markers.get(fam, "o"), float(size))

    return model_style, family_map


def create_num_objects_violin_grid(
    eval_df: pd.DataFrame,
    *,
    metadata_path: str | Path | None = "utils/metadata.json",
    num_objects_col: str | None = None,
    count_model_substring: str | None = None,
    subcategory_label_map: dict[str, str] | None = None,
    n_cols: int = 4,
    seed: int = 0,
    show: bool = False,
    output_dir: str | Path | None = None,
    run_name: str | None = None,
    filename: str = "num_objects_violin_grid.png",
    group_by: str = "model_id",
    save_per_category: bool = False,
    per_category_dirname: str | None = None,
    save_legend: bool = False,
    legend_filename: str | None = None,
    legend_cols: int = 4,
    legend_figsize: tuple[float, float] | None = None,
    y_limit_mode: str = "zero_to_max",
    y_pad: float = 0.0,
    n_label_offset: float = 0.08,
    split_values: tuple[float, float] | None = (3, 4),
    split_line_kwargs: dict | None = None,
) -> plt.Figure:
    if "sub_category" not in eval_df.columns or "model_id" not in eval_df.columns:
        raise KeyError("eval_df must include 'sub_category' and 'model_id' columns.")

    if num_objects_col is None:
        if "num_objects" in eval_df.columns:
            num_objects_col = "num_objects"
        elif "object_count" in eval_df.columns:
            num_objects_col = "object_count"
        else:
            raise KeyError("eval_df must include 'num_objects' or 'object_count'.")

    plot_df = eval_df.copy()
    plot_df["num_objects"] = pd.to_numeric(plot_df[num_objects_col], errors="coerce")

    if "accuracy" in plot_df.columns:
        plot_df["accuracy"] = pd.to_numeric(plot_df["accuracy"], errors="coerce")
    elif "is_correct" in plot_df.columns:
        plot_df["accuracy"] = pd.to_numeric(plot_df["is_correct"], errors="coerce")
    else:
        raise KeyError("eval_df must include 'accuracy' or 'is_correct'.")

    plot_df = plot_df.dropna(subset=["num_objects", "accuracy"])

    if count_model_substring:
        counts_source = plot_df[
            plot_df["model_id"].astype(str).str.contains(count_model_substring, na=False)
        ]
    else:
        dedup_cols = [c for c in ["idx", "sub_category", "num_objects"] if c in plot_df.columns]
        counts_source = plot_df.drop_duplicates(subset=dedup_cols) if dedup_cols else plot_df

    sub_categories = pd.unique(plot_df["sub_category"])
    if sub_categories.size == 0:
        raise ValueError("No sub_category values found after filtering.")

    model_style, family_map = _build_model_style(plot_df, metadata_path, group_by=group_by)
    rng = np.random.default_rng(seed)

    cols = max(1, n_cols)
    rows = math.ceil(len(sub_categories) / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).flatten()

    global_limits = None
    if y_limit_mode == "zero_to_max":
        if group_by == "family":
            plot_df["group_id"] = plot_df["model_id"].map(family_map).fillna("Other")
        else:
            plot_df["group_id"] = plot_df["model_id"].astype(str)
        agg_all = _macro_avg_by_question(plot_df, ["group_id", "num_objects"])
        if not agg_all.empty:
            global_limits = (
                float(agg_all["accuracy"].min()),
                float(agg_all["accuracy"].max()),
            )

    for i, cat in enumerate(sub_categories):
        ax = axes[i]
        df_cat = plot_df[plot_df["sub_category"] == cat].copy()
        if df_cat.empty:
            ax.set_visible(False)
            continue

        if group_by == "family":
            df_cat["group_id"] = df_cat["model_id"].map(family_map).fillna("Other")
        else:
            df_cat["group_id"] = df_cat["model_id"].astype(str)

        agg_df = _macro_avg_by_question(df_cat, ["group_id", "num_objects"])
        if agg_df.empty:
            ax.set_visible(False)
            continue

        num_values = sorted(pd.unique(agg_df["num_objects"]))
        num_to_pos = {value: idx for idx, value in enumerate(num_values)}
        agg_df["num_objects_pos"] = agg_df["num_objects"].map(num_to_pos)

        sns.violinplot(
            data=agg_df,
            x="num_objects_pos",
            y="accuracy",
            ax=ax,
            color="0.85",
            inner=None,
            cut=0,
            order=list(range(len(num_values))),
        )
        ax.axhline(y=0.25, color="gray", linestyle="--", linewidth=1)
        if split_values and split_values[0] in num_to_pos and split_values[1] in num_to_pos:
            left = num_to_pos[split_values[0]]
            right = num_to_pos[split_values[1]]
            line_x = (left + right) / 2
            ax.axvline(
                x=line_x,
                color="gray",
                linestyle=":",
                linewidth=1,
                **(split_line_kwargs or {}),
            )

        counts_series = (
            counts_source[counts_source["sub_category"] == cat]
            .groupby("num_objects", observed=True)["num_objects"]
            .count()
        )
        if global_limits:
            label_y = global_limits[0] - n_label_offset * max(1e-6, global_limits[1] - global_limits[0])
        else:
            label_y = -0.075
        for num_obj, count in counts_series.items():
            x_pos = num_to_pos.get(num_obj)
            if x_pos is None:
                continue
            ax.text(
                x_pos,
                label_y,
                f"N:{int(count)}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="black",
            )

        for group_id, df_m in agg_df.groupby("group_id"):
            x_vals = df_m["num_objects_pos"].to_numpy()
            y_vals = df_m["accuracy"].to_numpy()
            if x_vals.size == 0:
                continue
            jitter = rng.uniform(-0.2, 0.2, size=x_vals.size)
            x_jittered = x_vals + jitter
            color, marker, size = model_style.get(str(group_id), ("black", "o", 0.5))
            ax.scatter(
                x_jittered,
                y_vals,
                color=color,
                s=40 + 40 * size,
                alpha=0.8,
                edgecolor="white",
                linewidth=0.7,
                marker=marker,
            )

        label = subcategory_label_map.get(str(cat), str(cat)) if subcategory_label_map else str(cat)
        ax.set_title(label)
        ax.set_xlabel("Number of Objects")
        ax.set_ylabel("Accuracy")
        ax.grid(axis="y")
        if y_limit_mode == "fixed":
            ax.set_ylim(-0.1, 1.1)
        elif y_limit_mode == "zero_to_max" and global_limits:
            y_min, y_max = global_limits
            pad = y_pad * max(1e-6, y_max - y_min)
            ax.set_ylim(y_min - n_label_offset * max(1e-6, y_max - y_min), y_max + pad)
        else:
            y_min = agg_df["accuracy"].min()
            y_max = agg_df["accuracy"].max()
            if pd.isna(y_min) or pd.isna(y_max):
                ax.set_ylim(-0.1, 1.1)
            else:
                pad = y_pad * max(1e-6, y_max - y_min)
                ax.set_ylim(y_min - pad, y_max + pad)
        ax.set_xticks(range(len(num_values)))
        ax.set_xticklabels([str(v) for v in num_values])

    for j in range(len(sub_categories), len(axes)):
        axes[j].set_visible(False)

    legend_handles = []
    legend_labels = []
    for group_id, (color, marker, size) in model_style.items():
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                color="none",
                markerfacecolor=color,
                markeredgecolor="white",
                markersize=8 + 6 * size,
                linestyle="None",
            )
        )
        legend_labels.append(group_id)

    fig.legend(
        legend_handles,
        legend_labels,
        title="Model",
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.5, 1.02),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    run = run_name or globals().get("RUN_NAME", "default")
    out_dir = Path(output_dir) if output_dir is not None else Path("output") / run
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / filename, dpi=300, bbox_inches="tight")

    if save_legend:
        fig_legend = plt.figure(figsize=legend_figsize or (5 * legend_cols, 1.0))
        fig_legend.legend(
            legend_handles,
            legend_labels,
            title="Model",
            loc="center",
            ncol=legend_cols,
            frameon=False,
        )
        fig_legend.tight_layout()
        legend_name = legend_filename or filename.replace(".png", "_legend.png")
        fig_legend.savefig(out_dir / legend_name, dpi=300, bbox_inches="tight")
        plt.close(fig_legend)

    if save_per_category:
        sub_dirname = per_category_dirname or filename.replace(".png", "")
        per_cat_dir = out_dir / sub_dirname
        per_cat_dir.mkdir(parents=True, exist_ok=True)
        for cat in sub_categories:
            fig_cat, ax_cat = plt.subplots(1, 1, figsize=(5, 4))
            df_cat = plot_df[plot_df["sub_category"] == cat].copy()
            if df_cat.empty:
                plt.close(fig_cat)
                continue

            if group_by == "family":
                df_cat["group_id"] = df_cat["model_id"].map(family_map).fillna("Other")
            else:
                df_cat["group_id"] = df_cat["model_id"].astype(str)

            agg_df = _macro_avg_by_question(df_cat, ["group_id", "num_objects"])
            if agg_df.empty:
                plt.close(fig_cat)
                continue

            num_values = sorted(pd.unique(agg_df["num_objects"]))
            num_to_pos = {value: idx for idx, value in enumerate(num_values)}
            agg_df["num_objects_pos"] = agg_df["num_objects"].map(num_to_pos)

            sns.violinplot(
                data=agg_df,
                x="num_objects_pos",
                y="accuracy",
                ax=ax_cat,
                color="0.85",
                inner=None,
                cut=0,
                order=list(range(len(num_values))),
            )
            ax_cat.axhline(y=0.25, color="gray", linestyle="--", linewidth=1)
            if split_values and split_values[0] in num_to_pos and split_values[1] in num_to_pos:
                left = num_to_pos[split_values[0]]
                right = num_to_pos[split_values[1]]
                line_x = (left + right) / 2
                ax_cat.axvline(
                    x=line_x,
                    color="gray",
                    linestyle=":",
                    linewidth=1,
                    **(split_line_kwargs or {}),
                )

            counts_series = (
                counts_source[counts_source["sub_category"] == cat]
                .groupby("num_objects", observed=True)["num_objects"]
                .count()
            )
            label_y = agg_df["accuracy"].min() - n_label_offset * max(1e-6, agg_df["accuracy"].max() - agg_df["accuracy"].min())
            for num_obj, count in counts_series.items():
                x_pos = num_to_pos.get(num_obj)
                if x_pos is None:
                    continue
                ax_cat.text(
                    x_pos,
                    label_y,
                    f"N:{int(count)}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="black",
                )

            for group_id, df_m in agg_df.groupby("group_id"):
                x_vals = df_m["num_objects_pos"].to_numpy()
                y_vals = df_m["accuracy"].to_numpy()
                if x_vals.size == 0:
                    continue
                jitter = rng.uniform(-0.2, 0.2, size=x_vals.size)
                x_jittered = x_vals + jitter
                color, marker, size = model_style.get(str(group_id), ("black", "o", 0.5))
                ax_cat.scatter(
                    x_jittered,
                    y_vals,
                    color=color,
                    s=40 + 40 * size,
                    alpha=0.8,
                    edgecolor="white",
                    linewidth=0.7,
                    marker=marker,
                )

            label = subcategory_label_map.get(str(cat), str(cat)) if subcategory_label_map else str(cat)
            ax_cat.set_title(label)
            ax_cat.set_xlabel("Number of Objects")
            ax_cat.set_ylabel("Accuracy")
            ax_cat.grid(axis="y")
            if y_limit_mode == "fixed":
                ax_cat.set_ylim(-0.1, 1.1)
            elif y_limit_mode == "zero_to_max":
                y_min = float(agg_df["accuracy"].min())
                y_max = float(agg_df["accuracy"].max())
                pad = y_pad * max(1e-6, y_max - y_min)
                ax_cat.set_ylim(y_min - n_label_offset * max(1e-6, y_max - y_min), y_max + pad)
            else:
                y_min = agg_df["accuracy"].min()
                y_max = agg_df["accuracy"].max()
                if pd.isna(y_min) or pd.isna(y_max):
                    ax_cat.set_ylim(-0.1, 1.1)
                else:
                    pad = y_pad * max(1e-6, y_max - y_min)
                    ax_cat.set_ylim(y_min - pad, y_max + pad)
            ax_cat.set_xticks(range(len(num_values)))
            ax_cat.set_xticklabels([str(v) for v in num_values])

            fig_cat.tight_layout()
            safe_cat = _safe_filename(str(cat))
            fig_cat.savefig(per_cat_dir / f"num_objects_{safe_cat}.png", dpi=300, bbox_inches="tight")
            plt.close(fig_cat)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
