from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.markers as mmarkers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", message=".*edgecolor.*unfilled marker.*")

# /data0/sebastian.cavada/compositional-physics/tiny_vqa_deterministic/output/run_11_general_levels

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

    family_markers = {
        fam: markers[i % len(markers)] for i, fam in enumerate(unique_families)
    }

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


def create_scatter_by_family(
    eval_df: pd.DataFrame,
    *,
    metadata_path: str | Path | None = "utils/metadata.json",
    levels_sorted: list[str] | None = None,
    split_by_mode: bool = False,
    modes: list[str] | None = None,
    show: bool = True,
    output_dir: str | Path | None = None,
    run_name: str | None = None,
    filename: str = "levels_by_family.png",
) -> plt.Figure:
    levels_sorted = levels_sorted or [
        "baseline",
        "child",
        "teen",
        "undegrad",
        "graduate",
        "expert",
    ]

    if "model_id" not in eval_df.columns:
        raise KeyError("eval_df must include 'model_id'.")
    if "level" not in eval_df.columns:
        raise KeyError("eval_df must include 'level'.")
    if "sub_category" not in eval_df.columns:
        raise KeyError("eval_df must include 'sub_category'.")

    question_col = (
        "question_id_base" if "question_id_base" in eval_df.columns else "question_id"
    )
    if question_col not in eval_df.columns:
        raise KeyError("eval_df must include 'question_id' or 'question_id_base'.")

    if "accuracy" in eval_df.columns:
        eval_df = eval_df.copy()
        eval_df["accuracy"] = pd.to_numeric(eval_df["accuracy"], errors="coerce")
    elif "is_correct" in eval_df.columns:
        eval_df = eval_df.copy()
        eval_df["accuracy"] = pd.to_numeric(eval_df["is_correct"], errors="coerce")
    else:
        raise KeyError("eval_df must include 'accuracy' or 'is_correct'.")

    family_style, family_map = _build_model_style(
        eval_df,
        metadata_path=metadata_path,
        group_by="family",
    )
    eval_df = eval_df.copy()
    eval_df["family"] = eval_df["model_id"].map(family_map).fillna("Other")

    metadata_df = None
    if metadata_path is not None and Path(metadata_path).exists():
        metadata_df = _load_model_metadata(metadata_path)
    mode_map = {}
    if metadata_df is not None and "mode" in metadata_df.columns:
        mode_map = (
            metadata_df.dropna(subset=["model_id"])
            .set_index("model_id")["mode"]
            .to_dict()
        )

    line_styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 3))]

    def _plot_subset(
        plot_df: pd.DataFrame, title_suffix: str, output_name: str
    ) -> plt.Figure:
        q_scores = (
            plot_df.groupby(
                ["family", question_col, "level", "sub_category"], observed=True
            )["accuracy"]
            .mean()
            .reset_index()
            .dropna()
        )

        subcat_scores = (
            q_scores.groupby(["family", "level", "sub_category"], observed=True)[
                "accuracy"
            ]
            .mean()
            .reset_index(name="subcat_acc")
            .dropna()
        )

        final_plot_df = (
            subcat_scores.groupby(["family", "level"], observed=True)["subcat_acc"]
            .agg(mean_accuracy="mean", band_width="sem")
            .reset_index()
            .dropna()
        )

        level_map = {lvl: i for i, lvl in enumerate(levels_sorted)}
        final_plot_df["level_idx"] = final_plot_df["level"].map(level_map)

        fig, ax = plt.subplots(figsize=(12, 6))
        for i, (family, fam_data) in enumerate(final_plot_df.groupby("family")):
            fam_data = fam_data.sort_values("level_idx")
            if fam_data.empty:
                continue
            x = fam_data["level_idx"].values
            y = fam_data["mean_accuracy"].values
            y_err = fam_data["band_width"].values
            color, marker, _size = family_style.get(family, ("black", "o", 0.5))
            ls = line_styles[i % len(line_styles)]
            ax.plot(
                x, y, color=color, linestyle=ls, linewidth=2, alpha=0.85, label=family
            )
            ax.scatter(
                x, y, color=color, marker=marker, s=60, edgecolor="white", linewidth=0.7
            )
            if np.isfinite(y_err).any():
                ax.fill_between(x, y - y_err, y + y_err, color=color, alpha=0.12)

        ax.set_xticks(range(len(levels_sorted)))
        nice_labels = [level.capitalize() for level in levels_sorted]
        ax.set_xticklabels(nice_labels, fontsize=15, fontweight="bold")
        ax.set_xlabel("Difficulty Level", fontsize=16)
        ax.set_ylabel("Accuracy", fontsize=16)
        handles, labels = ax.get_legend_handles_labels()
        ncol = max(1, len(labels))
        ax.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=ncol,
            frameon=True,
            framealpha=0.9,
            fontsize=11,
        )
        ax.grid(True, linestyle="--", alpha=0.5)

        fig.tight_layout()

        run = run_name or globals().get("RUN_NAME", "default")
        out_dir = Path(output_dir) if output_dir is not None else Path("output") / run
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / output_name, dpi=300, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    if split_by_mode and mode_map:
        modes = modes or ["image-only", "general"]
        figures = []
        for mode in modes:
            mode_models = [mid for mid, m in mode_map.items() if m == mode]
            subset = eval_df[eval_df["model_id"].isin(mode_models)]
            if subset.empty:
                continue
            suffix = f" ({mode})"
            out_name = filename.replace(".png", f"_{_safe_filename(mode)}.png")
            figures.append(_plot_subset(subset, suffix, out_name))
        if figures:
            return figures[0]

    return _plot_subset(eval_df, "", filename)
