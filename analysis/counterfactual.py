from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

from utils.utils_read import _sanitize_answer, load_results


DEFAULT_BASE_PATH = Path(
    # "/Users/sebastiancavada/Desktop/tmp_paris/tiny_vqa_creation/output"
    "/data0/sebastian.cavada/compositional-physics/tiny_vqa_creation/output"
)

RUN = "28"

DEFAULT_RUNS = [
    f"run_{RUN}_counterfactual_shift",
    f"run_{RUN}_counterfactual_smaller",
    f"run_{RUN}_counterfactual_gravity",
]

RUN_MAP = {
    f"run_{RUN}_counterfactual_shift": "Shift",
    f"run_{RUN}_counterfactual_smaller": "Scaled",
    f"run_{RUN}_counterfactual_gravity": "Gravity",
}

FAMILY_STYLE = {
    "InternVLChat2": {"label": "InternVLChat2", "marker": "o", "color": "#3D73A9"},
    "LLaVAInterleave": {"label": "LLaVAInterleave", "marker": "<", "color": "#4E973F"},
    "LLaVAVideo": {"label": "LLaVAVideo", "marker": "s", "color": "#AAB8CF"},
    "Mantis": {"label": "Mantis", "marker": "^", "color": "#E3873A"},
    "Owl3": {"label": "Owl3", "marker": ">", "color": "#9DCB8C"},
    "Phi": {"label": "Phi", "marker": "v", "color": "#E7C79D"},
    "VILAModel": {"label": "VILAModel", "marker": "p", "color": "#C84039"},
}

FAMILY_ORDER = [
    FAMILY_STYLE["InternVLChat2"]["label"],
    FAMILY_STYLE["LLaVAInterleave"]["label"],
    FAMILY_STYLE["LLaVAVideo"]["label"],
    FAMILY_STYLE["Mantis"]["label"],
    FAMILY_STYLE["Owl3"]["label"],
    FAMILY_STYLE["Phi"]["label"],
    FAMILY_STYLE["VILAModel"]["label"],
]

FAMILY_MARKERS = {
    style["label"]: style["marker"] for style in FAMILY_STYLE.values()
}
FAMILY_COLORS = {
    style["label"]: style["color"] for style in FAMILY_STYLE.values()
}

PLOT_BG = "#FFFFFF"
GRID_MAJOR = "#E6E6E6"
GRID_MINOR = "#F2F2F2"

OBJECT_COUNT_PATTERN = re.compile(r"(?:^|[\\/_-])no-(\d+)(?:$|[\\/_-])")
IMAGE_MODE_PATTERN = re.compile(r"_(i|g)(?:$|[_-])")
ReducerName = Literal["mean", "median", "min", "max", "weighted_mean"]
REDUCER_CHOICES = ("mean", "median", "min", "max", "weighted_mean")


def load_metadata_map(metadata_path: Path) -> dict[str, dict]:
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    return {str(item["id"]): item for item in metadata if "id" in item}


def format_family_name(raw_family: str) -> str:
    style = FAMILY_STYLE.get(raw_family)
    if style is None:
        return str(raw_family)
    return str(style["label"])


def _reduce_values(
    values: pd.Series,
    reducer: ReducerName,
    *,
    weights: pd.Series | None = None,
) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.notna()
    if not valid.any():
        return float("nan")

    reduced_values = numeric[valid].astype(float)

    if reducer == "mean":
        return float(reduced_values.mean())
    if reducer == "median":
        return float(reduced_values.median())
    if reducer == "min":
        return float(reduced_values.min())
    if reducer == "max":
        return float(reduced_values.max())

    if weights is None:
        return float(reduced_values.mean())

    numeric_weights = pd.to_numeric(weights, errors="coerce").reindex(values.index)
    weight_slice = numeric_weights[valid].fillna(0.0).astype(float)
    weight_mask = weight_slice > 0.0
    if weight_mask.any():
        return float(np.average(reduced_values[weight_mask], weights=weight_slice[weight_mask]))
    return float(reduced_values.mean())


def _aggregate_grouped_metric(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    reducer: ReducerName,
    *,
    weight_col: str | None = None,
    output_col: str = "macro_accuracy",
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    grouped = df.groupby(group_cols, observed=True, sort=False)
    for group_key, frame in grouped:
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        row: dict[str, object] = {
            col_name: key_values[idx] for idx, col_name in enumerate(group_cols)
        }
        weights = frame[weight_col] if weight_col and weight_col in frame.columns else None
        row[output_col] = _reduce_values(frame[value_col], reducer, weights=weights)
        records.append(row)
    if not records:
        return pd.DataFrame(columns=[*group_cols, output_col])
    return pd.DataFrame.from_records(records)


def _extract_image_mode(idx_value: object) -> str:
    if idx_value is None:
        return "unknown"
    text = str(idx_value)
    match = IMAGE_MODE_PATTERN.search(text)
    if not match:
        return "unknown"
    code = match.group(1)
    if code == "i":
        return "single"
    if code == "g":
        return "multi"
    return "unknown"


def _extract_object_count(value: object) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (list, tuple, set, np.ndarray)):
        text = " ".join(str(v) for v in value)
    else:
        try:
            if pd.isna(value):
                return float("nan")
        except (TypeError, ValueError):
            pass
        text = str(value)

    match = OBJECT_COUNT_PATTERN.search(text)
    if not match:
        return float("nan")
    try:
        return float(int(match.group(1)))
    except (TypeError, ValueError):
        return float("nan")


def _ensure_object_count_column(df: pd.DataFrame) -> pd.DataFrame:
    if "object_count" in df.columns:
        df["object_count"] = pd.to_numeric(df["object_count"], errors="coerce")
        return df

    if "num_objects" in df.columns:
        df["object_count"] = pd.to_numeric(df["num_objects"], errors="coerce")
        return df

    source_cols = [col for col in ["simulation_id", "file_name", "idx"] if col in df.columns]
    if not source_cols:
        return df

    inferred = pd.Series(np.nan, index=df.index, dtype="float64")
    for col in source_cols:
        inferred = inferred.fillna(df[col].apply(_extract_object_count))

    if inferred.notna().any():
        df["object_count"] = inferred
        print("Inferred object_count from cached fields.")

    return df


def _resolve_object_count_column(df: pd.DataFrame) -> str | None:
    if "object_count" in df.columns:
        return "object_count"
    if "num_objects" in df.columns:
        return "num_objects"
    return None


def build_eval_df(base_path: Path, run_name: str) -> pd.DataFrame:
    results_dir = base_path / run_name / f"results_{run_name}_sanitized"
    if not results_dir.exists():
        raise FileNotFoundError(
            f"Missing sanitized results directory for {run_name}: {results_dir}"
        )

    model_cols = sorted(
        p.stem.replace("_val", "") for p in results_dir.glob("*_val.json")
    )
    if not model_cols:
        raise FileNotFoundError(
            f"No model result files found in sanitized directory: {results_dir}"
        )

    try:
        df = load_results(
            base_path,
            run_name,
            merge_model_answers=True,
            model_answers_wide=True,
            model_results_dir=results_dir,
            cache=True,
            add_sim_metadata=True,
        )
    except FileNotFoundError as exc:
        print(
            f"Metadata load failed for {run_name} ({exc}). "
            "Retrying without simulation metadata."
        )
        df = load_results(
            base_path,
            run_name,
            merge_model_answers=True,
            model_answers_wide=True,
            model_results_dir=results_dir,
            cache=False,
            add_sim_metadata=False,
        )
    df = _ensure_object_count_column(df)

    model_cols = [col for col in model_cols if col in df.columns]
    if not model_cols:
        raise ValueError(
            f"No matching model columns after loading results for {run_name}."
        )

    df["answer"] = df["answer"].apply(
        lambda answer: _sanitize_answer(answer, max_prefix_chars=None)
    )

    id_cols = [
        col
        for col in [
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
        if col in df.columns
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
    if "idx" in eval_df.columns:
        eval_df["image_mode"] = eval_df["idx"].apply(_extract_image_mode)
    else:
        eval_df["image_mode"] = "unknown"

    return eval_df


def collect_runs(base_path: Path, runs: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run_name in runs:
        print(f"Processing run: {run_name}")
        run_df = build_eval_df(base_path, run_name)
        if run_df.empty:
            raise ValueError(f"Run has no rows after loading: {run_name}")
        run_df["run_name"] = run_name
        frames.append(run_df)

    if not frames:
        raise ValueError("No data available for the selected runs.")

    return pd.concat(frames, ignore_index=True)


def build_run_axis(runs: list[str]) -> tuple[dict[str, str], dict[str, int]]:
    ordered_runs: list[str] = []
    seen: set[str] = set()
    for run_name in runs:
        if run_name in seen:
            continue
        seen.add(run_name)
        ordered_runs.append(run_name)

    run_label_map = {run_name: RUN_MAP.get(run_name, run_name) for run_name in ordered_runs}
    run_index = {run_name: idx for idx, run_name in enumerate(ordered_runs)}
    return run_label_map, run_index


def _accuracy_percent(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return float("nan")
    return float(numeric.mean() * 100.0)


def _count_valid(series: pd.Series) -> int:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return int(numeric.shape[0])


def _family_for_model(model_id: str, metadata_map: dict[str, dict]) -> str:
    item = metadata_map.get(str(model_id), {})
    return str(item.get("family", "Unknown"))


def aggregate_by_object_count(
    full_df: pd.DataFrame,
    metadata_map: dict[str, dict],
    macro_reducer: ReducerName = "mean",
) -> pd.DataFrame:
    count_col = _resolve_object_count_column(full_df)
    if count_col is None:
        print("Skipping object-count aggregation: no object count column found.")
        return pd.DataFrame(
            columns=[
                "run_name",
                "model_id",
                "object_count",
                "macro_accuracy",
                "total_questions",
                "model_family",
                "family_label",
                "image_mode",
            ]
        )

    group_cols = ["run_name", "model_id", "sub_category", count_col]
    if "image_mode" in full_df.columns:
        group_cols.append("image_mode")

    model_sub_obj = (
        full_df.groupby(group_cols, observed=True)
        .agg(
            model_accuracy=("is_correct", _accuracy_percent),
            total_questions=("is_correct", _count_valid),
        )
        .reset_index()
    )

    macro_group_cols = ["run_name", "model_id", count_col]
    if "image_mode" in model_sub_obj.columns:
        macro_group_cols.append("image_mode")

    grouped_acc = _aggregate_grouped_metric(
        model_sub_obj,
        macro_group_cols,
        "model_accuracy",
        macro_reducer,
        weight_col="total_questions",
        output_col="macro_accuracy",
    )
    grouped_counts = (
        model_sub_obj.groupby(macro_group_cols, observed=True)["total_questions"]
        .sum()
        .reset_index()
    )
    grouped = grouped_acc.merge(grouped_counts, on=macro_group_cols, how="left")
    if count_col != "object_count":
        grouped = grouped.rename(columns={count_col: "object_count"})

    grouped["model_family"] = grouped["model_id"].map(
        lambda mid: _family_for_model(mid, metadata_map)
    )
    grouped["family_label"] = grouped["model_family"].map(format_family_name)
    grouped["object_count"] = pd.to_numeric(grouped["object_count"], errors="coerce")
    grouped = grouped.dropna(subset=["object_count"]).copy()
    grouped["object_count"] = grouped["object_count"].astype(int)
    if "image_mode" not in grouped.columns:
        grouped["image_mode"] = "unknown"
    return grouped


def aggregate_macro_by_model(
    full_df: pd.DataFrame,
    metadata_map: dict[str, dict],
    min_object_count: int,
    macro_reducer: ReducerName = "mean",
) -> pd.DataFrame:
    count_col = _resolve_object_count_column(full_df)
    if count_col is None:
        print("No object count column found; skipping min-object-count filter.")
        filtered = full_df.copy()
    else:
        filtered = full_df[
            pd.to_numeric(full_df[count_col], errors="coerce") >= min_object_count
        ]

    group_cols = ["run_name", "model_id", "sub_category"]
    if "image_mode" in filtered.columns:
        group_cols.append("image_mode")

    model_sub = (
        filtered.groupby(group_cols, observed=True)
        .agg(
            model_accuracy=("is_correct", _accuracy_percent),
            total_questions=("is_correct", _count_valid),
        )
        .reset_index()
    )

    macro_group_cols = ["run_name", "model_id"]
    if "image_mode" in model_sub.columns:
        macro_group_cols.append("image_mode")

    grouped_acc = _aggregate_grouped_metric(
        model_sub,
        macro_group_cols,
        "model_accuracy",
        macro_reducer,
        weight_col="total_questions",
        output_col="macro_accuracy",
    )
    grouped_counts = (
        model_sub.groupby(macro_group_cols, observed=True)["total_questions"]
        .sum()
        .reset_index()
    )
    grouped = grouped_acc.merge(grouped_counts, on=macro_group_cols, how="left")

    grouped["model_family"] = grouped["model_id"].map(
        lambda mid: _family_for_model(mid, metadata_map)
    )
    grouped["family_label"] = grouped["model_family"].map(format_family_name)
    if "image_mode" not in grouped.columns:
        grouped["image_mode"] = "unknown"
    return grouped


def aggregate_family_summary(
    grouped_df: pd.DataFrame,
    family_reducer: ReducerName = "mean",
    *,
    include_image_mode: bool = False,
) -> pd.DataFrame:
    if grouped_df.empty:
        base_columns = ["run_name", "family_label", "macro_accuracy", "total_questions", "model_count"]
        if include_image_mode:
            base_columns.insert(2, "image_mode")
        return pd.DataFrame(columns=base_columns)

    group_cols = ["run_name", "family_label"]
    if include_image_mode and "image_mode" in grouped_df.columns:
        group_cols.append("image_mode")

    family_acc = _aggregate_grouped_metric(
        grouped_df,
        group_cols,
        "macro_accuracy",
        family_reducer,
        weight_col="total_questions",
        output_col="macro_accuracy",
    )
    family_questions = (
        grouped_df.groupby(group_cols, observed=True)["total_questions"]
        .sum()
        .reset_index()
    )
    family_models = (
        grouped_df.groupby(group_cols, observed=True)["model_id"]
        .nunique()
        .rename("model_count")
        .reset_index()
    )
    return (
        family_acc.merge(family_questions, on=group_cols, how="left")
        .merge(family_models, on=group_cols, how="left")
    )


def _legend_handles_labels(ax_list: list[plt.Axes]) -> tuple[list, list[str]]:
    handle_map = {}
    for ax in ax_list:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label not in handle_map:
                handle_map[label] = handle
    labels = list(handle_map.keys())
    handles = [handle_map[label] for label in labels]
    return handles, labels


def _apply_reference_style(ax: plt.Axes, *, y_label: bool = False) -> None:
    ax.set_facecolor(PLOT_BG)
    ax.grid(axis="y", which="major", linestyle="-", linewidth=0.9, color=GRID_MAJOR)
    ax.grid(axis="y", which="minor", linestyle="-", linewidth=0.7, color=GRID_MINOR)
    ax.grid(axis="x", visible=False)
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(2.5))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.6)
    ax.spines["bottom"].set_linewidth(1.6)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")

    if y_label:
        ax.set_ylabel("Accuracy (%)", fontsize=18, fontweight="bold")
    ax.tick_params(axis="y", labelsize=15, width=1.2)
    ax.tick_params(axis="x", width=1.2)


def _compute_y_limits(
    values: pd.Series,
    margin_ratio: float = 0.08,
    y_floor: float | None = None,
) -> tuple[float, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return 0.0, 1.0

    y_min = float(numeric.min())
    y_max = float(numeric.max())
    span = y_max - y_min
    if span <= 0.0:
        pad = max(0.5, abs(y_min) * 0.05)
    else:
        pad = span * margin_ratio
    y_low = y_min - pad
    y_high = y_max + pad
    if y_floor is not None:
        y_low = max(float(y_floor), y_low)
    return y_low, y_high


def _get_object_count_thresholds(full_df: pd.DataFrame) -> list[int]:
    count_col = _resolve_object_count_column(full_df)
    if count_col is None:
        return []
    numeric = pd.to_numeric(full_df[count_col], errors="coerce").dropna()
    if numeric.empty:
        return []
    return sorted(pd.unique(numeric.astype(int)))


def plot_object_count_scatter(
    full_df: pd.DataFrame,
    metadata_map: dict[str, dict],
    run_label_map: dict[str, str],
    run_index: dict[str, int],
    output_path: Path,
    *,
    macro_reducer: ReducerName = "mean",
    family_reducer: ReducerName = "mean",
    y_floor: float = 10.0,
) -> None:
    object_counts = _get_object_count_thresholds(full_df)
    if not object_counts:
        print("Skipping object-count plot: no object-count rows available.")
        return

    cols = 5
    rows = math.ceil(len(object_counts) / cols)
    fig, axes = plt.subplots(
        nrows=rows,
        ncols=cols,
        figsize=(5.8 * cols, 5.2 * rows),
        sharex=True,
        sharey=False,
    )
    fig.patch.set_facecolor(PLOT_BG)
    axes_flat = np.array(axes).reshape(-1)

    for idx, ax in enumerate(axes_flat):
        if idx >= len(object_counts):
            ax.axis("off")
            continue

        object_count_threshold = int(object_counts[idx])
        subset = aggregate_macro_by_model(
            full_df,
            metadata_map,
            min_object_count=object_count_threshold,
            macro_reducer=macro_reducer,
        )
        subset_family = aggregate_family_summary(subset, family_reducer=family_reducer)
        subset_plot = subset_family[subset_family["family_label"].isin(FAMILY_ORDER)]
        ax.set_title(
            f"Object Count >= {object_count_threshold}",
            fontsize=16,
            fontweight="bold",
            pad=10,
        )
        _apply_reference_style(ax, y_label=(idx % cols == 0))
        y_min, y_max = _compute_y_limits(subset_plot["macro_accuracy"], y_floor=y_floor)
        ax.set_ylim(y_min, y_max)

        for family_label in FAMILY_ORDER:
            family_df = subset_family[subset_family["family_label"] == family_label]
            if family_df.empty:
                continue

            run_df = family_df.copy()
            run_df = run_df[run_df["run_name"].isin(run_index)].copy()
            if run_df.empty:
                continue

            run_df["run_idx"] = run_df["run_name"].map(run_index)
            run_df = run_df.sort_values("run_idx")
            marker = FAMILY_MARKERS.get(family_label, "o")
            color = FAMILY_COLORS.get(family_label, "#4C72B0")

            ax.scatter(
                run_df["run_idx"],
                run_df["macro_accuracy"],
                label=family_label,
                s=260,
                alpha=0.95,
                marker=marker,
                color=color,
                edgecolors="none",
            )

        ax.set_xticks(list(range(len(run_label_map))))
        ax.set_xticklabels(
            list(run_label_map.values()),
            rotation=25,
            ha="right",
            fontsize=14,
            fontweight="bold",
        )

    used_axes = [ax for ax in axes_flat[: len(object_counts)]]
    handles, labels = _legend_handles_labels(used_axes)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=len(labels),
            title="Model Family",
            fontsize=13,
            title_fontsize=14,
            frameon=True,
            facecolor=PLOT_BG,
            edgecolor="#4f4f4f",
            columnspacing=1.2,
            handletextpad=0.5,
            borderpad=0.6,
        )
        fig.tight_layout(rect=(0.0, 0.18, 1.0, 1.0))
    else:
        fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved object-count scatter plot: {output_path}")


def _family_jitter_offsets(jitter_x: float) -> dict[str, float]:
    if jitter_x <= 0.0 or len(FAMILY_ORDER) <= 1:
        return {label: 0.0 for label in FAMILY_ORDER}
    offsets = np.linspace(-float(jitter_x), float(jitter_x), len(FAMILY_ORDER))
    return {label: float(offsets[idx]) for idx, label in enumerate(FAMILY_ORDER)}


def _scatter_family_points(
    ax: plt.Axes,
    family_summary_df: pd.DataFrame,
    run_index: dict[str, int],
    *,
    jitter_map: dict[str, float] | None = None,
) -> None:
    if jitter_map is None:
        jitter_map = {label: 0.0 for label in FAMILY_ORDER}

    grouped_df_plot = family_summary_df[family_summary_df["family_label"].isin(FAMILY_ORDER)]

    for family_label in FAMILY_ORDER:
        family_df = grouped_df_plot[grouped_df_plot["family_label"] == family_label]
        if family_df.empty:
            continue

        run_df = family_df.copy()
        run_df = run_df[run_df["run_name"].isin(run_index)].copy()
        if run_df.empty:
            continue

        run_df["run_idx"] = run_df["run_name"].map(run_index).astype(float)
        run_df["run_idx_jittered"] = run_df["run_idx"] + float(jitter_map.get(family_label, 0.0))
        run_df = run_df.sort_values("run_idx")

        ax.scatter(
            run_df["run_idx_jittered"],
            run_df["macro_accuracy"],
            s=270,
            marker=FAMILY_MARKERS.get(family_label, "o"),
            color=FAMILY_COLORS.get(family_label, "#4C72B0"),
            label=family_label,
            alpha=0.95,
            edgecolors="none",
        )


def plot_family_scatter(
    family_summary_df: pd.DataFrame,
    run_label_map: dict[str, str],
    run_index: dict[str, int],
    output_path: Path,
    *,
    title: str | None = None,
    y_floor: float = 10.0,
    jitter_x: float = 0.0,
    x_side_padding: float = 0.45,
) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 5.7))
    fig.patch.set_facecolor(PLOT_BG)
    _apply_reference_style(ax, y_label=True)

    grouped_df_plot = family_summary_df[family_summary_df["family_label"].isin(FAMILY_ORDER)]
    y_min, y_max = _compute_y_limits(grouped_df_plot["macro_accuracy"], y_floor=y_floor)
    ax.set_ylim(y_min, y_max)
    if title:
        ax.set_title(title, fontsize=18, fontweight="bold", pad=10)

    jitter_map = _family_jitter_offsets(jitter_x)
    _scatter_family_points(ax, grouped_df_plot, run_index, jitter_map=jitter_map)

    ax.set_xticks(list(range(len(run_label_map))))
    ax.set_xticklabels(
        list(run_label_map.values()),
        rotation=25,
        ha="right",
        fontsize=17,
        fontweight="bold",
    )
    ax.set_xlim(-float(x_side_padding), (len(run_label_map) - 1) + float(x_side_padding))
    ax.set_xlabel("", fontsize=1)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            labels,
            title="Model Family",
            title_fontsize=14,
            fontsize=13,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
            facecolor=PLOT_BG,
            edgecolor="#4f4f4f",
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved family scatter plot: {output_path}")


def plot_family_scatter_side_by_side(
    family_summary_by_mode_df: pd.DataFrame,
    run_label_map: dict[str, str],
    run_index: dict[str, int],
    output_path: Path,
    *,
    y_floor: float = 10.0,
    jitter_x: float = 0.10,
    x_side_padding: float = 0.55,
) -> None:
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16.8, 6.0), sharey=True)
    fig.patch.set_facecolor(PLOT_BG)
    axes_list = np.array(axes).reshape(-1)

    mode_specs = [
        ("single", "Single Image Questions"),
        ("multi", "Multi Image Questions"),
    ]
    plot_df = family_summary_by_mode_df[
        (family_summary_by_mode_df["image_mode"].isin(["single", "multi"]))
        & (family_summary_by_mode_df["family_label"].isin(FAMILY_ORDER))
    ].copy()
    y_min, y_max = _compute_y_limits(plot_df["macro_accuracy"], y_floor=y_floor)
    jitter_map = _family_jitter_offsets(jitter_x)

    for idx, (mode_name, panel_title) in enumerate(mode_specs):
        ax = axes_list[idx]
        _apply_reference_style(ax, y_label=(idx == 0))
        ax.set_title(panel_title, fontsize=17, fontweight="bold", pad=10)
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(-float(x_side_padding), (len(run_label_map) - 1) + float(x_side_padding))

        mode_df = plot_df[plot_df["image_mode"] == mode_name].copy()
        if mode_df.empty:
            ax.text(
                0.5,
                0.5,
                "No data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=13,
                color="#555555",
            )
        else:
            _scatter_family_points(ax, mode_df, run_index, jitter_map=jitter_map)

        ax.set_xticks(list(range(len(run_label_map))))
        ax.set_xticklabels(
            list(run_label_map.values()),
            rotation=25,
            ha="right",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("", fontsize=1)

    handles, labels = _legend_handles_labels(list(axes_list))
    if handles:
        fig.legend(
            handles,
            labels,
            title="Model Family",
            title_fontsize=14,
            fontsize=12,
            loc="center left",
            bbox_to_anchor=(0.86, 0.5),
            frameon=True,
            facecolor=PLOT_BG,
            edgecolor="#4f4f4f",
        )

    fig.subplots_adjust(left=0.07, right=0.84, bottom=0.20, top=0.88, wspace=0.12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved family side-by-side scatter plot: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run counterfactual analysis with family-level scatter plots."
        )
    )
    parser.add_argument("--base-path", type=Path, default=DEFAULT_BASE_PATH)
    parser.add_argument("--metadata-path", type=Path, default=Path("utils/metadata.json"))
    parser.add_argument("--runs", nargs="*", default=DEFAULT_RUNS)
    parser.add_argument("--min-object-count", type=int, default=5)
    parser.add_argument(
        "--macro-reducer",
        type=str,
        choices=REDUCER_CHOICES,
        default="mean",
        help=(
            "Reducer used to combine per-subcategory model accuracies into a single "
            "model macro score."
        ),
    )
    parser.add_argument(
        "--family-reducer",
        type=str,
        choices=REDUCER_CHOICES,
        default="mean",
        help=(
            "Reducer used to combine model scores into a family score for plotting."
        ),
    )
    parser.add_argument(
        "--jitter-x",
        type=float,
        default=0.10,
        help="Horizontal jitter applied per-family to reduce point overlap.",
    )
    parser.add_argument(
        "--x-side-padding",
        type=float,
        default=0.55,
        help="Padding added on the left and right sides of x-axis.",
    )
    parser.add_argument(
        "--y-floor",
        type=float,
        default=10.0,
        help="Lower y-axis bound for accuracy plots.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output/counterfactual"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    macro_reducer = args.macro_reducer
    family_reducer = args.family_reducer

    metadata_map = load_metadata_map(args.metadata_path)
    full_df = collect_runs(args.base_path, args.runs)
    run_label_map, run_index = build_run_axis(args.runs)

    object_count_df = aggregate_by_object_count(
        full_df,
        metadata_map,
        macro_reducer=macro_reducer,
    )
    macro_df = aggregate_macro_by_model(
        full_df,
        metadata_map,
        min_object_count=args.min_object_count,
        macro_reducer=macro_reducer,
    )
    family_summary = aggregate_family_summary(
        macro_df,
        family_reducer=family_reducer,
        include_image_mode=False,
    )
    family_summary_by_mode = aggregate_family_summary(
        macro_df,
        family_reducer=family_reducer,
        include_image_mode=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    object_count_csv = args.output_dir / "macro_by_model_object_count.csv"
    macro_csv = args.output_dir / "macro_by_model.csv"
    family_csv = args.output_dir / "family_macro_by_run.csv"
    family_mode_csv = args.output_dir / "family_macro_by_run_image_mode.csv"

    object_count_df.to_csv(object_count_csv, index=False)
    macro_df.to_csv(macro_csv, index=False)
    family_summary.to_csv(family_csv, index=False)
    family_summary_by_mode.to_csv(family_mode_csv, index=False)

    print(f"Saved CSV: {object_count_csv}")
    print(f"Saved CSV: {macro_csv}")
    print(f"Saved CSV: {family_csv}")
    print(f"Saved CSV: {family_mode_csv}")

    plot_object_count_scatter(
        full_df,
        metadata_map,
        run_label_map,
        run_index,
        args.output_dir / "macro_accuracy_by_object_count_scatter.png",
        macro_reducer=macro_reducer,
        family_reducer=family_reducer,
        y_floor=args.y_floor,
    )
    plot_family_scatter(
        family_summary,
        run_label_map,
        run_index,
        args.output_dir / "macro_accuracy_by_family_scatter.png",
        title="All Counterfactual Questions",
        y_floor=args.y_floor,
        jitter_x=args.jitter_x,
        x_side_padding=args.x_side_padding,
    )
    mode_plot_settings = [
        ("single", "Single Image Questions", "macro_accuracy_by_family_scatter_single.png"),
        ("multi", "Multi Image Questions", "macro_accuracy_by_family_scatter_multi.png"),
    ]
    for mode_name, mode_title, mode_filename in mode_plot_settings:
        mode_df = family_summary_by_mode[family_summary_by_mode["image_mode"] == mode_name].copy()
        if mode_df.empty:
            print(f"Skipping {mode_name} plot: no rows detected for image_mode={mode_name}.")
            continue
        plot_family_scatter(
            mode_df,
            run_label_map,
            run_index,
            args.output_dir / mode_filename,
            title=mode_title,
            y_floor=args.y_floor,
            jitter_x=args.jitter_x,
            x_side_padding=args.x_side_padding,
        )
    plot_family_scatter_side_by_side(
        family_summary_by_mode,
        run_label_map,
        run_index,
        args.output_dir / "macro_accuracy_by_family_scatter_single_multi.png",
        y_floor=args.y_floor,
        jitter_x=args.jitter_x,
        x_side_padding=args.x_side_padding,
    )


if __name__ == "__main__":
    main()
