from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

from utils.utils_read import _sanitize_answer, load_results


DEFAULT_BASE_PATH = Path(
    "/Users/sebastiancavada/Desktop/tmp_paris/tiny_vqa_creation/output"
)

RUN = "25"

DEFAULT_RUNS = [
    f"run_{RUN}_roi_ablation_baseline",
    f"run_{RUN}_roi_circling_no_text",
    f"run_{RUN}_roi_circling_no_text_layout_position",
    f"run_{RUN}_roi_circling_text",
    f"run_{RUN}_roi_circling_text_layout_position",
    f"run_{RUN}_no_roi_circling_yes_text_layout_position",
    f"run_{RUN}_no_roi_circling_no_text_layout_position",
]

RUN_MAP = {
    f"run_{RUN}_roi_ablation_baseline": "Text",
    f"run_{RUN}_roi_circling_text": "Text + Circle",
    f"run_{RUN}_no_roi_circling_yes_text_layout_position": "Text + Layout",
    f"run_{RUN}_roi_circling_text_layout_position": "Text + Circle + Layout",
    f"run_{RUN}_roi_circling_no_text_layout_position": "Circle + Layout",
    f"run_{RUN}_roi_circling_no_text": "Circle",
    f"run_{RUN}_no_roi_circling_no_text_layout_position": "Layout",
}
RUN_INDEX = {run_name: idx for idx, run_name in enumerate(RUN_MAP.keys())}

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


def load_metadata_map(metadata_path: Path) -> dict[str, dict]:
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    return {str(item["id"]): item for item in metadata if "id" in item}


def format_family_name(raw_family: str) -> str:
    style = FAMILY_STYLE.get(raw_family)
    if style is None:
        return str(raw_family)
    return str(style["label"])


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
    full_df: pd.DataFrame, metadata_map: dict[str, dict]
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
            ]
        )

    model_sub_obj = (
        full_df.groupby(
            ["run_name", "model_id", "sub_category", count_col], observed=True
        )
        .agg(
            model_accuracy=("is_correct", _accuracy_percent),
            total_questions=("is_correct", _count_valid),
        )
        .reset_index()
    )

    grouped = (
        model_sub_obj.groupby(["run_name", "model_id", count_col], observed=True)
        .agg(
            macro_accuracy=("model_accuracy", "mean"),
            total_questions=("total_questions", "sum"),
        )
        .reset_index()
    )
    if count_col != "object_count":
        grouped = grouped.rename(columns={count_col: "object_count"})

    grouped["model_family"] = grouped["model_id"].map(
        lambda mid: _family_for_model(mid, metadata_map)
    )
    grouped["family_label"] = grouped["model_family"].map(format_family_name)
    grouped["object_count"] = pd.to_numeric(grouped["object_count"], errors="coerce")
    grouped = grouped.dropna(subset=["object_count"]).copy()
    grouped["object_count"] = grouped["object_count"].astype(int)
    return grouped


def aggregate_macro_by_model(
    full_df: pd.DataFrame,
    metadata_map: dict[str, dict],
    min_object_count: int,
) -> pd.DataFrame:
    count_col = _resolve_object_count_column(full_df)
    if count_col is None:
        print("No object count column found; skipping min-object-count filter.")
        filtered = full_df.copy()
    else:
        filtered = full_df[
            pd.to_numeric(full_df[count_col], errors="coerce") >= min_object_count
        ]

    model_sub = (
        filtered.groupby(["run_name", "model_id", "sub_category"], observed=True)
        .agg(
            model_accuracy=("is_correct", _accuracy_percent),
            total_questions=("is_correct", _count_valid),
        )
        .reset_index()
    )

    grouped = (
        model_sub.groupby(["run_name", "model_id"], observed=True)
        .agg(
            macro_accuracy=("model_accuracy", "mean"),
            total_questions=("total_questions", "sum"),
        )
        .reset_index()
    )

    grouped["model_family"] = grouped["model_id"].map(
        lambda mid: _family_for_model(mid, metadata_map)
    )
    grouped["family_label"] = grouped["model_family"].map(format_family_name)
    return grouped


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
    output_path: Path,
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
        )
        subset_plot = subset[subset["family_label"].isin(FAMILY_ORDER)]
        ax.set_title(
            f"Object Count >= {object_count_threshold}",
            fontsize=16,
            fontweight="bold",
            pad=10,
        )
        _apply_reference_style(ax, y_label=(idx % cols == 0))
        y_min, y_max = _compute_y_limits(subset_plot["macro_accuracy"], y_floor=15.0)
        ax.set_ylim(y_min, y_max)

        for family_label in FAMILY_ORDER:
            family_df = subset[subset["family_label"] == family_label]
            if family_df.empty:
                continue

            run_df = (
                family_df.groupby("run_name", observed=True)["macro_accuracy"]
                .mean()
                .reset_index()
            )
            run_df = run_df[run_df["run_name"].isin(RUN_INDEX)].copy()
            if run_df.empty:
                continue

            run_df["run_idx"] = run_df["run_name"].map(RUN_INDEX)
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

        ax.set_xticks(list(range(len(RUN_MAP))))
        ax.set_xticklabels(
            list(RUN_MAP.values()),
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


def plot_family_scatter(grouped_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 5.7))
    fig.patch.set_facecolor(PLOT_BG)
    _apply_reference_style(ax, y_label=True)

    grouped_df_plot = grouped_df[grouped_df["family_label"].isin(FAMILY_ORDER)]
    ax.set_ylim(15.0, 34.0)

    for family_label in FAMILY_ORDER:
        family_df = grouped_df_plot[grouped_df_plot["family_label"] == family_label]
        if family_df.empty:
            continue

        run_df = (
            family_df.groupby("run_name", observed=True)["macro_accuracy"]
            .mean()
            .reset_index()
        )
        run_df = run_df[run_df["run_name"].isin(RUN_INDEX)].copy()
        if run_df.empty:
            continue

        run_df["run_idx"] = run_df["run_name"].map(RUN_INDEX)
        run_df = run_df.sort_values("run_idx")

        ax.scatter(
            run_df["run_idx"],
            run_df["macro_accuracy"],
            s=270,
            marker=FAMILY_MARKERS.get(family_label, "o"),
            color=FAMILY_COLORS.get(family_label, "#4C72B0"),
            label=family_label,
            alpha=0.95,
            edgecolors="none",
        )

    ax.set_xticks(list(range(len(RUN_MAP))))
    ax.set_xticklabels(
        list(RUN_MAP.values()),
        rotation=25,
        ha="right",
        fontsize=17,
        fontweight="bold",
    )
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run ablation analysis from test_ablation copy.ipynb as a regular Python "
            "script with formatted family labels and scatter plots."
        )
    )
    parser.add_argument("--base-path", type=Path, default=DEFAULT_BASE_PATH)
    parser.add_argument("--metadata-path", type=Path, default=Path("utils/metadata.json"))
    parser.add_argument("--runs", nargs="*", default=DEFAULT_RUNS)
    parser.add_argument("--min-object-count", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("output/test_ablation_copy"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_map = load_metadata_map(args.metadata_path)
    full_df = collect_runs(args.base_path, args.runs)

    object_count_df = aggregate_by_object_count(full_df, metadata_map)
    macro_df = aggregate_macro_by_model(
        full_df, metadata_map, min_object_count=args.min_object_count
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    object_count_csv = args.output_dir / "macro_by_model_object_count.csv"
    macro_csv = args.output_dir / "macro_by_model.csv"
    family_csv = args.output_dir / "family_macro_by_run.csv"

    object_count_df.to_csv(object_count_csv, index=False)
    macro_df.to_csv(macro_csv, index=False)
    family_summary = (
        macro_df.groupby(["run_name", "family_label"], observed=True)["macro_accuracy"]
        .mean()
        .reset_index()
    )
    family_summary.to_csv(family_csv, index=False)

    print(f"Saved CSV: {object_count_csv}")
    print(f"Saved CSV: {macro_csv}")
    print(f"Saved CSV: {family_csv}")

    plot_object_count_scatter(
        full_df,
        metadata_map,
        args.output_dir / "macro_accuracy_by_object_count_scatter.png",
    )
    plot_family_scatter(macro_df, args.output_dir / "macro_accuracy_by_family_scatter.png")


if __name__ == "__main__":
    main()
