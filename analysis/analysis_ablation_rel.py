from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MultipleLocator

from utils import (
    utils_read,
    utils_mapping,
    utils_graph
)


DEFAULT_BASE_PATH = Path(
    # "/Users/sebastiancavada/Desktop/tmp_paris/tiny_vqa_creation/output"
    "/data0/sebastian.cavada/compositional-physics/tiny_vqa_creation/output"
)

# def load_metadata_map(metadata_path: Path) -> dict[str, dict]:
#     with metadata_path.open("r", encoding="utf-8") as f:
#         metadata = json.load(f)
#     return {str(item["id"]): item for item in metadata if "id" in item}


# def format_family_name(raw_family: str) -> str:
#     style = FAMILY_STYLE.get(raw_family)
#     if style is None:
#         return str(raw_family)
#     return str(style["label"])


# def _extract_object_count(value: object) -> float:
#     if value is None:
#         return float("nan")
#     if isinstance(value, (list, tuple, set, np.ndarray)):
#         text = " ".join(str(v) for v in value)
#     else:
#         try:
#             if pd.isna(value):
#                 return float("nan")
#         except (TypeError, ValueError):
#             pass
#         text = str(value)

#     match = OBJECT_COUNT_PATTERN.search(text)
#     if not match:
#         return float("nan")
#     try:
#         return float(int(match.group(1)))
#     except (TypeError, ValueError):
#         return float("nan")


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


# def build_eval_df(base_path: Path, run_name: str) -> pd.DataFrame:
#     results_dir = base_path / run_name / f"results_{run_name}_sanitized"
#     if not results_dir.exists():
#         raise FileNotFoundError(
#             f"Missing sanitized results directory for {run_name}: {results_dir}"
#         )

#     model_cols = sorted(
#         p.stem.replace("_val", "") for p in results_dir.glob("*_val.json")
#     )
#     if not model_cols:
#         raise FileNotFoundError(
#             f"No model result files found in sanitized directory: {results_dir}"
#         )

#     try:
#         df = load_results(
#             base_path,
#             run_name,
#             merge_model_answers=True,
#             model_answers_wide=True,
#             model_results_dir=results_dir,
#             cache=True,
#             add_sim_metadata=True,
#         )
#     except FileNotFoundError as exc:
#         print(
#             f"Metadata load failed for {run_name} ({exc}). "
#             "Retrying without simulation metadata."
#         )
#         df = load_results(
#             base_path,
#             run_name,
#             merge_model_answers=True,
#             model_answers_wide=True,
#             model_results_dir=results_dir,
#             cache=False,
#             add_sim_metadata=False,
#         )
#     df = _ensure_object_count_column(df)

#     model_cols = [col for col in model_cols if col in df.columns]
#     if not model_cols:
#         raise ValueError(
#             f"No matching model columns after loading results for {run_name}."
#         )

#     df["answer"] = df["answer"].apply(
#         lambda answer: _sanitize_answer(answer, max_prefix_chars=None)
#     )

#     id_cols = [
#         col
#         for col in [
#             "idx",
#             "question_id",
#             "category",
#             "sub_category",
#             "num_objects",
#             "object_count",
#             "answer",
#             "mode_test",
#             "mode_val",
#             "mode",
#         ]
#         if col in df.columns
#     ]

#     eval_df = df.melt(
#         id_vars=id_cols,
#         value_vars=model_cols,
#         var_name="model_id",
#         value_name="model_answer",
#     )

#     valid = eval_df["model_answer"].notna() & eval_df["answer"].notna()
#     eval_df["is_correct"] = pd.NA
#     eval_df.loc[valid, "is_correct"] = (
#         eval_df.loc[valid, "model_answer"] == eval_df.loc[valid, "answer"]
#     )

#     return eval_df


def collect_runs(ablation_runs: list[str], base_path: Path, vqa_set: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for abl_run in ablation_runs:
        print(f"Processing run: {abl_run}")
        run_df = utils_read.build_eval_df(abl_run, base_path, vqa_set=vqa_set)
        if run_df.empty:
            raise ValueError(f"Run has no rows after loading: {abl_run}")
        run_df["run_name"] = abl_run
        frames.append(run_df)

    if not frames:
        raise ValueError("No data available for the selected runs.")

    return pd.concat(frames, ignore_index=True)


# def _accuracy_percent(series: pd.Series) -> float:
#     numeric = pd.to_numeric(series, errors="coerce").dropna()
#     if numeric.empty:
#         return float("nan")
#     return float(numeric.mean() * 100.0)


# def _count_valid(series: pd.Series) -> int:
#     numeric = pd.to_numeric(series, errors="coerce").dropna()
#     return int(numeric.shape[0])


# def _family_for_model(model_id: str, metadata_map: dict[str, dict]) -> str:
#     item = metadata_map.get(str(model_id), {})
#     return str(item.get("family", "Unknown"))


# def aggregate_by_object_count(
#     full_df: pd.DataFrame, metadata_map: dict[str, dict]
# ) -> pd.DataFrame:
#     count_col = _resolve_object_count_column(full_df)
#     if count_col is None:
#         print("Skipping object-count aggregation: no object count column found.")
#         return pd.DataFrame(
#             columns=[
#                 "run_name",
#                 "model_id",
#                 "object_count",
#                 "macro_accuracy",
#                 "total_questions",
#                 "model_family",
#                 "family_label",
#             ]
#         )

#     model_sub_obj = (
#         full_df.groupby(
#             ["run_name", "model_id", "sub_category", count_col], observed=True
#         )
#         .agg(
#             model_accuracy=("is_correct", _accuracy_percent),
#             total_questions=("is_correct", _count_valid),
#         )
#         .reset_index()
#     )

#     grouped = (
#         model_sub_obj.groupby(["run_name", "model_id", count_col], observed=True)
#         .agg(
#             macro_accuracy=("model_accuracy", "mean"),
#             total_questions=("total_questions", "sum"),
#         )
#         .reset_index()
#     )
#     if count_col != "object_count":
#         grouped = grouped.rename(columns={count_col: "object_count"})

#     grouped["model_family"] = grouped["model_id"].map(
#         lambda mid: _family_for_model(mid, metadata_map)
#     )
#     grouped["family_label"] = grouped["model_family"].map(format_family_name)
#     grouped["object_count"] = pd.to_numeric(grouped["object_count"], errors="coerce")
#     grouped = grouped.dropna(subset=["object_count"]).copy()
#     grouped["object_count"] = grouped["object_count"].astype(int)
#     return grouped


# def aggregate_macro_by_model(
#     full_df: pd.DataFrame,
#     metadata_map: dict[str, dict],
#     min_object_count: int,
# ) -> pd.DataFrame:
#     count_col = _resolve_object_count_column(full_df)
#     if count_col is None:
#         print("No object count column found; skipping min-object-count filter.")
#         filtered = full_df.copy()
#     else:
#         filtered = full_df[
#             pd.to_numeric(full_df[count_col], errors="coerce") >= min_object_count
#         ]

#     model_sub = (
#         filtered.groupby(["run_name", "model_id", "sub_category"], observed=True)
#         .agg(
#             model_accuracy=("is_correct", _accuracy_percent),
#             total_questions=("is_correct", _count_valid),
#         )
#         .reset_index()
#     )

#     grouped = (
#         model_sub.groupby(["run_name", "model_id"], observed=True)
#         .agg(
#             macro_accuracy=("model_accuracy", "mean"),
#             total_questions=("total_questions", "sum"),
#         )
#         .reset_index()
#     )

#     grouped["model_family"] = grouped["model_id"].map(
#         lambda mid: _family_for_model(mid, metadata_map)
#     )
#     grouped["family_label"] = grouped["model_family"].map(format_family_name)
#     return grouped


# def _legend_handles_labels(ax_list: list[plt.Axes]) -> tuple[list, list[str]]:
#     handle_map = {}
#     for ax in ax_list:
#         handles, labels = ax.get_legend_handles_labels()
#         for handle, label in zip(handles, labels):
#             if label not in handle_map:
#                 handle_map[label] = handle
#     labels = list(handle_map.keys())
#     handles = [handle_map[label] for label in labels]
#     return handles, labels


# def _compute_y_limits(
#     values: pd.Series,
#     margin_ratio: float = 0.08,
#     y_floor: float | None = None,
# ) -> tuple[float, float]:
#     numeric = pd.to_numeric(values, errors="coerce").dropna()
#     if numeric.empty:
#         return 0.0, 1.0

#     y_min = float(numeric.min())
#     y_max = float(numeric.max())
#     span = y_max - y_min
#     if span <= 0.0:
#         pad = max(0.5, abs(y_min) * 0.05)
#     else:
#         pad = span * margin_ratio
#     y_low = y_min - pad
#     y_high = y_max + pad
#     if y_floor is not None:
#         y_low = max(float(y_floor), y_low)
#     return y_low, y_high


# def _get_object_count_thresholds(full_df: pd.DataFrame) -> list[int]:
#     count_col = _resolve_object_count_column(full_df)
#     if count_col is None:
#         return []
#     numeric = pd.to_numeric(full_df[count_col], errors="coerce").dropna()
#     if numeric.empty:
#         return []
#     return sorted(pd.unique(numeric.astype(int)))


# def _get_plot_run_index(baseline_run_name: str) -> dict[str, int]:
#     run_names = [run_name for run_name in ABLATIONS_RUNS.keys() if run_name != baseline_run_name]
#     if not run_names:
#         raise ValueError("No non-baseline runs available for plotting.")
#     return {run_name: idx for idx, run_name in enumerate(run_names)}


# def add_accuracy_change_pp(
#     df: pd.DataFrame,
#     *,
#     value_col: str,
#     baseline_run_name: str,
#     group_cols: list[str],
#     baseline_col: str = "baseline_accuracy",
#     change_col: str = "accuracy_change_pp",
# ) -> pd.DataFrame:
#     if df.empty:
#         return df.copy()

#     missing_cols = [col for col in group_cols + ["run_name", value_col] if col not in df.columns]
#     if missing_cols:
#         raise ValueError(f"Missing columns for relative change: {missing_cols}")

#     baseline = (
#         df[df["run_name"] == baseline_run_name][group_cols + [value_col]]
#         .rename(columns={value_col: baseline_col})
#         .copy()
#     )
#     if baseline.empty:
#         raise ValueError(
#             f"No baseline rows found for run '{baseline_run_name}'. Check --runs."
#         )

#     merged = df.merge(baseline, on=group_cols, how="left")
#     baseline_vals = pd.to_numeric(merged[baseline_col], errors="coerce")
#     current_vals = pd.to_numeric(merged[value_col], errors="coerce")
#     merged[change_col] = np.where(
#         baseline_vals.notna(),
#         current_vals - baseline_vals,
#         np.nan,
#     )
#     return merged


# def plot_object_count_scatter(
#     full_df: pd.DataFrame,
#     metadata_map: dict[str, dict],
#     output_path: Path,
#     *,
#     baseline_run_name: str,
# ) -> None:
#     # object_counts = _get_object_count_thresholds(full_df)
#     # if not object_counts:
#     #     print("Skipping object-count plot: no object-count rows available.")
#     #     return
#     # plot_run_index = _get_plot_run_index(baseline_run_name)

#     cols = 5
#     rows = math.ceil(len(object_counts) / cols)
#     fig, axes = plt.subplots(
#         nrows=rows,
#         ncols=cols,
#         figsize=(5.8 * cols, 5.2 * rows),
#         sharex=True,
#         sharey=False,
#     )
#     fig.patch.set_facecolor(PLOT_BG)
#     axes_flat = np.array(axes).reshape(-1)

#     for idx, ax in enumerate(axes_flat):
#         if idx >= len(object_counts):
#             ax.axis("off")
#             continue

#         object_count_threshold = int(object_counts[idx])
#         subset = aggregate_macro_by_model(
#             full_df,
#             metadata_map,
#             min_object_count=object_count_threshold,
#         )
#         subset = add_accuracy_change_pp(
#             subset,
#             value_col="macro_accuracy",
#             baseline_run_name=baseline_run_name,
#             group_cols=["model_id"],
#         )
#         subset_plot = subset[subset["family_label"].isin(FAMILY_ORDER)]
#         ax.set_title(
#             f"Object Count >= {object_count_threshold}",
#             fontsize=16,
#             fontweight="bold",
#             pad=10,
#         )
#         _apply_reference_style(
#             ax,
#             y_label_text="Accuracy Change (pp)" if (idx % cols == 0) else None,
#         )
#         y_min, y_max = _compute_y_limits(subset_plot["accuracy_change_pp"])
#         ax.set_ylim(y_min, y_max)

#         for family_label in FAMILY_ORDER:
#             family_df = subset[subset["family_label"] == family_label]
#             if family_df.empty:
#                 continue

#             run_df = (
#                 family_df.groupby("run_name", observed=True)["accuracy_change_pp"]
#                 .mean()
#                 .reset_index()
#             )
#             run_df = run_df[run_df["run_name"].isin(plot_run_index)].copy()
#             if run_df.empty:
#                 continue

#             run_df["run_idx"] = run_df["run_name"].map(plot_run_index)
#             run_df = run_df.sort_values("run_idx")
#             run_df["run_idx_jitter"] = run_df["run_idx"] + JITTER_OFFSETS.get(
#                 family_label, 0.0
#             )
#             marker = FAMILY_MARKERS.get(family_label, "o")
#             color = FAMILY_COLORS.get(family_label, "#4C72B0")

#             ax.scatter(
#                 run_df["run_idx_jitter"],
#                 run_df["accuracy_change_pp"],
#                 label=family_label,
#                 s=260,
#                 alpha=0.95,
#                 marker=marker,
#                 color=color,
#                 edgecolors="none",
#             )

#         plot_run_names = list(plot_run_index.keys())
#         ax.set_xticks(list(range(len(plot_run_names))))
#         ax.set_xticklabels(
#             [ABLATIONS_LABELS[run_name] for run_name in plot_run_names],
#             rotation=25,
#             ha="right",
#             fontsize=14,
#             fontweight="bold",
#         )

#     used_axes = [ax for ax in axes_flat[: len(object_counts)]]
#     handles, labels = _legend_handles_labels(used_axes)
#     if handles:
#         fig.legend(
#             handles,
#             labels,
#             loc="lower center",
#             bbox_to_anchor=(0.5, 0.02),
#             ncol=len(labels),
#             title="Model Family",
#             fontsize=13,
#             title_fontsize=14,
#             frameon=True,
#             facecolor=PLOT_BG,
#             edgecolor="#4f4f4f",
#             columnspacing=1.2,
#             handletextpad=0.5,
#             borderpad=0.6,
#         )
#         fig.tight_layout(rect=(0.0, 0.18, 1.0, 1.0))
#     else:
#         fig.tight_layout()
#     output_path.parent.mkdir(parents=True, exist_ok=True)
#     fig.savefig(output_path, dpi=200, bbox_inches="tight")
#     plt.close(fig)
#     print(f"Saved object-count scatter plot: {output_path}")


def plot_ablation_scatter(
    runs_df: pd.DataFrame, 
    output_dir: Path, 
    ablations_runs: list[str],
    ablations_labels: list[str],
    accuracy_mode: str="baseline_change",  # baseline_change, absolute, baseline_rel_change
    filename: str="ablation_rel.png",
    *, 
    plot_baseline: bool=False,
    baseline_name: str="roi_ablation_baseline",
    group_by: str = "model_id",
    metadata_path: str | Path | None = "utils/metadata.json",
    legend_mode: list[str] = ["all"]  # improved, worsened, all
) -> None:
    plot_df = runs_df.copy()
    plot_df["accuracy"] *= 100
    
    plot_df = utils_read.macro_accuracy(plot_df, level="model_id", group_by=["run_name"])

    agg_df = (
        plot_df.groupby([group_by, "run_name"], observed=True)["accuracy"]
        .agg(["mean", "min", "max", "std"])
        .reset_index()
    )

    baseline_runname = ablations_runs.get(baseline_name)
    if accuracy_mode == "absolute":
        agg_df["accuracy_plot"] = agg_df["mean"]
    elif accuracy_mode == "baseline_change":
        agg_df["accuracy_plot"] = agg_df.apply(
            lambda row: row["mean"] - agg_df[
                (agg_df[group_by] == row[group_by]) & (agg_df["run_name"] == baseline_runname)
            ]["mean"].values[0],
            axis=1,
        )
    elif accuracy_mode == "baseline_rel_change":
        agg_df["accuracy_plot"] = agg_df.apply(
            lambda row: (row["mean"] - agg_df[
                (agg_df[group_by] == row[group_by]) & (agg_df["run_name"] == baseline_runname)
            ]["mean"].values[0]) / agg_df[
                (agg_df[group_by] == row[group_by]) & (agg_df["run_name"] == baseline_runname)
            ]["mean"].values[0] * 100,
            axis=1,
        )

    # Extract all run tags (eg, "circle", "layout", "name") from the ablation labels
    tag_extract_fct = lambda v: re.sub(r"[^A-Za-z0-9+\-.]+", "", v).lower().split("+")
    tags = np.hstack([tag_extract_fct(v) for k,v in ablations_labels.items()])
    tags = np.unique(tags)
    print("Found run tags:", tags)
    
    def run_has_tag(run:str, tag:str):
        run_short = [k for k, v in ablations_runs.items() if v == run][0]
        label = ablations_labels[run_short]
        return tag in tag_extract_fct(label)
    
    for tag in tags:
        agg_df[f"run_has_{tag}"] = agg_df["run_name"].apply(lambda run: run_has_tag(run, tag))
    
    # Compute change for each group
    groups_tags_change = {}
    for group in agg_df[group_by].unique():
        group_mask = agg_df[group_by] == group
        baseline_mask = (agg_df["run_name"] == baseline_runname) & group_mask
        assert baseline_mask.sum() == 1
        baseline_acc = agg_df.loc[baseline_mask, "mean"].values[0]

        tags_improve = []
        tags_worsen = []
        for t in tags:
            if t == "name":
                continue  # skip "name" tag for legend labeling
            t_mask = agg_df[f"run_has_{t}"] & group_mask
            t_mask = t_mask & agg_df[f"run_has_name"]  # exclude runs with no "name" tag
            if any( ((agg_df[t_mask]["mean"] - baseline_acc) / baseline_acc) > 0.05 ):
                tags_improve.append(t)
            if any( ((agg_df[t_mask]["mean"] - baseline_acc) / baseline_acc) < -0.05 ):
                tags_worsen.append(t)

        groups_tags_change[group] = {
            "improve": tags_improve,
            "worsen": tags_worsen,
        }

    if accuracy_mode == "absolute":
        groups_df = (
            agg_df[agg_df["run_name"] == baseline_runname].groupby(group_by, observed=True)["mean"]
            .mean()
            .sort_values(ascending=False)
            .reset_index(name="mean_accuracy")
        )

        groups = groups_df[group_by].tolist()  # sorted group list
        for group in groups:
            group_mask = agg_df[group_by] == group
            baseline_mask = (agg_df["run_name"] == baseline_runname) & group_mask
            name_mask = agg_df["run_has_name"] & group_mask
            circle_mask = agg_df["run_has_circle"] & group_mask
            layout_mask = agg_df["run_has_layout"] & group_mask

            name_acc = agg_df.loc[name_mask, "mean"].mean()
            circle_acc = agg_df.loc[circle_mask, "mean"].mean()
            layout_acc = agg_df.loc[layout_mask, "mean"].mean()
            
            circle_only = agg_df[agg_df["run_name"].isin([ablations_runs["roi_circling_no_text"]]) & group_mask]
            circle_only_acc = circle_only["mean"].values[0]

            baseline_acc = agg_df.loc[baseline_mask, "mean"].mean()
            change_fct = lambda acc: f"{(acc - baseline_acc):.1f}" if acc <= baseline_acc else f"+{(acc - baseline_acc):.1f}"

            circle_change = circle_only_acc - baseline_acc
            print(f"{group}:        Baseline: {baseline_acc:.2f}%, Name acc: {change_fct(name_acc)}%, Circle acc: {change_fct(circle_acc)}%, Layout acc: {change_fct(layout_acc)}%", f"circle change: {circle_change:.2f}%", ("*** IMPROVED ***" if circle_change > 1 else ("*** WORSENED ***" if circle_change < -1 else "not improved")))

            groups_df["baseline"] = baseline_acc
            groups_df["name_acc"] = name_acc
            groups_df["circle_acc"] = circle_acc
            groups_df["layout_acc"] = layout_acc
        pass
    
    # Keep only ablations existing in the agg_df
    ablations_runs = {abl: run for abl, run in ablations_runs.items() if run in agg_df["run_name"].unique()}
    ablations_labels = {abl: name for abl, name in ablations_labels.items() if abl in ablations_runs}

    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    run_idx = {run: idx for idx, (abl, run) in enumerate(ablations_runs.items())}
    agg_df["run_idx"] = agg_df["run_name"].map(run_idx)

    # Build model style
    model_style, family_map = utils_mapping._build_model_style(
        metadata_path,
        group_by=group_by
    )
    rng = np.random.default_rng(42)
    
    for group_name, df_m in agg_df.groupby(group_by):
        # Remove baseline points if not plotting baseline
        # if not plot_baseline:
        #     df_m = df_m[df_m["run_name"] != baseline_runname]
        
        x_vals = df_m["run_idx"].to_numpy()
        y_vals = df_m["accuracy_plot"].to_numpy()
        
        if x_vals.size == 0:
            continue

        jitter = rng.uniform(-0.20, 0.20, size=x_vals.size)
        x_jittered = x_vals + jitter
        color, marker, size = model_style[group_name]
        
        if groups_tags_change[group_name]["improve"]:
            alpha = 1.0
            zorder = 5
        else:
            alpha = 0.15
            zorder = 4

        # Plot scatter points
        ax.scatter(
            x_jittered,
            y_vals,
            color=color,
            s=size**2,
            alpha=alpha,
            edgecolor="white",
            linewidth=1,
            marker=marker,
            zorder=zorder
        )

    ax.set_xticks(list(range(len(ablations_labels))))
    ax.set_xticklabels(
        [name for abl, name in ablations_labels.items()],
        ha="center",
    )
    ax.set_xlabel("", fontsize=1)

    if accuracy_mode == "absolute":
        ax.set_ylabel("Accuracy (%)")
        ticks_step = 5.0
    elif accuracy_mode == "baseline_change":
        ax.set_ylabel("Accuracy wrt baseline (%)")
        ticks_step = 2.0
    elif accuracy_mode == "baseline_rel_change":
        ax.set_ylabel("Accuracy wrt baseline\n(rel. %)")
        ticks_step = 5.0
        
    legend_handles, legend_labels, legend_groups, title_str = utils_graph._build_group_legend_items(
        plot_df,
        group_by=group_by,
        metadata_path=metadata_path
    )

    improved = [] 
    worsen = [] 
    for i, (handle, label, group) in enumerate(zip(legend_handles, legend_labels, legend_groups)):
        tags_improve = groups_tags_change[group]["improve"]
        tags_worsen = groups_tags_change[group]["worsen"]

        if tags_improve:
            label += " (" + ", ".join([t.capitalize() for t in tags_improve])+")"
            # label += " +" + ",".join([f"+{t[0].capitalize()}" for t in tags_improve])
            improved.append(i)
        elif tags_worsen:
            label += " (" + ", ".join([t.capitalize() for t in tags_worsen])+")"
            # label += " -" + ",".join([f"-{t[0].capitalize()}" for t in tags_worsen])+")"
            worsen.append(i)
        
        legend_labels[i] = label

    groups = {}
    if "improved" in legend_mode or "all" in legend_mode:
        groups["Models improved"] = improved
    if "worsened" in legend_mode or "all" in legend_mode:
        groups["Models worsened"] = worsen
    
    l_pos = (1.05, 1.0)
    legend_artists = []
    for title, items in groups.items():
        group_handles = [legend_handles[i] for i in items]
        group_labels  = [legend_labels[i] for i in items]
        print("Group:", title, "Items:", group_labels)

        leg = ax.legend(
            group_handles, group_labels,
            title=title,
            loc="upper left",
            bbox_to_anchor=l_pos,
            frameon=True,
            borderaxespad=0.0,
            fontsize=8, 
            title_fontsize=9, 
            markerscale=0.7
        )
        ax.add_artist(leg)
        legend_artists.append(leg)
        # l_pos = (l_pos[0], l_pos[1] - 0.1 - 0.1 * len(group_handles))  # vertical spacing between groups
        l_pos = (l_pos[0] + 1.1, l_pos[1])  # vertical spacing between groups
    
    # ax.legend(legend_handles, 
    #           legend_labels, 
    #           title=title_str, 
    #           bbox_to_anchor=(1.05, 1), 
    #           loc='upper left', 
    #           fontsize=8, 
    #           title_fontsize=9, 
    #           markerscale=0.7)

    utils_graph.paperformat(ax, ticks_step=ticks_step)
    if accuracy_mode in ["baseline_change", "baseline_rel_change"]:
        ax.axhline(0, color="#000000", linestyle="-", linewidth=1.5, zorder=3)

        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda y, _: f"{'+' if y > 0 else ''}{y:.0f}")
        )
    
    for ticklabel in ax.get_xticklabels():
        ticklabel.set_fontsize(ticklabel.get_fontsize()*0.50)
    
    fname = Path(output_dir) / filename
    fname.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(fname, 
                dpi=300, 
                bbox_inches="tight",
                bbox_extra_artists=legend_artists
                )
    plt.close(fig)

    print(f"Saved plot to: {fname}")


def main() -> None:
    ablations_labels = {
        f"roi_ablation_baseline": "Name",
        f"roi_circling_text": "Name\n+Circle",
        f"no_roi_circling_yes_text_layout_position": "Name\n+Layout",
        # f"roi_circling_text_layout_position": "Name\n+Circle\n+Layout",
        # f"roi_circling_no_text_layout_position": "Circle\n+Layout",
        f"roi_circling_no_text": "Circle",
        f"no_roi_circling_no_text_layout_position": "Layout",
        # f"ablation_physics_duration_text": "Name\n+Duration",
        # f"ablation_physics_mass_text": "Name\n+Mass",
    }

    parser = argparse.ArgumentParser(
        description=(
            "Run ablation analysis with accuracy change in percentage points "
            "based on the Text baseline run."
        )
    )
    parser.add_argument("--run-name", default="run_28")
    parser.add_argument("--base-path", type=Path, default="../output")
    parser.add_argument("--metadata-path", type=Path, default=Path("utils/metadata.json"))
    parser.add_argument("--ablations-runs", nargs="*", default=list(ablations_labels.keys()))
    parser.add_argument("--ablations-names", nargs="*", default=list(ablations_labels.values()))
    parser.add_argument(
        "--vqa-set",
        default="10K",
        help="VQA set to use for ablations (e.g., 10K, 30K, karo_5K).",
    )
    # parser.add_argument("--min-object-count", type=int, default=5)
    parser.add_argument("--baseline-run", type=str, default=None)
    parser.add_argument(
        "--family",
        default=None,
        help="Family to filter",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
    )
    args = parser.parse_args()

    run_name = args.run_name
    
    if args.ablations_runs or args.ablations_names:
        if len(args.ablations_runs) != len(args.ablations_names):
            raise ValueError("Number of ablation runs and names must match.")
        ablations_labels = dict(zip(args.ablations_runs, args.ablations_names))

    ablations_runs = {abl: run_name + "_" + abl for abl in args.ablations_runs}
    eval_df = collect_runs(ablations_runs.values(), args.base_path, args.vqa_set)

    baseline_name = args.baseline_run or list(ablations_labels.keys())[0]

    output_dir = args.output_dir / (run_name + "_ablations") / args.vqa_set
    cur_output_dir = output_dir

    if args.family is not None:
        print("Filtering to family:", args.family)
        eval_df = eval_df[eval_df['model_family'] == args.family]
        assert eval_df['idx'].nunique() > 0, f"No entries found for family {args.family} in eval_df after filtering. Check if family name is correct and if there are entries for that family."

        # Use subdirectory for family-specific results
        cur_output_dir = output_dir / f"family_{args.family}"
    cur_output_dir.mkdir(parents=True, exist_ok=True)

    # object_count_df = aggregate_by_object_count(full_df, metadata_map)
    # object_count_change_df = add_accuracy_change_pp(
    #     object_count_df,
    #     value_col="macro_accuracy",
    #     baseline_run_name=baseline_run_name,
    #     group_cols=["model_id", "object_count"],
    # )
    # macro_df = aggregate_macro_by_model(
    #     full_df, metadata_map, min_object_count=args.min_object_count
    # )
    # macro_change_df = add_accuracy_change_pp(
    #     macro_df,
    #     value_col="macro_accuracy",
    #     baseline_run_name=baseline_name,
    #     group_cols=["model_id"],
    # )

    # args.output_dir.mkdir(parents=True, exist_ok=True)
    # object_count_csv = args.output_dir / "macro_by_model_object_count_change_pp.csv"
    # macro_csv = args.output_dir / "macro_by_model_change_pp.csv"
    # family_csv = args.output_dir / "family_macro_by_run_change_pp.csv"

    # object_count_change_df.to_csv(object_count_csv, index=False)
    # macro_change_df.to_csv(macro_csv, index=False)
    # family_summary = (
    #     macro_change_df.groupby(["run_name", "family_label"], observed=True)[
    #         "accuracy_change_pp"
    #     ]
    #     .mean()
    #     .reset_index()
    # )
    # family_summary.to_csv(family_csv, index=False)

    # print(f"Saved CSV: {object_count_csv}")
    # print(f"Saved CSV: {macro_csv}")
    # print(f"Saved CSV: {family_csv}")

    # plot_object_count_scatter(
    #     full_df,
    #     metadata_map,
    #     args.output_dir / "accuracy_change_pp_by_object_count_scatter.png",
    #     baseline_name=baseline_name,
    # )
    # for group in utils_read.GROUPINGS:
    for group in ["model"]:
        cur_df, group_by = utils_read.apply_group(eval_df, group)
        
        print(f"Processing: grouping by {group_by}: with {len(cur_df)} entries")
        # for accuracy_mode in ["baseline_change", "baseline_rel_change", "absolute"]:
        for accuracy_mode in ["absolute"]:
            plot_ablation_scatter(
                cur_df,
                cur_output_dir,
                group_by=group_by,
                accuracy_mode=accuracy_mode,
                filename=f"ablation_{accuracy_mode}_{group}.png",
                plot_baseline=accuracy_mode == "absolute",
                ablations_runs=ablations_runs,
                ablations_labels=ablations_labels,
                baseline_name=baseline_name,
                legend_mode=["all"]  # or all
            )


if __name__ == "__main__":
    main()