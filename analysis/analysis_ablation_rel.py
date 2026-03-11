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



def plot_ablation(
    runs_df: pd.DataFrame, 
    output_dir: Path, 
    ablations_runs: list[str],
    ablations_tags: list[str],
    accuracy_mode: str="baseline_change",  # baseline_change, absolute, baseline_rel_change
    filename: str="ablation_rel.png",
    *, 
    plot_baseline: bool=False,
    baseline_name: str="roi_ablation_baseline",
    group_by: str = "model_id",
    metadata_path: str | Path | None = "utils/metadata.json",
    legend: list[str]|bool = ["all"],  # improved, worsened, all
    change_rel_threshold: int = 5  # 5% relative change threshold for improvement/worsening
) -> None:
    print("LEEEGEND:", legend)

    plot_df = runs_df.copy()
    plot_df["accuracy"] *= 100
    
    plot_df = utils_read.macro_accuracy(plot_df, level="model_id", group_by=["run_name"])

    agg_df = (
        plot_df.groupby([group_by, "run_name"], observed=True)["accuracy"]
        .agg(["mean", "min", "max", "std"])
        .reset_index()
    )

    baseline_runname = ablations_runs.get(baseline_name)
    agg_df["accuracy_change"] = agg_df.apply(
        lambda row: row["mean"] - agg_df[
            (agg_df[group_by] == row[group_by]) & (agg_df["run_name"] == baseline_runname)
        ]["mean"].values[0],
        axis=1,
    )
    agg_df["accuracy_rel_change"] = agg_df.apply(
        lambda row: (row["mean"] - agg_df[
            (agg_df[group_by] == row[group_by]) & (agg_df["run_name"] == baseline_runname)
        ]["mean"].values[0]) / agg_df[
            (agg_df[group_by] == row[group_by]) & (agg_df["run_name"] == baseline_runname)
        ]["mean"].values[0] * 100,
        axis=1,
    )

    if accuracy_mode == "absolute":
        agg_df["accuracy_plot"] = agg_df["mean"]
    elif accuracy_mode == "baseline_change":
        agg_df["accuracy_plot"] = agg_df["accuracy_change"]
    elif accuracy_mode == "baseline_rel_change":
        agg_df["accuracy_plot"] = agg_df["accuracy_rel_change"]
    
    # Extract all run tags (eg, "circle", "layout", "name") from the ablation labels
    tags = np.hstack([v for k,v in ablations_tags.items()])
    tags = np.unique(tags)
    print("Found run tags:", tags)
    
    def run_has_tag(run:str, tag:str):
        run_short = [k for k, v in ablations_runs.items() if v == run][0]
        tags = ablations_tags[run_short]
        return tag in tags
    
    for tag in tags:
        agg_df[f"run_tag_{tag.lower()}"] = agg_df["run_name"].apply(lambda run: run_has_tag(run, tag))
    
    # Mark runs as improved or worsened compared to baseline    
    # Keep only ablations existing in the agg_df
    ablations_runs = {abl: run for abl, run in ablations_runs.items() if run in agg_df["run_name"].unique()}
    ablations_tags = {abl: name for abl, name in ablations_tags.items() if abl in ablations_runs}

    sx = 0.2 + len(tags)
    if not plot_baseline:
        sx -= 0.5
    if legend:
        sx += 1.2
    figsize = (sx, 3.1)
    print("figsize:", figsize)
    fig, ax = plt.subplots(figsize=figsize)

    run_idx = {run: idx for idx, (abl, run) in enumerate(ablations_runs.items())}
    agg_df["run_idx"] = agg_df["run_name"].map(run_idx)

    agg_baseline_mask = agg_df["run_name"] == baseline_runname

    # Build model style
    model_style, family_map = utils_mapping._build_model_style(
        metadata_path,
        group_by=group_by
    )
    rng = np.random.default_rng(42)
    
    for group, df_m in agg_df.groupby(group_by):
        # Remove baseline points if not plotting baseline
        if not plot_baseline:
            df_m = df_m[df_m["run_name"] != baseline_runname]
        
        x_vals = df_m["run_idx"].to_numpy()
        y_vals = df_m["accuracy_plot"].to_numpy()
        
        if x_vals.size == 0:
            continue

        jitter = rng.uniform(-0.20, 0.20, size=x_vals.size)
        x_jittered = x_vals + jitter
        color, marker, size, edge = model_style[group]
        
        improve = any((agg_df[agg_df[group_by] == group]["accuracy_rel_change"] > change_rel_threshold) & ~agg_baseline_mask)
        if improve:
            alphas = []
            for _, r in df_m.iterrows():
                if r["run_name"] == baseline_runname:
                    alphas.append(1.0)
                elif r["accuracy_rel_change"] > change_rel_threshold:
                    alphas.append(1.0)
                else:
                    alphas.append(0.1)
            alpha = alphas
            zorder = 5
        else:
            alpha = 0.10
            zorder = 4

        # Plot scatter points
        ax.scatter(
            x_jittered,
            y_vals,
            color=color,
            s=size**2,
            alpha=alpha,
            edgecolor=edge,
            linewidth=1,
            marker=marker,
            zorder=zorder
        )
        # if improve:
        #     ax.plot(
        #         x_vals[x_vals.argsort()],
        #         y_vals[x_vals.argsort()],
        #         color=color,
        #         alpha=0.5,
        #         linewidth=1.5,
        #         zorder=3
        #     )


    def xtick_label(abl_tags):
        if abl_tags == ["Name"]:
            return "Name\n(baseline)"

        label = "\n".join([t for t in abl_tags])
        abl_mask = None
        for tag in tags:
            tag_mask = agg_df[f"run_tag_{tag.lower()}"] == (True if tag in abl_tags else False)
            abl_mask = tag_mask if abl_mask is None else (abl_mask & tag_mask)
        
        return label

    def xtick_stats(abl_tags):
        abl_mask = None
        for tag in tags:
            tag_mask = agg_df[f"run_tag_{tag.lower()}"] == (True if tag in abl_tags else False)
            abl_mask = tag_mask if abl_mask is None else (abl_mask & tag_mask)

        rel_change = agg_df.loc[abl_mask, "accuracy_rel_change"]
        up = int((rel_change > change_rel_threshold).sum())
        down = int((rel_change < -change_rel_threshold).sum())
        flat = int(((rel_change >= -change_rel_threshold) & (rel_change <= change_rel_threshold)).sum())
        return up, down, flat

    xticks = list(range(len(ablations_tags)))
    xtick_items = list(ablations_tags.items())
    if not plot_baseline:
        xticks = [x for x, t in zip(xticks, ablations_tags.values()) if t != ["Name"]]
        xtick_items = [x for x, t in zip(xtick_items, ablations_tags.values()) if t != ["Name"]]

    xtick_labels = [xtick_label(abl_tags) for _, abl_tags in xtick_items]
    xtick_stats_values = [xtick_stats(abl_tags) for _, abl_tags in xtick_items]

    if len(xticks) == 1:
        ax.set_xlim(xticks[0] - 0.7, xticks[0] + 0.7)
    ax.set_xticks(xticks)
    ax.set_xticklabels(
        xtick_labels,
        ha="center",
    )

    stats_fontsize = 6
    for i, (x, (up, down, flat)) in enumerate(zip(xticks, xtick_stats_values)):
        if plot_baseline and x == 0:
            continue # skip baseline stats
        stats_y = -0.08 - 0.04 * xtick_labels[i].count("\n")
        ax.text(
            x - 0.17, stats_y,
            f"↗{up}",
            transform=ax.get_xaxis_transform(),
            ha="right", va="top",
            fontsize=stats_fontsize, color="tab:green"
        )
        ax.text(
            x, stats_y,
            f"≈{flat}",
            transform=ax.get_xaxis_transform(),
            ha="center", va="top",
            fontsize=stats_fontsize, color="#666666"
        )
        ax.text(
            x + 0.15, stats_y,
            f"↘{down}",
            transform=ax.get_xaxis_transform(),
            ha="left", va="top",
            fontsize=stats_fontsize, color="tab:red"
        )
    ax.set_xlabel("", fontsize=1)
    # ax.axvline(0.5, color="#666666", linewidth=1, alpha=0.5, zorder=-1, linestyle='--')

    if runs_df["category"].nunique() == 1:
        cat = runs_df["category"].unique()[0]
        ylabel_color = utils_mapping.mapping_cat_colors[cat]+"CC"
        ylabel = utils_mapping.mapping_cat_short[cat]
    else:
        ylabel = "Overall Accuracy"
        ylabel_color = "black"

    if accuracy_mode == "absolute":
        ylabel += " (%)"
        ticks_step = 5.0
    elif accuracy_mode == "baseline_change":
        ylabel += "\n(change %)"
        ticks_step = 2.0
    elif accuracy_mode == "baseline_rel_change":
        ylabel += "\n(rel. change %)"
        ticks_step = 5.0
    
    ax.set_ylabel(ylabel, color=ylabel_color)

    legend_artists = []
    if legend:
        legend_handles, legend_labels, legend_groups, title_str = utils_graph._build_group_legend_items(
            plot_df,
            group_by=group_by,
            metadata_path=metadata_path
        )

        improved = [] 
        worsen = [] 
        for i, (handle, label, group) in enumerate(zip(legend_handles, legend_labels, legend_groups)):
            group_mask = agg_df[group_by] == group

            tags_improve = []
            tags_worsen = []
            for tag in tags:
                group_tag_mask = group_mask & (agg_df[f"run_tag_{tag.lower()}"] == True)
                if any((agg_df[group_tag_mask]["accuracy_rel_change"] > change_rel_threshold) & ~agg_baseline_mask):
                    tags_improve.append(tag)
                if any((agg_df[group_tag_mask]["accuracy_rel_change"] < -change_rel_threshold) & ~agg_baseline_mask):
                    tags_worsen.append(tag)

            if tags_improve:
                # label += " (" + ", ".join([t for t in tags_improve])+")"
                # label += " +" + ",".join([f"+{t[0].capitalize()}" for t in tags_improve])
                improved.append(i)
            elif tags_worsen:
                # label += " (" + ", ".join([t for t in tags_worsen])+")"
                # label += " -" + ",".join([f"-{t[0].capitalize()}" for t in tags_worsen])+")"
                worsen.append(i)
            
            legend_labels[i] = label

        groups = []
        if "improved" in legend or "all" in legend:
            title = f"Model improved\n($\greater$ {change_rel_threshold}% rel. change)"
            if change_rel_threshold == 0:
                title = "Model improved"
            groups.append((title, improved))
        if "worsened" in legend or "all" in legend:
            title = f"Model worsened\n($\less$ {(-change_rel_threshold)}% rel. change)"
            if change_rel_threshold == 0:
                title = "Model worsened"
            groups.append((title, worsen))
        
        l_pos = (1.05, 1.0)
        for title, items in groups:
            group_handles = [legend_handles[i] for i in items]
            group_labels  = [legend_labels[i] for i in items]
            # print("Group:", title, "Items:", group_labels)

            leg = ax.legend(
                group_handles, group_labels,
                title=title,
                loc="upper left",
                bbox_to_anchor=l_pos,
                frameon=True,
                borderaxespad=0.0,
                fontsize=8, 
                title_fontsize=9, 
                markerscale=0.7,
                ncol=2 if len(group_handles)>20 else 1
            )
            leg.get_title().set_ha("center")
            leg.get_title().set_multialignment("center")
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

    utils_graph.paperformat(ax, figsize=figsize, ticks_step=ticks_step)
    if accuracy_mode in ["baseline_change", "baseline_rel_change"]:
        ax.axhline(0, color="#000000", linestyle="-", linewidth=1.5, zorder=3)

        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda y, _: f"{'+' if y > 0 else ''}{y:.0f}")
        )
    
    for ticklabel in ax.get_xticklabels():
        ticklabel.set_fontsize(ticklabel.get_fontsize()*0.50)
    
    output_dir = Path(output_dir)
    fpath = output_dir / filename
    fpath.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(fpath, 
                dpi=300, 
                bbox_inches="tight",
                bbox_extra_artists=legend_artists
                )
    plt.close(fig)

    print(f"Saved plot to: {fpath}")


def main() -> None:
    ablations = {
        "spatial": {
            "roi_ablation_baseline": ["Name"],
            "roi_circling_text": ["Name", "ROI"],
            "no_roi_circling_yes_text_layout_position": ["Name", "Location"],
            "roi_circling_text_layout_position": ["Name", "ROI", "Location"],
            "roi_circling_no_text_layout_position": ["ROI", "Location"],
            "roi_circling_no_text": ["ROI"],
            "no_roi_circling_no_text_layout_position": ["Location"],
        },
        "physics": {
            "roi_ablation_baseline": ["Name"],
            "ablation_physics_duration_text": ["Name", "Duration"],
            "ablation_physics_mass_approx_text": ["Name", "Approx. Mass"],
            "ablation_physics_mass_text": ["Name", "Exact Mass"],
        },
        "llmbias": {
            "roi_ablation_baseline": ["Name"],
            "ablation_no_object": ["Name", "ROI masked"],
        }
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
    # parser.add_argument("--ablations-runs", nargs="*", default=list(ablations_tags.keys()))
    # parser.add_argument("--ablations-tags", nargs="*", default=list(ablations_tags.values()))
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
    for ablation_set_name, ablation_set in ablations.items():
        ablations_tags = ablation_set
        ablations_runs = {abl: run_name + "_" + abl for abl in list(ablations_tags.keys())}
        eval_df = collect_runs(ablations_runs.values(), args.base_path, args.vqa_set)

        baseline_name = args.baseline_run or list(ablations_tags.keys())[0]

        output_dir = args.output_dir / (run_name + "_ablations") / args.vqa_set
        cur_output_dir = output_dir

        if args.family is not None:
            print("Filtering to family:", args.family)
            eval_df = eval_df[eval_df['model_family'] == args.family]
            assert eval_df['idx'].nunique() > 0, f"No entries found for family {args.family} in eval_df after filtering. Check if family name is correct and if there are entries for that family."

            # Use subdirectory for family-specific results
            cur_output_dir = output_dir / f"family_{args.family}"
        cur_output_dir.mkdir(parents=True, exist_ok=True)

        for group in ["model"]:
            cur_df, group_by = utils_read.apply_group(eval_df, group)
            
            print(f"Processing: grouping by {group_by}: with {len(cur_df)} entries")
        

            if ablation_set_name == "llmbias":
                accuracy_modes = ["baseline_change"]
                change_rel_threshold = 0  # any change should be considered
            else:
                # accuracy_modes = ["absolute"]
                # accuracy_modes = ["baseline_change", "baseline_rel_change", "absolute"]
                accuracy_modes = ["baseline_change", "absolute"]
                change_rel_threshold = 5  # 5% change threshold for improvement/worsening

            for acc_mode in accuracy_modes:
                plot_ablation(
                    cur_df,
                    cur_output_dir,
                    group_by=group_by,
                    accuracy_mode=acc_mode,
                    filename=f"ablation_{ablation_set_name}_{acc_mode}_{group}.png",
                    plot_baseline=acc_mode == "absolute",
                    ablations_runs=ablations_runs,
                    ablations_tags=ablations_tags,
                    baseline_name=baseline_name,
                    legend=["improved"] if ablation_set_name != "physics" else False,  # or all
                    # legend_mode=["None"]  # or all
                    change_rel_threshold=change_rel_threshold
                )


if __name__ == "__main__":
    main()