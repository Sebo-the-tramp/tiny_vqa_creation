from __future__ import annotations

import argparse
import glob
import json
import math
import re
from pathlib import Path

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter, MultipleLocator

from utils import (
    utils_read,
    utils_mapping,
    utils_graph
)


DEFAULT_BASE_PATH = Path(
    "/data0/sebastian.cavada/compositional-physics/tiny_vqa_creation/output"
)


def replace_subcategory_in_json_file(
    file_path: str | Path,
    *,
    old_subcategory: str = "object_persistence",
    new_subcategory: str = "object_identity",
) -> int:
    file_path = Path(file_path)
    text = file_path.read_text(encoding="utf-8")

    old_literal = f'"sub_category": "{old_subcategory}"'
    new_literal = f'"sub_category": "{new_subcategory}"'

    replaced_count = text.count(old_literal)
    updated_text = text.replace(old_literal, new_literal)

    if updated_text != text:
        file_path.write_text(updated_text, encoding="utf-8")

    return replaced_count


def collect_runs(run_name:str, base_path: Path, vqa_set: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

      # check that base_path is a parent of 'test'
    files = glob.glob(str(base_path / f"{run_name}_counterfactual_*" / f"test_{run_name}_counterfactual_*_{vqa_set}.json"))
    for file in files:
        print(f"Processing file: {file}")
        
        cf_run_name = Path(file).parent.name
        count = replace_subcategory_in_json_file(file, old_subcategory="object_persistence", new_subcategory="object_identity")
        print(f"     Replaced {count} occurrence(s) of 'object_persistence' -> 'object_identity' in file if needed.")

        run_df = utils_read.build_eval_df(cf_run_name, base_path, vqa_set=vqa_set)
        if run_df.empty:
            raise ValueError(f"Run has no rows after loading: {file}")
        run_df["run_name"] = cf_run_name.replace(f"{run_name}_counterfactual_", "")
        frames.append(run_df)

    if not frames:
        raise ValueError("No data available for the selected runs.")

    return pd.concat(frames, ignore_index=True)

def plot_counterfact(
    runs_df: pd.DataFrame, 
    output_dir: Path, 
    accuracy_mode: str="baseline_change",  # baseline_change, absolute, baseline_rel_change
    filename: str="counterfactual_rel.png",
    *, 
    plot_baseline: bool=False,
    baseline_run_name: str="factual",
    group_by: str = "model_id",
    metadata_path: str | Path | None = "utils/metadata.json",
    legend: list[str]|bool = ["all"],  # improved, worsened, all
    change_rel_threshold: int = 5  # 5% relative change threshold for improvement/worsening
) -> None:
    plot_df = runs_df.copy()
    plot_df["accuracy"] *= 100
    
    fact_df = plot_df[plot_df["run_name"] == baseline_run_name].copy()
    counterfact_df = plot_df[plot_df["run_name"] != baseline_run_name].copy()
    cf_questions = counterfact_df["question_id"].unique()
    assert all(f.startswith("CF_") for f in cf_questions)

    f_questions = pd.Series(cf_questions, dtype="string").str.replace(r"^CF_", "F_", regex=True)
    fact_df = fact_df[fact_df["question_id"].isin(f_questions)]

    print(f"Fact df: {len(fact_df)} entries: {fact_df['question_id'].nunique()} questions")
    print(f"Counterfact df: {len(counterfact_df)} entries: {counterfact_df['question_id'].nunique()} questions")

    print(f"Fact df: {len(fact_df)} entries: {sorted(fact_df['question_id'].unique())}")
    print(f"Counterfact df: {len(counterfact_df)} entries: {sorted(counterfact_df['question_id'].unique())}")

    assert fact_df["question_id"].nunique() == counterfact_df["question_id"].nunique(), "Number of unique questions should be the same in factual and counterfactual sets after filtering."
    assert set(fact_df["question_id"].unique()) == set([q[1:] for q in counterfact_df["question_id"].unique()]), "Factual and counterfactual question IDs should match after filtering."

    agg_df = utils_read.macro_accuracy(plot_df, level="model_id", group_by=["run_name"])

    agg_df["accuracy_change"] = agg_df.apply(
        lambda row: row["accuracy"] - agg_df[
            (agg_df[group_by] == row[group_by]) & (agg_df["run_name"] == baseline_run_name)
        ]["accuracy"].values[0],
        axis=1,
    )
    agg_df["accuracy_rel_change"] = agg_df.apply(
        lambda row: (row["accuracy"] - agg_df[
            (agg_df[group_by] == row[group_by]) & (agg_df["run_name"] == baseline_run_name)
        ]["accuracy"].values[0]) / agg_df[
            (agg_df[group_by] == row[group_by]) & (agg_df["run_name"] == baseline_run_name)
        ]["accuracy"].values[0] * 100,
        axis=1,
    )

    if accuracy_mode == "absolute":
        agg_df["accuracy_plot"] = agg_df["accuracy"]
    elif accuracy_mode == "baseline_change":
        agg_df["accuracy_plot"] = agg_df["accuracy_change"]
    elif accuracy_mode == "baseline_rel_change":
        agg_df["accuracy_plot"] = agg_df["accuracy_rel_change"]
    
    sx = 0.5 + 1.5*counterfact_df["run_name"].nunique()
    if not plot_baseline:
        sx -= 0.5
    if legend:
        sx += 1.2
    figsize = (sx, 3.1)
    fig, ax = plt.subplots(figsize=figsize)

    runs = sorted(fact_df["run_name"].unique().tolist() + counterfact_df["run_name"].unique().tolist())
    run_idx = {run: idx for idx, run in enumerate(runs)}  # Important to keep as string for seaborn order
    agg_df["run_idx"] = agg_df["run_name"].map(run_idx)

    agg_baseline_mask = agg_df["run_name"] == baseline_run_name

    # Build model style
    model_style, family_map = utils_mapping._build_model_style(
        metadata_path,
        group_by=group_by
    )
    rng = np.random.default_rng(42)

    sns.violinplot(
        data=agg_df,
        x="run_idx",
        y="accuracy_plot",
        ax=ax,
        color="0.90",
        # inner="box",
        inner=None,
        cut=0,
        width=1.0,
        linewidth=0.5,
        order=list(range(len(runs))),
        zorder=2
    )
    
    for group, df_m in agg_df.groupby(group_by):
        # Remove baseline points if not plotting baseline
        if not plot_baseline:
            df_m = df_m[df_m["run_name"] != baseline_run_name]
        
        x_vals = df_m["run_idx"].to_numpy().astype(int)
        y_vals = df_m["accuracy_plot"].to_numpy()
        
        if x_vals.size == 0:
            continue

        jitter = rng.uniform(-0.20, 0.20, size=x_vals.size)
        x_jittered = x_vals + jitter
        color, marker, size, edge = model_style[group]
        
        # improve = any((agg_df[agg_df[group_by] == group]["accuracy_rel_change"] > change_rel_threshold) & ~agg_baseline_mask)
        # if improve:
        #     alphas = []
        #     for _, r in df_m.iterrows():
        #         if r["run_name"] == baseline_run_name:
        #             alphas.append(1.0)
        #         elif r["accuracy_rel_change"] > change_rel_threshold:
        #             alphas.append(1.0)
        #         else:
        #             alphas.append(0.1)
        #     alpha = alphas
        #     zorder = 5
        # else:
        #     alpha = 0.10
        #     zorder = 4
        alpha = 0.9
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

    # def xtick_stats(abl_tags):
    #     abl_mask = None
    #     for tag in tags:
    #         tag_mask = agg_df[f"run_tag_{tag.lower()}"] == (True if tag in abl_tags else False)
    #         abl_mask = tag_mask if abl_mask is None else (abl_mask & tag_mask)

    #     rel_change = agg_df.loc[abl_mask, "accuracy_rel_change"]
    #     up = int((rel_change > change_rel_threshold).sum())
    #     down = int((rel_change < -change_rel_threshold).sum())
    #     flat = int(((rel_change >= -change_rel_threshold) & (rel_change <= change_rel_threshold)).sum())
    #     return up, down, flat

    xticks = list(range(len(runs)))
    xtick_items = list(runs)
    if not plot_baseline:
        xticks = [x for x, t in zip(xticks, runs) if t != baseline_run_name]
        xtick_items = [x for x, t in zip(xtick_items, runs) if t != baseline_run_name]

    xtick_labels_map = {
        "factual": "Original VQA\n(factual)",
        "shift": "Object\nshift",
        "smaller": "Object\nresize",
        "gravity": "Lower\ngravity",
    }
    xtick_labels = [xtick_labels_map.get(run, run.capitalize()) for run in xtick_items]

    if len(xticks) == 1:
        ax.set_xlim(xticks[0] - 0.7, xticks[0] + 0.7)
    ax.set_xticks(xticks)
    ax.set_xticklabels(
        xtick_labels,
        ha="center",
    )

    # stats_fontsize = 6
    # for i, (x, (up, down, flat)) in enumerate(zip(xticks, xtick_stats_values)):
    #     if plot_baseline and x == 0:
    #         continue # skip baseline stats
    #     stats_y = -0.08 - 0.04 * xtick_labels[i].count("\n")
    #     ax.text(
    #         x - 0.17, stats_y,
    #         f"↗{up}",
    #         transform=ax.get_xaxis_transform(),
    #         ha="right", va="top",
    #         fontsize=stats_fontsize, color="tab:green"
    #     )
    #     ax.text(
    #         x, stats_y,
    #         f"≈{flat}",
    #         transform=ax.get_xaxis_transform(),
    #         ha="center", va="top",
    #         fontsize=stats_fontsize, color="#666666"
    #     )
    #     ax.text(
    #         x + 0.15, stats_y,
    #         f"↘{down}",
    #         transform=ax.get_xaxis_transform(),
    #         ha="left", va="top",
    #         fontsize=stats_fontsize, color="tab:red"
    #     )
    ax.set_xlabel("", fontsize=1)
    if plot_baseline:
        ax.axvline(0.5, color="#666666", linewidth=1, alpha=0.5, zorder=-1, linestyle='--')

    if runs_df["category"].nunique() == 1:
        cat = runs_df["category"].unique()[0]
        ylabel_color = utils_mapping.mapping_cat_colors[cat]+"CC"
        ylabel = utils_mapping.mapping_cat_short[cat]
    else:
        ylabel = "Partial Accuracy"
        ylabel_color = "black"

    if accuracy_mode == "absolute":
        ylabel += " (%)"
        ticks_step = 5.0
    elif accuracy_mode == "baseline_change":
        ylabel += " (change %)"
        ticks_step = 10.0
    elif accuracy_mode == "baseline_rel_change":
        ylabel += "\n(rel. change %)"
        ticks_step = 5.0
    
    ax.set_ylabel(ylabel, color=ylabel_color)

    extra_artists = []
    y_bracket = -0.15   # below tick labels (axis-relative)
    y_text = y_bracket - 0.03  # further below the bracket

    # x in data coords (tick index), y in axis coords
    line = ax.plot(
        [1-0.4, 3+0.4], [y_bracket, y_bracket],
        transform=ax.get_xaxis_transform(),
        color="#8B0000",      # dark red
        linewidth=4.0,        # thick
        solid_capstyle="butt",
        clip_on=False,
        zorder=10,
    )

    text = ax.text(
        2, y_text,            # centered between 1 and 3
        "Counterfactual VQA",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        color="#8B0000",
        fontsize=10,
        fontweight="bold",
        clip_on=False,
    )
    extra_artists.append(line[0])
    extra_artists.append(text)

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

            if any((agg_df[group_mask]["accuracy_rel_change"] > change_rel_threshold) & ~agg_baseline_mask):
                improved.append(i)
            if any((agg_df[group_mask]["accuracy_rel_change"] < -change_rel_threshold) & ~agg_baseline_mask):
                worsen.append(i)

            legend_labels[i] = label

        groups = []
        if "improved" in legend or "all" in legend:
            title = f"Model improved\n($\greater$ {change_rel_threshold}% rel. change)"
            if change_rel_threshold == 0:
                title = "Model improved\n($>$ 0% rel. change)"
            groups.append((title, improved))
        if "worsened" in legend or "all" in legend:
            title = f"Model worsened\n($\less$ {(-change_rel_threshold)}% rel. change)"
            if change_rel_threshold == 0:
                title = "Model worsened\n($<$ 0% rel. change)"
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
            extra_artists.append(leg)
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
    
    for i, ticklabel in enumerate(ax.get_xticklabels()):
        ticklabel.set_fontsize(ticklabel.get_fontsize()*0.60)

        if not (plot_baseline and xtick_items[i] == baseline_run_name):
            ticklabel.set_color("#8B0000")
    
    output_dir = Path(output_dir)
    fpath = output_dir / filename
    fpath.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(fpath, 
                dpi=300, 
                bbox_inches="tight",
                bbox_extra_artists=extra_artists,
                pad_inches=0.
                )
    plt.close(fig)

    print(f"Saved plot to: {fpath}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run counterfactual analysis."
        )
    )
    parser.add_argument("--run-name", default="run_28")
    parser.add_argument("--base-path", type=Path, default="../output")
    parser.add_argument("--metadata-path", type=Path, default=Path("utils/metadata.json"))
    parser.add_argument(
        "--counterfactual-vqa-set",
        default="karo_10K",
        help="VQA set to use for counterfactual analysis (e.g., 10K, 30K, karo_5K).",
    )
    parser.add_argument(
        "--factual-vqa-set",
        default="150K",
        help="VQA set to use for factual analysis (e.g., 10K, 30K, karo_5K).",
    )
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

    cf_eval_df = collect_runs(args.run_name, args.base_path, args.counterfactual_vqa_set)
    f_df = utils_read.build_eval_df(args.run_name+"_general", args.base_path, args.factual_vqa_set)
    f_df["run_name"] = "factual"

    eval_df = pd.concat([cf_eval_df, f_df], ignore_index=True)

    output_dir = args.output_dir / (run_name + "_counterfactual") / args.counterfactual_vqa_set
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
    

        # if counterfactual_set_name == "llmbias":
        #     accuracy_modes = ["baseline_change"]
        #     change_rel_threshold = 0  # any change should be considered
        # else:
            # accuracy_modes = ["absolute"]
            # accuracy_modes = ["baseline_change", "baseline_rel_change", "absolute"]
        accuracy_modes = ["baseline_change", "absolute"]
        change_rel_threshold = 5  # 5% change threshold for improvement/worsening

        for acc_mode in accuracy_modes:
            plot_counterfact(
                cur_df,
                cur_output_dir,
                group_by=group_by,
                accuracy_mode=acc_mode,
                filename=f"counterfactual_{acc_mode}_{group}.png",
                # plot_baseline=acc_mode == "absolute",
                plot_baseline=True,
                legend=None,  # or all
                # legend_mode=["None"]  # or all
                change_rel_threshold=change_rel_threshold
            )


if __name__ == "__main__":
    main()