from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import utils.utils_read

if not Path("/scratch/project/eu-25-92/composite_physics/dataset/simulation_v4/").exists():
    utils.utils_read.SIM_PATH_MODIFIER = lambda x: x.replace("/scratch/project/eu-25-92/composite_physics/dataset/simulation_v4/", "/nfs/data/workspaces/rdechare/codes/physics-sim/output/sims/v4/")

from utils.utils_read import load_results, _sanitize_answer
import utils.utils_graph as utils_graph
import utils.utils_graph_correlation as utils_graph_correlation
from utils.utils_graph_correlation import (
    create_num_objects_category_curve,
    create_num_objects_violin_grid,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-path",
        default="../output/",
    )
    parser.add_argument("--run-name", default="run_28_general")
    parser.add_argument(
        "--mode",
        choices=["all", "general", "image-only", "mixed"],
        default="all",
        help="Filter by model mode; all keeps all models.",
    )
    parser.add_argument(
        "--family",
        default=None,
        help="Family to filter",
    )
    parser.add_argument(
        "--vqa-set",
        default="150K",
        help="VQA set to use (e.g., 10K, 30K, karo_5K).",
    )
    balancing_group = parser.add_mutually_exclusive_group(required=False)
    balancing_group.add_argument(
        "--balanced",
        dest="balanced",
        action="store_true",
        help="Balance the dataset, assessing that every object_count splits have similar question distribution than the overall vqa sets, for correlation analysis.",
    )
    balancing_group.add_argument(
        "--unbalanced",
        dest="balanced",
        action="store_false",
        help="Use the maximum set available without any question balancing (this is because balancing might have side effect to drastically increase variance for rare questions).",
    )
    parser.set_defaults(balanced=False)
    args = parser.parse_args()

    utils_graph.RUN_NAME = args.run_name
    utils_graph_correlation.RUN_NAME = args.run_name

    output_dir = Path("output") / args.run_name / args.vqa_set / "correlation"
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_df = utils.utils_read.build_eval_df(args.base_path, vqa_set=args.vqa_set)

    if args.family is not None:
        print("Filtering to family:", args.family)
        eval_df = eval_df[eval_df['model_family'] == args.family]
        assert eval_df['idx'].nunique() > 0, f"No entries found for family {args.family} in eval_df after filtering. Check if family name is correct and if there are entries for that family."
    
        # Use subdirectory for family-specific results
        output_dir = output_dir / f"family_{args.family}"
    
    if args.balanced:
        print("Creating the largest balanced subset across object counts for the correlation analysis..")
        dfs = []
        for mode in ["image-only", "general"]:
            print(f"    Balancing {mode} models")
            # Create balancing set per mode, to avoid penalizing video models
            mode_df = eval_df[eval_df["model_mode"] == mode]
            mode_df = utils.utils_read.balanced_split_df(mode_df, ["model_family", "model_id", "object_count"], ["question_id"], max_size=None)
            dfs.append(mode_df)
        balanced_df = pd.concat(dfs, ignore_index=True)

        print("Num questions for object_count "+", ".join(str(x) for x in sorted(balanced_df['object_count'].unique())))
        for col in balanced_df.groupby(["model_family", "model_id"], observed=True):
            print(f"    {col[0][1]}: \t{col[1]['object_count'].value_counts().sort_index().values}")

        eval_df_src = balanced_df
        print(f"Kept {balanced_df.shape[0]} balanced rows from the initial {eval_df.shape[0]} rows")
    else:
        print(f"Not balancing; using all {eval_df.shape[0]} rows for the correlation analysis")
        eval_df_src = eval_df

    for category_filter in ["category"]:
        if category_filter == "*":
            # Mix all categories to get overall results
            eval_df = eval_df_src.copy()
            eval_df["category"] = "all"
            eval_df["sub_category"] = "all"
        else:
            eval_df = eval_df_src

        for mode_label, mode_df in utils.utils_read.select_eval_df(
            eval_df, mode=args.mode
        ):
            cur_output_dir = output_dir / mode_label / f"{'balanced' if args.balanced else 'unbalanced'}"
            cur_output_dir.mkdir(parents=True, exist_ok=True)

            for group in utils.utils_read.GROUPINGS:
                cur_df, group_by = utils.utils_read.apply_group(mode_df, group)
                
                create_num_objects_violin_grid(
                    cur_df,
                    group_by=group_by,
                    save_per_category=True,
                    save_grid=True,
                    legend_cols=6,
                    output_dir=cur_output_dir / f"numobj_{category_filter}_{group}",
                    category_column=category_filter if category_filter != "*" else "category",
                )


if __name__ == "__main__":
    main()
