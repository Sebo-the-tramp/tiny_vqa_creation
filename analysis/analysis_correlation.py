from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import utils.utils_read

if not Path("/scratch/project/eu-25-92/composite_physics/dataset/simulation_v4/").exists():
    utils.utils_read.sim_path_fct = lambda x: x.replace("/scratch/project/eu-25-92/composite_physics/dataset/simulation_v4/", "/nfs/data/workspaces/rdechare/codes/physics-sim/output/sims/v4/")

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
    parser.add_argument("--run-name", default="run_24_general_obj_num")
    parser.add_argument(
        "--mode",
        choices=["mixed", "general", "image-only"],
        default="mixed",
        help="Filter by model mode; mixed keeps all models.",
    )
    parser.add_argument(
        "--split-by-mode",
        action="store_true",
        help="Generate separate outputs per model mode when --mode=mixed.",
    )
    parser.add_argument(
        "--family-marker-mode",
        choices=["distinct", "rotated"],
        default="distinct",
        help="Use distinct shapes per family or rotate a base shape per family.",
    )
    # parser.add_argument(
    #     "--family-marker-base",
    #     default="^",
    #     help="Base marker to rotate when --family-marker-mode=rotated.",
    # )
    parser.add_argument(
        "--family",
        default=None,
        help="Family to filter",
    )
    parser.add_argument(
        "--vqa-set",
        default="30K",
        help="VQA set to use (e.g., 10K, 30K, karo_5K).",
    )
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
    
    eval_df_src = eval_df

    for category_filter in ("category", "*"):
        if category_filter == "*":
            # Mix all categories to get overall results
            eval_df = eval_df_src.copy()
            eval_df["category"] = "all"
            eval_df["sub_category"] = "all"
        else:
            eval_df = eval_df_src

        for mode_label, mode_df in utils.utils_read.select_eval_df(
            eval_df, mode=args.mode, split_by_mode=args.split_by_mode
        ):
            if category_filter == "category":
                create_num_objects_category_curve(
                    mode_df,
                    # sample_frac=0.8,
                    output_dir=output_dir,
                    filename=f"num_objects_category_curve_{category_filter}_{mode_label}.png",
                    category_column=category_filter,
                    y_limit_mode="fixed",
                    run_name="all"
                )
            
            for x in range(1):
                create_num_objects_violin_grid(
                    mode_df,
                    group_by="model_id",
                    # metadata_path="analysis/utils/metadata.json",
                    save_per_category=True,
                    per_category_dirname=f"{'all/' if category_filter == '*' else ''}num_objects_per_model_{mode_label}",
                    save_grid=True,
                    save_legend=True,
                    legend_filename=f"num_objects_legend_models_{mode_label}.png",
                    legend_cols=6,
                    # sample_frac=0.8,
                    # sample_seed=x,
                    output_dir=output_dir,
                    family_marker_mode=args.family_marker_mode,
                    # family_marker_base=args.family_marker_base,
                    split_values=None,
                    category_column=category_filter if category_filter != "*" else "category",
                )

            create_num_objects_violin_grid(
                mode_df,
                group_by="model_family",
                save_per_category=True,
                per_category_dirname=f"{'all/' if category_filter == '*' else ''}num_objects_per_family_{mode_label}",
                save_grid=False,
                save_legend=True,
                legend_filename=f"num_objects_legend_families_{mode_label}.png",
                legend_cols=4,
                # sample_frac=0.8,
                output_dir=output_dir,
                family_marker_mode=args.family_marker_mode,
                # family_marker_base=args.family_marker_base,
                split_values=None,
                category_column=category_filter if category_filter != "*" else "category",
            )


if __name__ == "__main__":
    main()
