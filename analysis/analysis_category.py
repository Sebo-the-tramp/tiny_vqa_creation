from __future__ import annotations

import argparse
from fileinput import filename
from pathlib import Path

import pandas as pd
import tqdm

import utils.utils_read

if not Path("/scratch/project/eu-25-92/composite_physics/dataset/simulation_v4/").exists():
    utils.utils_read.sim_path_fct = lambda x: x.replace("/scratch/project/eu-25-92/composite_physics/dataset/simulation_v4/", "/nfs/data/workspaces/rdechare/codes/physics-sim/output/sims/v4/")

# from utils.utils_read import load_results, _sanitize_answer
import utils.utils_graph as utils_graph
import utils.utils_graph_correlation as utils_graph_correlation
from utils.utils_graph_correlation import (
    create_category_accuracy,
    create_num_objects_violin_grid
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-path",
        default="../output",
    )
    parser.add_argument("--run-name", default="run_26_general")
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
    parser.add_argument(
        "--vqa-set",
        default="30K",
        help="VQA set to use (e.g., 10K, 30K, karo_5K).",
    )
    args = parser.parse_args()

    utils_graph.RUN_NAME = args.run_name

    output_dir = Path("output") / args.run_name / args.vqa_set / "category"
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_df = utils.utils_read.build_eval_df(args.base_path, vqa_set=args.vqa_set)
    
    for mode_label, mode_df in utils.utils_read.select_eval_df(
        eval_df, mode=args.mode, split_by_mode=args.split_by_mode
    ):
        for group_by in ["model_family", "model_id", "model_best"]:
            fname = f"acc_by_cat_{mode_label}_{group_by}.png"

            if group_by == "model_best":
                # Compute the per model accuracy and keep only best overall model per family
                model_accuracy = mode_df.groupby(['model_family', 'model_id'])['accuracy'].mean().reset_index()
                best_models = model_accuracy.loc[model_accuracy.groupby('model_family')['accuracy'].idxmax()]
                cur_df = mode_df[mode_df['model_id'].isin(best_models['model_id'])]
                
                group_by = "model_id"
            else:
                cur_df = mode_df
            
            print(f"Processing mode: {mode_label}, grouping by {group_by}: with {len(cur_df)} entries")
            create_category_accuracy(
                cur_df,
                # group_by="family",
                # save_per_category=True,
                # per_category_dirname=f"category_{mode_label}",
                # save_grid=False,
                # save_legend=True,
                # legend_filename=f"category_{mode_label}.png",
                # legend_cols=4,
                # sample_frac=1.0,
                output_dir=output_dir,
                family_marker_mode=args.family_marker_mode,
                # metadata_path="utils/metadata.json",
                filename=fname,
                y_limit_mode="",
                group_by=group_by,
                show_legend=True,
                bars=False,
            )


if __name__ == "__main__":
    main()
