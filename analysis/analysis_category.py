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
    parser.add_argument("--run-name", default="run_28_general")
    parser.add_argument(
        "--mode",
        choices=["all", "general", "image-only", "mixed"],
        default="all",
        help="Filter by model mode; all keeps all models.",
    )
    parser.add_argument(
        "--family-marker-mode",
        choices=["distinct", "rotated"],
        default="distinct",
        help="Use distinct shapes per family or rotate a base shape per family.",
    )
    parser.add_argument(
        "--vqa-set",
        default="150K",
        help="VQA set to use (e.g., 10K, 30K, karo_5K).",
    )
    args = parser.parse_args()

    utils_graph.RUN_NAME = args.run_name

    output_dir = Path("output") / args.run_name / args.vqa_set / "category"
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_df = utils.utils_read.build_eval_df(args.base_path, vqa_set=args.vqa_set)
    
    for mode_label, mode_df in utils.utils_read.select_eval_df(
        eval_df, mode=args.mode
    ):
        cur_output_dir = output_dir / mode_label
        cur_output_dir.mkdir(parents=True, exist_ok=True)
        
        for group in utils.utils_read.GROUPINGS:
            fname = f"acc_{group}.png"
            cur_df, group_by = utils.utils_read.apply_group(mode_df, group)
            
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
                output_dir=cur_output_dir,
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
