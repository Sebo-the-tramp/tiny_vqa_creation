from __future__ import annotations

import argparse
from fileinput import filename
from pathlib import Path

import utils.utils_read
from utils.utils_graph_correlation import (
    create_accuracy,
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
        "--vqa-set",
        default="150K",
        help="VQA set to use (e.g., 10K, 30K, karo_5K).",
    )
    args = parser.parse_args()

    output_dir = Path("output") / args.run_name / args.vqa_set / "categorical"
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_df = utils.utils_read.build_eval_df(args.run_name, args.base_path, vqa_set=args.vqa_set)
    
    for mode_label, mode_df in utils.utils_read.select_eval_df(
        eval_df, mode=args.mode
    ):
        cur_output_dir = output_dir / mode_label
        cur_output_dir.mkdir(parents=True, exist_ok=True)
        
        for group in utils.utils_read.GROUPINGS:
            cur_df, group_by = utils.utils_read.apply_group(mode_df, group)
            
            print(f"Processing mode: {mode_label}, grouping by {group_by}: with {len(cur_df)} entries")
            for level in ["category", "sub_category", "question_id"]:
                create_accuracy(
                    cur_df,
                    output_dir=cur_output_dir,
                    level=level,
                    filename=f"acc_{level}_{group}.png",
                    y_limit_mode="",
                    group_by=group_by,
                    show_legend=True,
                    bars=False,
                )


if __name__ == "__main__":
    main()
