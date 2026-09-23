from __future__ import annotations

import argparse
from pathlib import Path

import utils.utils_read
import utils.utils_graph as utils_graph
from utils.utils_graph_correlation import (
    create_material_stiffness_violin_grid,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-path",
        default="../output",
    )
    parser.add_argument("--run-name", default="run_24_general_yms_variations")
    parser.add_argument(
        "--vqa-set",
        default="10K",
        help="VQA set to use (e.g., 10K, 30K, karo_5K).",
    )
    args = parser.parse_args()

    utils_graph.RUN_NAME = args.run_name

    output_dir = Path("output") / args.run_name / args.vqa_set / "yms" / "mixed"
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_df = utils.utils_read.build_eval_df(args.run_name, args.base_path, vqa_set=args.vqa_set, columns=["object-yms"])

    # print(eval_df.head().to_string())
    for group in utils.utils_read.GROUPINGS:
        print(f"Analyzing YMS group by: {group}")
        fname = f"yms_{group}.png"
        cur_df, group_by = utils.utils_read.apply_group(eval_df, group)
        
        for category_col in ["category"]:
            fig = create_material_stiffness_violin_grid(
                cur_df,
                output_dir=output_dir,
                show=False,
                save_grid=True,
                y_limit_mode="fit",
                group_by=group_by,  # model_id or family
                category_col=category_col,  # sub_category or category
                filename=fname
            )


if __name__ == "__main__":
    main()
