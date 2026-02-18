from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils.utils_graph_levels import _load_model_metadata
from utils.utils_read import load_results, _sanitize_answer
import utils.utils_graph as utils_graph
from utils.utils_graph import (
    create_graph_from_eval_balanced,
    create_sub_categories_summary,
    create_correlation_common_sense,
    create_accuracy_bench_vs_common_sense
)

from utils.utils_paper import print_heatmap_table_latex
import utils.utils_mapping

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-path",
        default="../output/",
    )
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
    parser.add_argument("--run-name", default="run_24_general")
    args = parser.parse_args()

    utils_graph.RUN_NAME = args.run_name

    eval_df = utils.utils_read.build_eval_df(args.base_path, vqa_set=args.vqa_set)

    output_dir = Path("output") / args.run_name / args.vqa_set / "commonsense"
    if args.family is not None:
        print("Filtering to family:", args.family)
        eval_df = eval_df[eval_df['model_family'] == args.family]
        assert eval_df['idx'].nunique() > 0, f"No entries found for family {args.family} in eval_df after filtering. Check if family name is correct and if there are entries for that family."

        # Use subdirectory for family-specific results
        output_dir = output_dir / f"family_{args.family}"

    output_dir.mkdir(parents=True, exist_ok=True)

    # eval_df_single_image = eval_df[eval_df["idx"].astype(str).str.contains("_i")]
    # acc_mat_single, _ = create_graph_from_eval_balanced(
    #     eval_base=eval_df_single_image,
    #     index_to_use="question_id",
    #     title="Balanced accuracy by question_id and general models - single-image task",
    #     color_by_mode=True,
    #     show=False,
    #     include_counts=True,
    #     color_question_id_by_subcategory=True,
    # )


    # eval_df_multi_image = eval_df[eval_df["idx"].astype(str).str.contains("_g")]
    # eval_df_multi_image = eval_df_multi_image.groupby("model_id").filter(
    #     lambda g: g["model_answer"].notna().any()
    # )
    # acc_mat_multi, _ = create_graph_from_eval_balanced(
    #     eval_base=eval_df_multi_image,                     # your row-level eval with is_correct
    #     index_to_use="question_id",
    #     title="Balanced accuracy by question_id and general models - multi-image task",
    #     color_by_mode=True,
    #     show=False,
    #     include_counts=True,
    #     color_question_id_by_subcategory=True,
    # )

    # print_heatmap_table_latex(
    #     acc_mat_single, output_path=str(output_dir / "heatmap_table_single.txt")
    # )
    # print_heatmap_table_latex(
    #     acc_mat_multi, output_path=str(output_dir / "heatmap_table_multi.txt")
    # )

    categories = eval_df["category"].unique()
    # print("Categories:", categories)
    for cat in np.hstack([categories, "all"]):
        print("Processing:", cat)
        eval_df_sub = eval_df.copy() if cat == "all" else eval_df[eval_df["category"] == cat].copy()

        acc_mat, _ = create_graph_from_eval_balanced(
            eval_base=eval_df_sub,
            index_to_use="sub_category",
            title="Balanced accuracy by sub_category and model",
            color_by_mode=True,    
            show=False,
            out_dir=output_dir,
        )

        # create_sub_categories_summary(
        #     acc_mat=acc_mat,
        #     title="Sub-category accuracy summary - all",
        #     show=False,
        # )

        # create_correlation_common_sense(
        #     eval_df_sub,
        #     acc_mat,
        #     title="Correlation common sense",
        #     show=False,
        # )


        if cat == "all":
            cat_label = "Overall accuracy (%)"
        else:
            cat_label = utils.utils_mapping.mapping_cat_short.get(cat)
        create_accuracy_bench_vs_common_sense(
            eval_df_sub,
            acc_mat,
            out_filename="cs_" + cat + ".png" if cat != "all" else "cs_correlation.png",
            show_legend=cat == "all",
            group_by="model_id",
            ylabel= cat_label,
            legend_fontsize=12 if cat != "all" else 10,
            show_xlabel= cat == "all",
            figsize=(6, 2.5) if cat == "all" else (4, 2.5),
            out_dir=output_dir,
        )

    

if __name__ == "__main__":
    main()
