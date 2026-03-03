from __future__ import annotations

import argparse
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from utils import (
    utils_read,
    utils_graph,
    utils_mapping,
)

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
        default="150K",
        help="VQA set to use (e.g., 10K, 30K, karo_5K).",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "general", "image-only", "mixed"],
        default="all",
        help="Filter by model mode; all keeps all models.",
    )
    parser.add_argument("--run-name", default="run_28_general")
    args = parser.parse_args()

    eval_df = utils_read.build_eval_df(args.run_name, 
                                       args.base_path, 
                                       vqa_set=args.vqa_set,
    )

    output_dir = Path("output") / args.run_name / args.vqa_set / "commonsense"
    output_dir.mkdir(parents=True, exist_ok=True)

    # eval_df_single_image = eval_df[eval_df["idx"].astype(str).str.contains("_i")]
    # acc_mat_single, _ = utils_graph.create_graph_from_eval_balanced(
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

    benchmarks = ["all", "MMBench V1.1", "MMStar", "MMMU", "MathVista", "HallusionBench Avg.", "AI2D", "OCRBench", "MMVet"]
    
    for mode_label, mode_df in utils_read.select_eval_df(
        eval_df, mode=args.mode
    ):
        cur_output_dir = output_dir / mode_label
        if args.family is not None:
            print("Filtering to family:", args.family)
            eval_df = eval_df[eval_df['model_family'] == args.family]
            assert eval_df['idx'].nunique() > 0, f"No entries found for family {args.family} in eval_df after filtering. Check if family name is correct and if there are entries for that family."

            # Use subdirectory for family-specific results
            cur_output_dir = cur_output_dir / f"family_{args.family}"
        cur_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Compute the matrix for all categories only once
        utils_graph.create_graph_from_eval_balanced(
            eval_base=mode_df,
            index_to_use="sub_category",
            filename=f"accuracy_matrix.png",
            color_by_mode=True,
            show=False,
            out_dir=cur_output_dir,
        )
        
        categories = mode_df["category"].unique()
        # print("Categories:", categories)
        for cat in np.hstack([categories, "all"]):
            print(" Processing:", cat)
            if cat == "all":
                cat_df = mode_df
                cat_label = "Overall accuracy (%)"
            else:
                cat_df = mode_df[mode_df["category"] == cat]
                cat_label = utils_mapping.mapping_cat_short.get(cat)
            
            for bench in benchmarks:
                print("\n     Bench:", bench)

                fpath = utils_graph.get_benchmark_filepath(cur_output_dir, cat, bench)
                # if fpath.exists():
                #     print(f"         {fpath} already exists, skipping...")
                #     continue

                fpath.parent.mkdir(parents=True, exist_ok=True)
                utils_graph.create_accuracy_bench_vs_common_sense(
                    cat_df,
                    out_filename=fpath.name ,
                    show_legend=cat == "all",
                    group_by="model_id",
                    ylabel= cat_label,
                    legend_fontsize=12 if cat != "all" else 10,
                    show_xlabel= cat == "all",
                    figsize=(6, 2.5) if cat == "all" else (4, 2.5),
                    out_dir=fpath.parent,
                    # ylim=(0, 60)
                    benchmark=bench,
                )

        utils_graph.create_benchmarks_violin(
            mode_df,
            output_dir=cur_output_dir,
            filename="benchmarks_violin.png",
            benchmarks=benchmarks,
            figsize=(8.5, 4),
        )



if __name__ == "__main__":
    main()
