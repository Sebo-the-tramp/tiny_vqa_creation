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
    parser.add_argument("--run-name", default="run_24_general_yms-variations")
    parser.add_argument(
        "--vqa-set",
        default="10K",
        help="VQA set to use (e.g., 10K, 30K, karo_5K).",
    )
    args = parser.parse_args()

    utils_graph.RUN_NAME = args.run_name

    output_dir = Path("output") / args.run_name / args.vqa_set / "yms" / "mixed"
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_df = utils.utils_read.build_eval_df(args.base_path, vqa_set=args.vqa_set, columns=["object-yms"])

    # print(eval_df.head().to_string())
    for group in utils.utils_read.GROUPINGS:
        print(f"Analyzing YMS group by: {group}")
        fname = f"yms_violin_{group}.png"
        cur_df, group_by = utils.utils_read.apply_group(eval_df, group)
        
        # for category_col in ["category", "sub_category"]:
        for category_col in ["category"]:
            #  fig = create_material_stiffness_violin_grid(
            fig = create_material_stiffness_violin_grid(
                cur_df,
                output_dir=output_dir,
                run_name=args.run_name,
                show=False,
                save_per_category=True,
                save_grid=True,
                save_legend=True,
                y_limit_mode="fit",
                group_by=group_by,  # model_id or family
                category_col=category_col,  # sub_category or category
                show_legend=False,        
                # stiffness_labels=("Soft\n($\\text{yms} \leq 2e4$)", "Medium\n($2e4 > \\text{yms} \leq 1e6$)", "Stiff\n($\\text{yms} > 1e6$)"),
                stiffness_labels=("Soft", "Medium", "Stiff"),
                filename=fname
            )

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
    #     eval_base=eval_df_multi_image,  # your row-level eval with is_correct
    #     index_to_use="question_id",
    #     title="Balanced accuracy by question_id and general models - multi-image task",
    #     color_by_mode=True,
    #     show=False,
    #     include_counts=True,
    #     color_question_id_by_subcategory=True,
    # )


if __name__ == "__main__":
    main()
