from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

import utils.utils_read
import utils.utils_graph as utils_graph
from utils.utils_graph_correlation import (
    create_roi_material_yms_violin_grid,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-path",
        default="../output",
    )
    parser.add_argument("--run-name", default="run_28_general")
    parser.add_argument(
        "--vqa-set",
        default="150K-s17000-s6",
        help="VQA set to use (e.g., 10K, 30K, karo_5K).",
    )
    args = parser.parse_args()

    utils_graph.RUN_NAME = args.run_name

    output_dir = Path("output") / args.run_name / args.vqa_set / "roi-physics-shift" / "mixed"
    output_dir.mkdir(parents=True, exist_ok=True)

    vqa_roi_path = Path(f"output/{args.run_name}/{args.vqa_set}/vqa_roi.json")
    assert vqa_roi_path.exists(), f"VQA ROI file not found at: {vqa_roi_path}. Please run vqa_roi_extract.py first to generate this file."
    vqa_roi_df = utils.utils_read._read_json_dataframe(vqa_roi_path)

    eval_df = utils.utils_read.build_eval_df(args.run_name, args.base_path, vqa_set=args.vqa_set, columns=["object-yms"])
    
    eval_roi_df = eval_df.merge(vqa_roi_df, on="idx", how="inner")

    roi_material_yms = eval_roi_df["roi_object_props"].apply(lambda x: list(x["material"]["youngs_modulus_pa"].values()))
    roi_material_yms_mean = roi_material_yms.apply(lambda x: np.array(x).mean())
    roi_yms = eval_roi_df["roi_object_props"].apply(lambda x: x["props"]["yms"])

    # eval_roi_df["roi_yms_shift_log"] = np.log10(roi_yms) - np.log10(roi_material_yms_mean)  # Not good, because log10 can't be higher that 0.xxx when increasing.
    # eval_roi_df["roi_yms_shift_log"] = np.round(eval_roi_df["roi_yms_shift_log"]*2, 0)/2.

    eval_roi_df["roi_yms_shift_log"] = (roi_yms - roi_material_yms_mean)/roi_material_yms_mean
    eval_roi_df["roi_yms_shift_log"] = np.round(eval_roi_df["roi_yms_shift_log"]*2, 0)/2
    print("values: ", eval_roi_df["roi_yms_shift_log"].value_counts().sort_index())
    
    eval_roi_df["roi_scale"] = eval_roi_df["roi_object_props"].apply(lambda x: x["scale"])
    eval_roi_df["roi_scale"] = np.round(eval_roi_df["roi_scale"]*2, 0)/2


    # print(eval_df.head().to_string())
    # for group in utils.utils_read.GROUPINGS:
    for group in ["model"]:
        print(f"Analyzing YMS group by: {group}")
        cur_df, group_by = utils.utils_read.apply_group(eval_roi_df, group)
        
        for category_col in ["category"]:
            fig = create_roi_material_yms_violin_grid(
                cur_df,
                output_dir=output_dir,
                show=False,
                save_grid=True,
                y_limit_mode="fit",
                group_by=group_by,  # model_id or family
                category_col=category_col,  # sub_category or category
                stiffness_col="roi_yms_shift_log",
                filename=f"roi_yms_{group}.png"
            )
            
            fig = create_roi_material_yms_violin_grid(
                cur_df,
                output_dir=output_dir,
                show=False,
                save_grid=True,
                y_limit_mode="fit",
                group_by=group_by,  # model_id or family
                category_col=category_col,  # sub_category or category
                stiffness_col="roi_scale",
                filename=f"roi_scale_{group}.png"
            )


if __name__ == "__main__":
    main()
