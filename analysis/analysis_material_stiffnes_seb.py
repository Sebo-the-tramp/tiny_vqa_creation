from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils.utils_read import load_results, _sanitize_answer
import utils.utils_graph as utils_graph
from utils.utils_graph import (
    create_graph_from_eval_balanced,
    create_sub_categories_summary,
    create_correlation_common_sense,
    create_accuracy_bench_vs_common_sense,
)

from utils.utils_graph_correlation import (
    create_material_stiffness_violin_grid,
)

from utils.utils_paper import print_heatmap_table_latex


def build_eval_df(base_path: str | Path) -> pd.DataFrame:
    base = Path(base_path)

    run_folder = Path(utils_graph.RUN_NAME)

    df = load_results(
        base,
        run_folder,
        merge_model_answers=True,
        model_answers_wide=True,
        cache=True,
        add_sim_metadata=True
    )

    results_dir = base / run_folder / f"results_{run_folder}_sanitized"
    model_cols = sorted(
        p.stem.replace("_val", "") for p in results_dir.glob("*_val.json")
    )
    model_cols = [c for c in model_cols if c in df.columns]
    if not model_cols:
        raise ValueError(f"No model answer columns found in {results_dir}")

    df["answer"] = df["answer"].apply(
        lambda a: _sanitize_answer(a, max_prefix_chars=None)
    )

    id_cols = [
        c
        for c in [
            "idx",
            "question_id",
            "category",
            "sub_category",
            "num_objects",
            "object_count",
            "answer",
            "mode_test",
            "mode_val",
            "mode",
            "object-yms"
        ]
        if c in df.columns
    ]

    eval_df = df.melt(
        id_vars=id_cols,
        value_vars=model_cols,
        var_name="model_id",
        value_name="model_answer",
    )

    valid = eval_df["model_answer"].notna() & eval_df["answer"].notna()
    eval_df["is_correct"] = pd.NA
    eval_df.loc[valid, "is_correct"] = (
        eval_df.loc[valid, "model_answer"] == eval_df.loc[valid, "answer"]
    )

    if "mode_val" in eval_df.columns:
        eval_df["mode_y"] = eval_df["mode_val"]
    elif "mode_test" in eval_df.columns:
        eval_df["mode_y"] = eval_df["mode_test"]
    elif "mode" in eval_df.columns:
        eval_df["mode_y"] = eval_df["mode"]

    return eval_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-path",
        default="/data0/sebastian.cavada/compositional-physics/tiny_vqa_creation/output/",
    )
    parser.add_argument("--run-name", default="run_24_general_yms-variations")
    args = parser.parse_args()

    utils_graph.RUN_NAME = args.run_name

    output_dir = Path("/data0/sebastian.cavada/compositional-physics/tiny_vqa_creation/analysis/output") / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_df = build_eval_df(args.base_path)

    # print(eval_df.head().to_string())
    fig = create_material_stiffness_violin_grid(
        eval_df,
        output_dir=output_dir,
        show=False,
        save_per_category=True,
        save_grid=True,
        save_legend=True,
        y_limit_mode="fit",
        show_legend=False,        
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
