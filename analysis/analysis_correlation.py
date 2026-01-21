from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils.utils_read import load_results, _sanitize_answer
import utils.utils_graph as utils_graph
import utils.utils_graph_correlation as utils_graph_correlation
from utils.utils_graph_correlation import (
    create_num_objects_violin_grid,
)

# from utils.utils_paper import print_heatmap_table_latex


def build_eval_df(base_path: str | Path) -> pd.DataFrame:
    base = Path(base_path)

    run_folder = Path(utils_graph.RUN_NAME)

    df = load_results(
        base,
        run_folder=run_folder,
        merge_model_answers=True,
        model_answers_wide=True,
        cache=True,
        add_sim_metadata=True,
    )

    results_dir = base / run_folder / f"results_{run_folder}"
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
        default="/data0/sebastian.cavada/compositional-physics/tiny_vqa_deterministic/output/",
    )
    parser.add_argument("--run-name", default="run_20_general_obj_num")
    args = parser.parse_args()

    utils_graph.RUN_NAME = args.run_name
    utils_graph_correlation.RUN_NAME = args.run_name

    output_dir = Path("output") / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_df = build_eval_df(args.base_path)

    for x in range(1, 1):
        create_num_objects_violin_grid(
            eval_df,
            group_by="model_id",
            save_per_category=True,
            per_category_dirname="num_objects_per_model",
            save_grid=True,
            save_legend=True,
            legend_filename="num_objects_legend_models.png",
            legend_cols=6,
            sample_frac=0.8,
            sample_seed=x,
            y_limit_mode="fixed",
        )

    create_num_objects_violin_grid(
        eval_df,
        group_by="family",
        save_per_category=True,
        per_category_dirname="num_objects_per_family",
        save_grid=False,
        save_legend=True,
        legend_filename="num_objects_legend_families.png",
        legend_cols=4,
        sample_frac=0.8,
    )


if __name__ == "__main__":
    main()
