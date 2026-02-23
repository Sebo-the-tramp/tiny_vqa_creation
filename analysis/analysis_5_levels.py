from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import utils.utils_graph as utils_graph
from utils.utils_read import load_results_levels, _sanitize_answer
import utils.utils_graph_levels as utils_graph_levels
from utils.utils_graph_levels import create_scatter_by_family


def build_eval_df(base_path: str | Path) -> pd.DataFrame:
    base = Path(base_path)
    df = load_results_levels(
        base,
        run_folder=Path(utils_graph.RUN_NAME),
        merge_model_answers=True,
        model_answers_wide=True,
        cache=True,
    )

    results_dir = base / utils_graph.RUN_NAME / f"results_{utils_graph.RUN_NAME}"
    model_cols = sorted(
        p.stem.replace("_val", "") for p in results_dir.glob("*_val.json")
    )
    model_cols = [c for c in model_cols if c in df.columns]
    if not model_cols:
        raise ValueError(f"No model answer columns found in {results_dir}")

    df["answer"] = df["answer"].apply(_sanitize_answer)
    if "level" not in df.columns:
        df["level"] = df["idx"].astype(str).str.extract(r"level_([^_]+)", expand=False)

    id_cols = [
        c
        for c in [
            "idx",
            "question_id",
            "category",
            "sub_category",
            "level",
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
    parser.add_argument("--run-name", default="run_11_general_levels")
    args = parser.parse_args()

    utils_graph.RUN_NAME = args.run_name
    utils_graph_levels.RUN_NAME = args.run_name

    output_dir = Path("output") / args.run_name / "levels" / "mixed"
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_df = build_eval_df(args.base_path)

    create_scatter_by_family(eval_df, 
                             filename="levels_by_family.png",
                             split_by_mode=True, 
                             show=False, \
                            #  levels_sorted=["child", "teen", "undegrad", "graduate", "expert"] \
                             )


if __name__ == "__main__":
    main()
