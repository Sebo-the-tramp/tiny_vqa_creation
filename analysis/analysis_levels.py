from __future__ import annotations

import argparse
from pathlib import Path
import re
from tokenize import group

import pandas as pd

from utils import (
    utils_read,
    utils_mapping,
    utils_graph,
    utils_graph_levels
)

def _old_build_eval_df(base_path: str | Path) -> pd.DataFrame:
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
        default="../output/",
    )
    parser.add_argument("--run-name", default="run_24_general_levels")
    parser.add_argument(
        "--vqa-set",
        default="10K",
        help="VQA set to use (e.g., 10K, 30K, karo_5K).",
    )
    args = parser.parse_args()

    output_dir = Path("output") / args.run_name / "levels" / "general"

    eval_df = utils_read.build_eval_df(args.run_name, args.base_path, args.vqa_set)

    eval_df["level"] = eval_df["idx"].apply(lambda x: re.fullmatch(utils_read._LEVEL_RE, x).groups()[2])
    eval_df["level"] = eval_df["level"].apply(lambda x: "undergrad" if x == "undegrad" else x)  # fix typo
    eval_df["idx"] = eval_df["idx"].apply(lambda x: re.fullmatch(utils_read._LEVEL_RE, x).groups()[0])

    for group in utils_read.GROUPINGS + ["family"]:
        cur_df, group_by = utils_read.apply_group(eval_df, group)
        
        for acc_mode in ["absolute", "baseline_change"]:
            utils_graph_levels.create_levels_plot(cur_df, 
                                    filename=f"levels_{acc_mode}_{group}.png",
                                    output_dir=output_dir,
                                    group_by=group_by,
                                    accuracy_mode=acc_mode,
                                    show=False,
                                    )

if __name__ == "__main__":
    main()
