from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils.utils_read import load_results, _sanitize_answer
import utils.utils_graph as utils_graph
from utils.utils_graph import create_graph_from_eval_balanced

DEFAULT_ABLATIONS = [
    "run_23_ablation_baseline",
    "run_23_roi_circling_no_text",
    "run_23_roi_circling_no_text_layout_position",
    "run_23_roi_circling_text",
    "run_23_roi_circling_text_layout_position",
    "run_23_black",
]


def _get_results_dir(base: Path, run_name: str) -> Path:
    sanitized = base / run_name / f"results_{run_name}_sanitized"
    if sanitized.exists():
        return sanitized
    return base / run_name / f"results_{run_name}"


def build_eval_df(base_path: str | Path, run_name: str) -> pd.DataFrame:
    base = Path(base_path)

    df = load_results(
        base,
        run_folder=run_name,
        merge_model_answers=True,
        model_answers_wide=True,
        cache=True,
    )

    results_dir = _get_results_dir(base, run_name)
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
        default="/data0/sebastian.cavada/compositional-physics/tiny_vqa_creation/output/",
    )
    parser.add_argument("--ablations", nargs="*", default=DEFAULT_ABLATIONS)
    parser.add_argument("--output-run-name", default="run_23_roi_ablation")
    args = parser.parse_args()

    utils_graph.RUN_NAME = args.output_run_name

    eval_dfs = []
    for run_name in args.ablations:
        eval_dfs.append(build_eval_df(args.base_path, run_name))

    acc_mats: dict[str, pd.DataFrame] = {}
    for run_name, eval_df in zip(args.ablations, eval_dfs):
        eval_df_single_image = eval_df[eval_df["idx"].astype(str).str.contains("_i")]
        print("number of questions:", len(eval_df_single_image))
        print(run_name)

        acc_mat, _ = create_graph_from_eval_balanced(
            eval_base=eval_df_single_image,
            index_to_use="question_id",
            title=(
                "Balanced accuracy by question_id and general models - "
                f"single-image - {run_name}"
            ),
            color_by_mode=True,
        )

        acc_mats[run_name] = acc_mat


if __name__ == "__main__":
    main()
