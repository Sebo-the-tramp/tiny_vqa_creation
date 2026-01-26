from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import utils.utils_read

if not Path("/scratch/project/eu-25-92/composite_physics/dataset/simulation_v4/").exists():
    utils.utils_read.sim_path_fct = lambda x: x.replace("/scratch/project/eu-25-92/composite_physics/dataset/simulation_v4/", "/nfs/data/workspaces/rdechare/codes/physics-sim/output/sims/v4/")

from utils.utils_read import load_results, _sanitize_answer
import utils.utils_graph as utils_graph
import utils.utils_graph_correlation as utils_graph_correlation
from utils.utils_graph_correlation import (
    create_num_objects_violin_grid,
)

# from utils.utils_paper import print_heatmap_table_latex


def add_model_mode(
    eval_df: pd.DataFrame, metadata_path: str | Path = "utils/metadata.json"
) -> pd.DataFrame:
    path = Path(metadata_path)
    if not path.exists():
        eval_df["model_mode"] = pd.NA
        return eval_df

    metadata_df = pd.read_json(path)
    if "id" in metadata_df.columns and "model_id" not in metadata_df.columns:
        metadata_df = metadata_df.rename(columns={"id": "model_id"})
    mode_map = (
        metadata_df.dropna(subset=["model_id"])
        .set_index("model_id")["mode"]
        .to_dict()
    )
    eval_df["model_mode"] = eval_df["model_id"].map(mode_map).fillna("unknown")
    return eval_df


def iter_mode_slices(eval_df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    if "model_mode" not in eval_df.columns:
        return [("all", eval_df)]

    slices = []
    for mode in ("image-only", "general"):
        subset = eval_df[eval_df["model_mode"] == mode]
        if not subset.empty:
            slices.append((mode, subset))

    unknown = eval_df[eval_df["model_mode"] == "unknown"]
    if not unknown.empty:
        slices.append(("unknown", unknown))

    return slices or [("all", eval_df)]


def select_eval_df(
    eval_df: pd.DataFrame, *, mode: str, split_by_mode: bool
) -> list[tuple[str, pd.DataFrame]]:
    if mode != "mixed":
        subset = eval_df[eval_df["model_mode"] == mode]
        return [(mode, subset)]
    if split_by_mode:
        return iter_mode_slices(eval_df)
    return [("mixed", eval_df)]


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
        default="/data0/sebastian.cavada/compositional-physics/tiny_vqa_creation/output/",
    )
    parser.add_argument("--run-name", default="run_23_general_obj_num")
    parser.add_argument(
        "--mode",
        choices=["mixed", "general", "image-only"],
        default="mixed",
        help="Filter by model mode; mixed keeps all models.",
    )
    parser.add_argument(
        "--split-by-mode",
        action="store_true",
        help="Generate separate outputs per model mode when --mode=mixed.",
    )
    args = parser.parse_args()

    utils_graph.RUN_NAME = args.run_name
    utils_graph_correlation.RUN_NAME = args.run_name

    output_dir = Path("output_plots") / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_df = build_eval_df(args.base_path)
    eval_df = add_model_mode(eval_df)

    for mode_label, mode_df in select_eval_df(
        eval_df, mode=args.mode, split_by_mode=args.split_by_mode
    ):
        for x in range(1):
            create_num_objects_violin_grid(
                mode_df,
                group_by="model_id",
                save_per_category=True,
                per_category_dirname=f"num_objects_per_model_{mode_label}",
                save_grid=True,
                save_legend=True,
                legend_filename=f"num_objects_legend_models_{mode_label}.png",
                legend_cols=6,
                sample_frac=0.8,
                sample_seed=x,
                y_limit_mode="fixed",
                output_dir=output_dir,
            )

        create_num_objects_violin_grid(
            mode_df,
            group_by="family",
            save_per_category=True,
            per_category_dirname=f"num_objects_per_family_{mode_label}",
            save_grid=False,
            save_legend=True,
            legend_filename=f"num_objects_legend_families_{mode_label}.png",
            legend_cols=4,
            sample_frac=0.8,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()
