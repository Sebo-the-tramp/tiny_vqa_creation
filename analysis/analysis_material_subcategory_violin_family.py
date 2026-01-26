from __future__ import annotations

import argparse
from pathlib import Path

import utils.utils_graph as utils_graph
import utils.utils_graph_correlation as utils_graph_correlation
from analysis_material_subcategory_violin import (
    add_model_mode,
    build_eval_df,
    plot_subcategory_violin,
)


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
        "--top-category",
        default="material_understandgin",
        help="Top-level category to plot.",
    )
    parser.add_argument(
        "--exclude-top-categories",
        default="",
        help="Comma-separated list of top categories to exclude (e.g., temporal).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--title",
        default=None,
        help="Optional plot title; omit for no title.",
    )
    args = parser.parse_args()

    utils_graph.RUN_NAME = args.run_name
    utils_graph_correlation.RUN_NAME = args.run_name

    output_dir = Path("output") / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_df = build_eval_df(args.base_path)
    eval_df = add_model_mode(eval_df)
    if args.mode != "mixed":
        eval_df = eval_df[eval_df["model_mode"] == args.mode]

    safe_category = str(args.top_category).replace("/", "_").replace(" ", "_")
    output_path = output_dir / f"{safe_category}_subcategory_violin_{args.mode}_family.png"

    plot_subcategory_violin(
        eval_df,
        top_category=args.top_category,
        output_path=output_path,
        group_by="family",
        exclude_categories=[
            c for c in args.exclude_top_categories.split(",") if c.strip()
        ],
        seed=args.seed,
        title=args.title,
    )


if __name__ == "__main__":
    main()
