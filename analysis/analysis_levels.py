from __future__ import annotations

import argparse
from pathlib import Path
import re
import pandas as pd

from utils import (
    utils_read,
    utils_mapping,
    utils_graph,
    utils_graph_levels
)

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
    print("Models loaded:", eval_df["model_id"].unique())

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
