from __future__ import annotations

import argparse
from fileinput import filename
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import tqdm

import utils.utils_read

if not Path("/scratch/project/eu-25-92/composite_physics/dataset/simulation_v4/").exists():
    utils.utils_read.sim_path_fct = lambda x: x.replace("/scratch/project/eu-25-92/composite_physics/dataset/simulation_v4/", "/nfs/data/workspaces/rdechare/codes/physics-sim/output/sims/v4/")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-path",
        default="../output",
    )
    parser.add_argument("--run-name", default="run_26_general")
    parser.add_argument(
        "--vqa-set",
        default="10K",
        help="VQA set to use (e.g., 10K, 30K, karo_5K).",
    )
    parser.add_argument(
        "--mode",
        default='range',
        help="Mode to split the VQA set (range, sample, cumulative).",
    )
    parser.add_argument(
        "--sampling",
        default=0.5,
        type=float,
        help="Fraction of the VQA set to use (e.g., 0.5 for 50%).",
    )
    parser.add_argument(
        "--num",
        default=2,
        type=int,
        help="Number of splits to create from the VQA set.",
    )
    parser.add_argument(
        "--skip-existing",
        default=False,
        action="store_true",
        help="Skip existing splits if they already exist.",
    )
    args = parser.parse_args()
    assert args.mode in ['range', 'sample', 'cumulative'], "Mode must be one of 'range', 'sample', 'cumulative'."
    

    test_path = Path(args.base_path) / args.run_name / f"test_{args.run_name}_{args.vqa_set}.json"
    df = utils.utils_read._read_json_dataframe(test_path)

    print(f"Entries in {test_path}: {len(df)}")

    questions = df["question_id"].unique()
    print(f"Questions in df ({len(questions)}):")

    questions_idx = {}
    for q in questions:
        questions_idx[q] = df.loc[df["question_id"] == q, "idx"].unique()
        print(f"  '{q}' unique idxs ({len(questions_idx[q])}):")
    
    for s in range(args.num):
        print(f"\n\n\nCreating new split {s+1}/{args.num} with mode '{args.mode}' and sampling fraction {args.sampling}:")
        split_df = df.copy()

        if args.mode == 'range':
            suffix = f"-r{args.num}-s{s}"
        elif args.mode == 'cumulative':
            suffix = f"-c{args.num}-s{s}"
        elif args.mode == 'sample':
            suffix = f"-s{args.sampling}-s{s}"

        split_path = str(test_path.with_suffix("")) + suffix + ".json"
        pkl_path = test_path.parent / f"merged_results_{args.vqa_set}" + suffix + ".pkl"
        
        if args.skip_existing and Path(split_path).exists():
            print(f"  Skipping existing split {split_path}")
            continue

        for q in questions:
            q_idxs = questions_idx[q]

            if args.mode == 'range':
                q_s, q_e = int(len(q_idxs) * s / args.num), int(len(q_idxs) * (s + 1) / args.num)
                sampled_q_idxs = q_idxs[q_s:q_e]
                print(f"  '{q}' idxs deterministic {q_s}:{q_e} ({len(sampled_q_idxs)})")
                # print({sampled_q_idxs})
            elif args.mode == 'cumulative':
                q_e = int(len(q_idxs) * (s + 1) / args.num)
                sampled_q_idxs = q_idxs[:q_e]
                print(f"  '{q}' idxs cumulative up to {q_e} ({len(sampled_q_idxs)})")
                # print({sampled_q_idxs})
            elif args.mode == 'sample':
                sampled_q_idxs = pd.Series(q_idxs).sample(frac=args.sampling, random_state=s).tolist()
                print(f"  '{q}' idxs sampled ({len(sampled_q_idxs)})")
                # print({sampled_q_idxs})
                
                split_df = split_df[~((split_df["question_id"] == q) & (~split_df["idx"].isin(sampled_q_idxs)))]
        
        print(f"\nSaving split df to {split_path}")
        print(f"Entries in {split_path}: {len(split_df)}")
        split_df.to_json(
            split_path,
            orient="records",   # list of objects: [ {...}, {...} ]
            lines=False,        # not JSONL
            indent=4,           # pretty format like original
            force_ascii=False,
        )

        # If cache exists, delete it
        if pkl_path.exists():
            print(f"Deleting existing cache at {pkl_path} for split {split_path}")
            pkl_path.unlink()
        
        # If merged cache exists, delete it
        pkl_meta_path = pkl_path.parent / f"merged_results_{args.vqa_set}_vqa-split-{args.mode}.pkl"
        if pkl_meta_path.exists():
            print(f"Deleting existing cache at {pkl_meta_path}")
            pkl_meta_path.unlink()
            


if __name__ == "__main__":
    main()
