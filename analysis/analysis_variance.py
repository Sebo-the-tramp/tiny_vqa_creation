from __future__ import annotations

import argparse
import contextlib
from fileinput import filename
from multiprocessing.pool import Pool
from multiprocessing import Manager
from pathlib import Path

import pandas as pd
import tqdm

import utils.utils_read

if not Path("/scratch/project/eu-25-92/composite_physics/dataset/simulation_v4/").exists():
    utils.utils_read.sim_path_fct = lambda x: x.replace("/scratch/project/eu-25-92/composite_physics/dataset/simulation_v4/", "/nfs/data/workspaces/rdechare/codes/physics-sim/output/sims/v4/")

# from utils.utils_read import load_results, _sanitize_answer
import utils.utils_graph as utils_graph
import utils.utils_graph_correlation as utils_graph_correlation
import utils.utils_graph_variance

def run_load_results(base_path, run_name, vqa_set, sets_to_load):
    df = utils.utils_read.load_results(
        base_path,
        run_folder=run_name,
        merge_model_answers=True,
        model_answers_wide=True,
        cache=True,
        add_sim_metadata=True,
        vqa_set=vqa_set
    )
    sets_to_load.remove(vqa_set)
    print(f"Loaded VQA set: {vqa_set}. Remaining sets to load: {len(sets_to_load)}.")
    return df

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-path",
        default="../output",
    )
    parser.add_argument("--run-name", default="run_26_general")
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
    parser.add_argument(
        "--vqa-set",
        default="30K",
        help="VQA set to use (e.g., 10K, 30K, karo_5K).",
    )
    parser.add_argument(
        "--vqa-split-mode",
        default='sample',
        help="Mode to split the VQA set (range, sample, cumulative).",
    )
    args = parser.parse_args()

    utils_graph.RUN_NAME = args.run_name
    utils_graph_correlation.RUN_NAME = args.run_name

    output_dir = Path("output") / args.run_name / "variance" / f"{args.vqa_set}_{args.vqa_split_mode}"
    output_dir.mkdir(parents=True, exist_ok=True)

    mode_letter = args.vqa_split_mode[0]
    path = Path(args.base_path) / args.run_name / f"test_{args.run_name}_{args.vqa_set}-{mode_letter}*.json"
    merged_path = Path(args.base_path) / args.run_name / f"merged_results_{args.vqa_set}_vqa-split-{args.vqa_split_mode}.pkl"
    if not merged_path.exists():
        globbed_files = list(path.parent.glob(path.name))
        assert globbed_files, f"No files found for pattern: {path}"
        
        all_vqa_sets = [args.vqa_set]  # Always include the originally specified VQA set
        if not globbed_files:
            print(f"No files found for pattern: {path}")
        else:
            print(f"Found {len(globbed_files)} files for pattern: {path}")
            for f in globbed_files:
                vqa_set = f.stem.split(f"test_{args.run_name}_")[-1]
                all_vqa_sets.append(vqa_set)

        print(f"{len(all_vqa_sets)} available VQA sets for '{args.run_name}': {all_vqa_sets}")
        manager = Manager()
        sets_to_load = manager.list(all_vqa_sets)
        if all_vqa_sets:
            workers = 4
            print(f"Loading {len(all_vqa_sets)} VQA pickles with {workers} workers in parallel..")
            with Pool(processes=workers) as pool:
                pool.starmap(run_load_results, [(args.base_path, args.run_name, vqa_set, sets_to_load) for vqa_set in all_vqa_sets])
        

        print(f"\n\nLoading VQA dataframes..")
        all_vqa_sets_dfs = []
        for vqa_set in all_vqa_sets:
            vqaset_df = utils.utils_read.build_eval_df(args.base_path, vqa_set=vqa_set)
            vqaset_df["vqa_set"] = vqa_set
            vqaset_df["vqa_set_count"] = vqaset_df["idx"].nunique()  # Count unique questions per vqa_set
            
            print(f"Loaded VQA dataframe for set: {vqa_set}. Rows: {len(vqaset_df)}, Unique questions: {vqaset_df['vqa_set_count'].iloc[0]}")
            all_vqa_sets_dfs.append(vqaset_df)
        
        print(f"\n\nMerging all dataframes..")
        merged_df = pd.concat(all_vqa_sets_dfs, ignore_index=True)

        merged_df.to_pickle(merged_path)
        print(f"Merged dataframe saved to {merged_path}. Total rows: {len(merged_df)}")
    
    print(f"Loading merged dataframe from {merged_path}..")
    merged_df = pd.read_pickle(merged_path)
    print(f"Merged dataframe loaded. Total rows: {len(merged_df)}")
        
    for mode_label, mode_df in utils.utils_read.select_eval_df(
        merged_df, mode=args.mode, split_by_mode=args.split_by_mode
    ):
        for group in ["model_family", "model_id", "model_best"]:
            if group == "model_best":
                # Compute the per model accuracy and keep only best overall model per family
                model_accuracy = mode_df.groupby(['model_family', 'model_id'])['accuracy'].mean().reset_index()
                best_models = model_accuracy.loc[model_accuracy.groupby('model_family')['accuracy'].idxmax()]
                cur_df = mode_df[mode_df['model_id'].isin(best_models['model_id'])]
                
                group_by = "model_id"
            else:
                cur_df = mode_df
                group_by = group
            
            print(f"Processing mode: {mode_label}, grouping by {group_by}: with {len(cur_df)} entries")
            for cat in list(cur_df["category"].unique()) + ["all"]:
                print(f"  Category: {cat}, entries: {len(cur_df[cur_df['category'] == cat])}")

                utils.utils_graph_variance.create_variance_curve(
                    cur_df,
                    output_dir=output_dir / mode_label,
                    filename= f"var_{group}"+(f"_{cat}" if cat != "all" else "")+".png",
                    y_limit_mode="",
                    group_by=group_by,
                    category=cat,
            )


if __name__ == "__main__":
    main()
