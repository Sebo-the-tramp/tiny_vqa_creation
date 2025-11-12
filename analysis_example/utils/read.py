import os
import json
import pandas as pd
import tqdm

# ROOT_DIR_SIMULATIONS = "/mnt/data1/sebastiancavada/datasets/3D_VQA/simulations"
# ROOT_DIR_SIMULATIONS = "/Users/sebastiancavada/Desktop/tmp_Paris/vqa/data/output/sims/dl3dv-hf-gso2/3-cg"
ROOT_DIR_SIMULATIONS = "/data/sebastian.cavada/compositional-physics/tiny_vqa_creation/output/results_tmp_test"

def read_results_test_and_gt(scene_path, run_name="results_tmp_test_run01_1K"):
    # result_dir = os.path.join(scene_path, "results_tmp_test_run01_1K")
    result_dir = os.path.join(scene_path, f"{run_name}/results_{run_name}")

    answers = []
    if not os.path.exists(result_dir):
        print(f"Results directory does not exist: {result_dir}")
        return answers

    for file_name in os.listdir(result_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(result_dir, file_name)
            result_model = {"model": file_name.replace("_val.json", "")}
            with open(file_path, 'r') as f:
                data = json.load(f)
                result_model["results"] = data
            answers.append(result_model)

    gt_path = os.path.join(scene_path, f"{run_name}/val_answer_{run_name}.json")
    print("GT path:", gt_path)
    gt = {}
    gt_dict = {}
    if os.path.exists(gt_path):
        with open(gt_path, 'r') as f:
            gt = json.load(f)        
        for item in gt:
            gt_dict[item['idx']] = item            
    else:
        print(f"Ground truth file does not exist: {gt_path}")

    test_path = os.path.join(scene_path, f"{run_name}/test_{run_name}.json")
    print("Test path:", test_path)
    test = {}
    test_dict = {}
    if os.path.exists(test_path):
        with open(test_path, 'r') as f:
            test = json.load(f)
        for item in test:
            test_dict[item['idx']] = item
    else:
        print(f"Test file does not exist: {test_path}")

    return answers, gt_dict, test_dict


# Function that works when index is correct
def merge_gt(answers, gt):    
    for model in answers:
        for answer in model["results"]:            
            qid = answer["idx"]
            if gt[qid]['idx'] == qid:
                answer["gt"] = gt[qid]['answer']                
            else:
                answer["gt"] = None
    return answers

# same thing here
def merge_test(answers, test):
    for model in answers:
        for answer in model["results"]:
            qid = answer["idx"]
            if test[qid]['idx'] == qid:
                answer["question"] = test[qid]['question']
                answer["question_id"] = test[qid]['question_id']
                answer["split"] = test[qid]['split']
                answer["scene_id"] = test[qid]['scene']
                answer["source"] = test[qid]['source']
                answer['images'] = test[qid]['file_name']
                answer["category"] = test[qid]['category']
                answer["sub_category"] = test[qid]['sub_category']
                answer["simulation_id"] = test[qid]['simulation_id']
                answer['mode'] = test[qid]['mode']                
            else:
                answer["question"] = None
    return answers

def normalize_choice(x):
    if x is None:
        return None
    x = str(x).strip().upper()
    return x[0] if x and x[0] in "ABCD" else "Wrong"

def load_from_model_records(model_records: list[dict]):
    """
    Input: a list of dicts like the one you pasted, one per model:
      {
        'model': 'InternVL2-2B',
        'mode': 'general',
        'results': [{...}, {...}, ...],
        'family': 'InternVLChat2',
        'source': 'OpenGVLab/InternVL2-2B',
        'params_b': 2.0,
        'release_year': 2024,
        'license': 'mit',
        'notes': '',
        'tags': [],
        'release_type': 'open_weigths'
      }
    Output: items_df, preds_df, models_df, eval_df
    """
    # ---- models table
    models_df = pd.json_normalize(model_records)
    if "model" in models_df.columns:
        models_df = models_df.rename(columns={"model":"model_id"})

    # Coerce numerics early (handles "7B", "2.7 b", etc.)
    def _parse_params_b(x):
        if pd.isna(x):
            return pd.NA
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        # Extract leading number and optional "B/b"
        # e.g. "7B", "2.7 b", "Params: 13B", "13.0"
        import re
        m = re.search(r'([0-9]*\.?[0-9]+)\s*[bB]?', s)
        return float(m.group(1)) if m else pd.NA

    if "params_b" in models_df.columns:
        models_df["params_b"] = models_df["params_b"].map(_parse_params_b).astype("Float32")

    if "release_year" in models_df.columns:
        models_df["release_year"] = pd.to_numeric(models_df["release_year"], errors="coerce").astype("Int16")


    keep_cols = ["model_id","family","source","params_b","release_year","license","notes","release_type", "mode"]
    models_df = models_df[[c for c in keep_cols if c in models_df.columns]].drop_duplicates("model_id")

    # ---- items & predictions (normalize nested results)
    item_rows = {}   # keyed by idx to dedupe across models
    pred_rows = []
    for m in tqdm.tqdm(model_records):
        model_id = m.get("model") or m.get("model_id")
        run_id = m.get("run_id", "default")
        for r in m["results"]:
            idx = r["idx"]

            # Build items (only once per idx; sanity check GT consistency)
            if idx not in item_rows:
                item_rows[idx] = {
                    "idx": idx,
                    "split": r.get("split"),
                    "question": r.get("question"),
                    "question_id": r.get("question_id"),
                    "file_name": r.get("images"),
                    "mode": r.get("mode") or r.get("mode"),
                    "scene": r.get("scene_id") or r.get("scene"),
                    "object": r.get("object"),
                    "source": r.get("source"),
                    "difficulty": r.get("difficulty"),
                    "ability_type": r.get("ability_type"),
                    "category": r.get("category"),
                    "sub_category": r.get("sub_category"),
                    "correct_answer": normalize_choice(r.get("gt")),
                    "num_objects": r.get("num_objects"),
                }
            else:
                prev = item_rows[idx]["correct_answer"]
                cur = normalize_choice(r.get("gt"))
                if prev is not None and cur is not None and prev != cur:
                    raise ValueError(f"Inconsistent ground-truth for idx={idx}: {prev} vs {cur}")

            # Build predictions
            pred_rows.append({
                "model_id": model_id,
                "idx": idx,
                "answer": normalize_choice(r.get("answer")),
                "run_id": run_id,
            })

    items_df = pd.DataFrame.from_records(list(item_rows.values()))
    preds_df = pd.DataFrame.from_records(pred_rows)

    # ---- join + correctness
    eval_df = preds_df.merge(items_df, on="idx", how="left")
    eval_df["is_correct"] = (eval_df["answer"] == eval_df["correct_answer"])
    eval_df = eval_df.merge(models_df, on="model_id", how="left")

    # Ensure numeric types survive merges
    if "params_b" in eval_df.columns:
        eval_df["params_b"] = pd.to_numeric(eval_df["params_b"], errors="coerce").astype("Float32")
    if "release_year" in eval_df.columns:
        eval_df["release_year"] = pd.to_numeric(eval_df["release_year"], errors="coerce").astype("Int16")

    # memory‑friendly dtypes
    for col in ["model_id","split","mode","scene","difficulty","ability_type","category","sub_category","correct_answer","answer", "question_id"]:
        if col in eval_df.columns:
            eval_df[col] = eval_df[col].astype("category")

    return items_df, preds_df, models_df, eval_df

simulations_metadata_cache = {}

# here we shall add all the simulation metadata that we need/want
def merge_sim_metadata(answers_vlm, mapping_fct=None):
    pbar = tqdm.tqdm(range(len(answers_vlm)))
    for m_i in pbar:
        model = answers_vlm[m_i]
        for a_i in range(len(model["results"])):
            answer = model["results"][a_i]
            pbar.set_description(f"answer: {a_i}/{len(model["results"])}")
            simulation_path = answer["simulation_id"]
            if mapping_fct is not None:
                simulation_path = mapping_fct(simulation_path)
            # print("simulation_id", simulation_id)

            # Check cache first
            if simulation_path in simulations_metadata_cache:
                # print(f"Using cached metadata for simulation: {simulation_id}")
                sim_metadata = simulations_metadata_cache[simulation_path]
            else:
                # print("Cache miss for simulation:", simulation_id)
                sim_path = os.path.join(ROOT_DIR_SIMULATIONS, simulation_path)
                if os.path.exists(sim_path):
                    # print(f"Loading simulation metadata from: {sim_path}")
                    with open(sim_path, 'r') as f:
                        sim_metadata = json.load(f)
                    # Store in simulations_metadata_cache
                    simulations_metadata_cache[simulation_path] = sim_metadata
                else:
                    raise FileNotFoundError(f"Simulation metadata file not found: {sim_path}")
            
            answer['num_objects'] = len(sim_metadata["objects"].keys())
    
    return answers_vlm

