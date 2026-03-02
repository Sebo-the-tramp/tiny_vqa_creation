import json
from pathlib import Path
import re
import time
from typing import List
import tqdm

import pandas as pd

from utils import utils_mapping
from utils import utils_graph


SIM_PATH_MODIFIER = lambda x: x.replace("simulation.json", "simulation_kinematics_min.json")
if not Path("/scratch/project/eu-25-92/composite_physics/dataset/simulation_v4/").exists():
    SIM_PATH_MODIFIER = lambda x: x.replace("/scratch/project/eu-25-92/composite_physics/dataset/simulation_v4/", "/nfs/data/workspaces/rdechare/codes/physics-sim/output/sims/v4/")

# _ANSWER_RE = re.compile(r"(?i)^\s*([a-d])(?:[^a-z0-9]|$)")
# _ANSWER_RE = re.compile(r"\b([A-D])\s*[\.\,\:\)]")
_ANSWER_RE = re.compile(r"(?:^([A-D])\b|\b([A-D])\b\s*[\.\,\:\)]?$)", re.IGNORECASE)
_IDX_RE = r"([0-9]+_[gi])"
_LEVEL_RE = r"([0-9]+_[gi])(_level_([^_]+))"

try:
    import orjson
except ImportError as exc:
    raise ImportError(
        "orjson is required for streaming simulation.json; install it first."
    ) from exc

_SIM_METADATA_CACHE: dict[str, dict] = {}
TIMESTART = 0.01
SAMPLING_RATE = 25
RENDER_STEP = 1.0 / SAMPLING_RATE

def _load_model_metadata(metadata_path: str | Path = "utils/metadata.json") -> pd.DataFrame:
    path = Path(metadata_path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    if "id" in df.columns:
        df = df.rename(columns={"id": "model_id"})
    return df

def _assert_all_model_ids_in_metadata(
    eval_df: pd.DataFrame, metadata_df: pd.DataFrame
) -> None:
    eval_ids = set(eval_df["model_id"].dropna().unique())
    metadata_ids = set(metadata_df["model_id"].dropna().unique())
    missing = sorted(eval_ids - metadata_ids)
    if missing:
        preview = ", ".join(missing[:10])
        suffix = "" if len(missing) <= 10 else f" ... (+{len(missing) - 10} more)"
        raise KeyError(
            "model_id values missing from metadata.json: "
            f"{preview}{suffix}"
        )

def _curate_invalid_and_unanswered(eval_df: pd.DataFrame) -> pd.DataFrame:
    # Remove "general" VQA (ie, video questions) for image-only models (impossible)
    incompatible = (eval_df["mode_test"]=="general") & (eval_df["model_mode"]=="image-only")
    eval_df = eval_df[~incompatible]
    
    # Filter models with 100% empty answers "" (we assume the model crashed)
    models_empty = (
        eval_df.groupby(["model_family", "model_id"], observed=True)["model_answer"]
        .apply(lambda s: s.eq("").all())
    )
    if models_empty.sum() > 0:
        invalid_df = models_empty[models_empty].index.tolist()
        for fam, mid in invalid_df:
            print(f"[WARNING] Removing model {fam}, {mid} having ONLY empty answers.")

        eval_df = eval_df[~eval_df.set_index(["model_family", "model_id"]).index.isin(invalid_df)]

    # Check for missing answers (missing means the question idx was not found in the model's results json)
    missing_answers = eval_df["model_answer"].isna()
    if missing_answers.sum() > 0:
        print(f"[WARNING] Missing {missing_answers.sum()} / {len(eval_df)} model answers:")
        for model_id, group in eval_df[missing_answers].groupby("model_id"):
            print(f"  {model_id}: {len(group)} missing answers") # (idx: {group['idx'].tolist()})")
        
        eval_df = eval_df[~missing_answers]

    return eval_df

def build_eval_df(  
        run_name: str,
        base_path: str | Path, 
        vqa_set: str = "10K",
        metadata_path: str | Path = "utils/metadata.json",
        return_paths: dict | None = None,
        columns: list[str] = [],  # Columns to preserve
        cache: bool = True,
        *,
        excluded_questions: list[str] = ["F_OCCLUSION_PERCENTAGE_OBJECT", "F_MATERIAL_IDENTIFICATION_OBJECT_LEVEL_3", "F_CAMERA_ZOOM_BEHAVIOR", "F_FOCAL_LENGTH_CLASS"],
        # could also be excluded: F_MATERIAL_IDENTIFICATION_OBJECT_LEVEL_2
        # could also be excluded: F_CAMERA_ZOOM_BEHAVIOR
        exclude_models: list[str] = ["MiniCPM-V2.5"]
    ) -> pd.DataFrame:
    base = Path(base_path)

    run_folder = Path(run_name)

    df = load_results(
        base,
        run_folder=run_folder,
        merge_model_answers=True,
        model_answers_wide=True,
        cache=cache,
        add_sim_metadata=True,
        vqa_set=vqa_set,
        return_paths=return_paths,
    )


    
    results_dir = base / run_folder / f"results_{run_folder}"
    model_cols = sorted(
        p.stem.replace("_val", "") for p in results_dir.glob("*_val.json")
    )
    model_cols = [c for c in model_cols if c in df.columns]
    if not model_cols:
        raise ValueError(f"No model answer columns found in {results_dir}")

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
            "scene",
            "source",
        ] + columns
        if c in df.columns
    ]

    eval_df = df.melt(
        id_vars=id_cols,
        value_vars=model_cols,
        var_name="model_id",
        value_name="model_answer",
    )
    assert set(df["answer"].unique()) <= {"A", "B", "C", "D"}, "Error, answers should in: {'A', 'B', 'C', 'D'}"

    metadata_df = _load_model_metadata(metadata_path=metadata_path)
    _assert_all_model_ids_in_metadata(eval_df=eval_df, metadata_df=metadata_df)

    for col in ["family", "params_b", "release_year", "mode", "priority"]:
        col_map = metadata_df.set_index("model_id")[col].to_dict()
        eval_df["model_"+col] = eval_df["model_id"].map(col_map)
    
    # Remove model that may have crashed or question invalid (eg, video question for image-only model)
    eval_df = _curate_invalid_and_unanswered(eval_df)

    # Verify all answers are either valid (A-D), empty ('') or invalid ('?' => typically, when model answers jiberish instead of A-D)
    assert set(eval_df["model_answer"].dropna().unique()) <= {"A", "B", "C", "D", "?", ""}, "Models answers should be in: {'A', 'B', 'C', 'D', '?', ''}. Values found: " + str(eval_df["model_answer"].unique())
    
    valid = eval_df["model_answer"].notna()
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
    
    if excluded_questions:
        print(f"[WARNING] Excluding questions: {excluded_questions} => Dropping {len(eval_df[eval_df['question_id'].isin(excluded_questions)])} entries.")
        eval_df = eval_df.loc[~eval_df["question_id"].isin(excluded_questions), :].copy()  # Important to copy, to avoid caveats in pandas slices
    if exclude_models:
        print(f"[WARNING] Excluding models: {exclude_models} => Dropping {len(eval_df[eval_df['model_id'].isin(exclude_models)])} entries.")
        eval_df = eval_df.loc[~eval_df["model_id"].isin(exclude_models), :].copy()  # Important to copy, to avoid caveats in pandas slices
    
    # Convert accuracy column
    assert any(col in eval_df.columns for col in ["accuracy", "is_correct"]), "Expected 'accuracy' or 'is_correct' column in eval_df"
    acc_column = "accuracy" if "accuracy" in eval_df.columns else "is_correct"
    eval_df["accuracy"] = pd.to_numeric(eval_df[acc_column], errors="coerce")
    
    missing_acc = eval_df["accuracy"].isna()
    if missing_acc.sum() > 0:
        total_rows = len(eval_df)
        print(
            f"[WARNING] Dropping {missing_acc.sum()} / {total_rows} rows with NaN in accuracy"
        )

    eval_df = eval_df[~missing_acc]

    # Fix some incorrect mappings
    object_identity_qids = ("F_PERSISTENCE_OBJECT_PRESENT", "F_PERSISTENCE_OBJECT_DISAPPEAR")
    eval_df.loc[eval_df["question_id"].isin(object_identity_qids), "sub_category"] = "object_identity"
    
    object_count_qids = ("F_PERSISTENCE_OBJECT_TOTAL_COUNT", "F_PERSISTENCE_OBJECT_TOTAL_COUNT_HIDDEN")
    eval_df.loc[eval_df["question_id"].isin(object_count_qids), "sub_category"] = "object_count"
    
    # Assert and update mappings: category <=> subcategory <=> question
    questions_map = (
        eval_df[["question_id", "sub_category", "category"]]
        .drop_duplicates("question_id")
        .set_index("question_id")
    )
    for qid, row in questions_map.iterrows():
        subcat = row["sub_category"]
        cat = row["category"]

        # Verify mapping is consistent
        assert utils_mapping.question_to_subcat.get(qid, subcat) == subcat, f"Inconsistent mapping for question {qid}: previously mapped to subcat {utils_mapping.question_to_subcat[qid]}, now getting {subcat}"
        assert utils_mapping.subcat_to_cat.get(subcat, cat) == cat, f"Inconsistent mapping for subcategory {subcat}: previously mapped to category {utils_mapping.subcat_to_cat[subcat]}, now getting {cat}"
        
        # Verify keys exist in the main mapping dictionaries
        assert cat in utils_mapping.categories, f"Category {cat} not in predefined categories utils_mapping.categories: {utils_mapping.categories}"
        assert cat in utils_mapping.mapping_cat_colors, f"Category {cat} not in predefined categories utils_mapping.mapping_cat_colors: {utils_mapping.mapping_cat_colors}"
        assert cat in utils_mapping.mapping_cat_short, f"Category {cat} not in predefined categories utils_mapping.mapping_cat_short: {utils_mapping.mapping_cat_short}"
        assert subcat in utils_mapping.subcategories, f"Subcategory {subcat} not in predefined subcategories utils_mapping.subcategories: {utils_mapping.subcategories}"

        # Populate the mappings
        utils_mapping.question_to_subcat[qid] = subcat
        utils_mapping.subcat_to_cat[subcat] = cat

    # Assert family mappings
    family = eval_df["model_family"].unique()
    for fam in family:
        assert fam in utils_mapping.family_marker, f"Family {fam} not in predefined families utils_mapping.family_marker: {utils_mapping.family_marker}"
    
    print(f"Loaded {eval_df['idx'].nunique()} VQA ({eval_df['question_id'].nunique()} questions types) with {len(eval_df)} answers, {eval_df['model_id'].nunique()} models, {eval_df['model_family'].nunique()} families.")
    return eval_df

def load_results(
    base_path: str | Path,
    run_folder: str | None = None,
    drop_cols: list[str] | None = None,
    keep_cols: list[str] | None = None,
    add_sim_metadata: bool = False,
    sim_path_col: str = "simulation_id",
    merge_model_answers: bool = False,
    model_answers_wide: bool = True,
    model_results_dir: str | Path | None = None,
    cache: bool = True,
    cache_path: str | Path | None = None,
    vqa_set: str = "10K",
    return_paths: dict | None = None,
) -> pd.DataFrame:
    base = Path(base_path)

    test_path = base / run_folder / f"test_{run_folder}_{vqa_set}.json"
    val_path = base / run_folder / f"val_answer_{run_folder}.json"

    if cache_path is None:
        cache_path = base / run_folder / f"merged_results_{vqa_set}.pkl"
    else:
        cache_path = Path(cache_path)

    if return_paths is not None:
        return_paths["val"] = val_path
        return_paths["test"] = test_path
        return_paths["cache"] = cache_path
    
    if cache and cache_path.exists():
        print("Cache found at ", cache_path)
        df_cached = _load_cached_df(cache_path)
        required_cols = []
        if add_sim_metadata:
            required_cols.append("object_count")
            required_cols.append("visible_pixels_interested_object")
        if merge_model_answers:
            results_dir = (
                Path(model_results_dir)
                if model_results_dir is not None
                else base / run_folder / f"results_{run_folder}"
            )
            required_cols.extend(
                p.stem.replace("_val", "") for p in results_dir.glob("*_val.json")
            )
        missing = [col for col in required_cols if col not in df_cached.columns]
        if len(missing) > 0:
            print("[WARNING] Models missing:", missing)
        
        return df_cached

            # reply = input("There are missing columns in the cache. Proceed or reload? (y=use cache, n=reload): ").strip().lower()
            # if reply == "y":
            #     return df_cached

    df_test = _read_json_dataframe(test_path)
    df_val = _read_json_dataframe(val_path)

    if any(df_test["idx"].apply(lambda x: re.fullmatch(_LEVEL_RE, x))):
        print("Level answers detected. Duplicating val to match models idx.")
        levels_string = df_test["idx"].apply(lambda x: re.fullmatch(_LEVEL_RE, x).groups()[1])
        levels_unique = levels_string.unique()
        
        df_val = pd.concat([df_val.assign(idx=df_val["idx"] + level) for level in levels_unique], ignore_index=True)



    # FOR AGENT Keep the hardcoded columns #
    # print("Processing columns...")
    # 
    # drop_cols = ["scene", "source"]
    # if drop_cols and keep_cols:
    #     raise ValueError("Use only one of drop_cols or keep_cols.")
    # if keep_cols is not None:
    #     df_test = df_test[keep_cols]
    # elif drop_cols is not None:
    #     df_test = df_test.drop(columns=drop_cols, errors="ignore")

    # Keep all test questions; val answers may include extra idx.
    df = df_test.merge(df_val, on="idx", how="left", suffixes=("_test", "_val"))

    if add_sim_metadata:
        if sim_path_col not in df.columns:
            raise KeyError(f"Column '{sim_path_col}' not found in merged DataFrame.")
        
        pbar = tqdm.tqdm(print("Reading scenes simulation..."), total=len(df))
            
        def read_scene_metadata(p):
            pbar.update(1)
            return read_simulation_metadata(str(p))
        
        df["object_count"] = df[sim_path_col].apply(
            lambda p: read_scene_metadata(p)["object_count"]
        )

        # here we can add the material of the simulation too
        df['object-yms'] = df[sim_path_col].apply(
            lambda s: re.search(r'(?<!\w)(stiff|soft|medium)(?!\w)', s).group(1)
            if re.search(r'(?<!\w)(stiff|soft|medium)(?!\w)', s)
            else "mixed"
        )

        # if the interested objects are 2  we take the average of the visible pixels
        df["visible_pixels_interested_object"] = df.apply(
            lambda row: find_interested_object_pixels_count(
                str(row[sim_path_col]),
                str(row["question"]),
                row["file_name"],
            )["visible_pixels_interested_object"],
            axis=1,
        )

    print("Merging model answers...")
    # print(df.head().to_string())

    if merge_model_answers:
        results_dir = (
            Path(model_results_dir)
            if model_results_dir is not None
            else base / run_folder / f"results_{run_folder}"
        )
        assert results_dir.exists(), f"Model results directory not found: {results_dir}"
        df_models = load_model_answers(results_dir, wide=model_answers_wide)
        if model_answers_wide:
            df = df.merge(df_models, on="idx", how="left")
        else:
            raise ValueError("model_answers_wide=False not supported in load_results.")

    if cache:
        _save_cached_df(df, cache_path)

    return df

def load_results_levels(
    base_path: str | Path,
    run_folder: str | None = None,
    drop_cols: list[str] | None = None,
    keep_cols: list[str] | None = None,
    add_sim_metadata: bool = False,
    sim_path_col: str = "simulation_id",
    merge_model_answers: bool = False,
    model_answers_wide: bool = True,
    model_results_dir: str | Path | None = None,
    cache: bool = True,
    cache_path: str | Path | None = None,
    vqa_set: str = "10K"
) -> pd.DataFrame:
    base = Path(base_path)
    assert False, "load_results_levels is not implemented yet. It should be similar to load_results but with additional processing to handle different levels of the dataset (e.g., question difficulty levels). You can start by copying load_results and then modifying it to include the necessary logic for handling levels."

    test_path = base / run_folder / f"test_{run_folder}_{vqa_set}.json"
    val_path = base / run_folder / f"val_answer_{run_folder}.json"

    print(f"Loading test data from: {test_path}")
    print(f"Loading val data from: {val_path}")


    if cache_path is None:
        cache_path = (
            base / run_folder / f"merged_results_{vqa_set}.pkl"
            if run_folder
            else base / f"merged_results_{vqa_set}.pkl"
        )
    else:
        cache_path = Path(cache_path)

    if cache and cache_path.exists():
        df_cached = _load_cached_df(cache_path)
        required_cols = []
        if add_sim_metadata:
            required_cols.append("object_count")
        if merge_model_answers:
            results_dir = (
                Path(model_results_dir)
                if model_results_dir is not None
                else base / run_folder / f"results_{run_folder}_{vqa_set}"
            )
            required_cols.extend(
                p.stem.replace("_val", "") for p in results_dir.glob("*_val.json")
            )
        if all(col in df_cached.columns for col in required_cols):
            return df_cached

    df_test = _read_json_dataframe(test_path)
    df_val = _read_json_dataframe(val_path)

    # FOR AGENT Keep the hardcoded columns #
    print("Processing columns...")

    drop_cols = ["scene", "source", "file_name"]

    if drop_cols and keep_cols:
        raise ValueError("Use only one of drop_cols or keep_cols.")
    if keep_cols is not None:
        df_test = df_test[keep_cols]
    elif drop_cols is not None:
        df_test = df_test.drop(columns=drop_cols, errors="ignore")

    def _base_idx(val: object) -> str:
        text = str(val)
        parts = text.split("_")
        return "_".join(parts[:2]) if len(parts) >= 2 else text

    df_test["idx_base"] = df_test["idx"].apply(_base_idx)
    df_val["idx_base"] = df_val["idx"].apply(_base_idx)

    # Keep all test questions; val answers may include extra idx.
    df = df_test.merge(df_val, on="idx_base", how="left", suffixes=("_test", "_val"))
    df = df.rename(columns={"idx_test": "idx"})

    if add_sim_metadata:
        if sim_path_col not in df.columns:
            raise KeyError(f"Column '{sim_path_col}' not found in merged DataFrame.")
        df["object_count"] = df[sim_path_col].apply(
            lambda p: read_simulation_metadata(str(p))["object_count"]
        )

    if merge_model_answers:
        results_dir = (
            Path(model_results_dir)
            if model_results_dir is not None
            else base / run_folder / f"results_{run_folder}_{vqa_set}"
        )
        df_models = load_model_answers(results_dir, wide=model_answers_wide)
        if model_answers_wide:
            df = df.merge(df_models, on="idx", how="left")
        else:
            raise ValueError("model_answers_wide=False not supported in load_results.")

    if cache:
        _save_cached_df(df, cache_path)

    return df

def macro_accuracy(df: pd.DataFrame, level: str, group_by: list[str]|str = None) -> pd.DataFrame:
    levels = ["question_id", "sub_category", "category", "model_id", "model_family"]
    if "run_name" in df.columns:
        levels = levels + ["run_name"]
    assert level in levels, "Invalid value for 'level' argument"

    if group_by is not None:
        group_by = [group_by] if isinstance(group_by, str) else group_by
        levels = levels + [g for g in group_by if g not in levels]
        

    level_df = df
    # Compute a level accuracy by iteratively averaging the higher levels (eg, for level=category, we first average by question, then sub_category, then category)
    for level_idx, level_name in enumerate(levels):
        level_df = (
            level_df.groupby(
                levels[level_idx:]
                # ["run_name", "model_family", "model_id", "category", "sub_category", "question_id"]
            )["accuracy"]
            .agg(
                accuracy="mean",
                accuracy_min="min",
                accuracy_max="max",
                accuracy_std="std",
            )
            .reset_index()
        )
        if level_name == level:
            return level_df

    return None

def read_simulation_metadata(
    simulation_json_path: str | Path, verbose: bool = False
) -> dict:
    simulation_json_path = Path(SIM_PATH_MODIFIER(simulation_json_path))
    
    cache_key = str(simulation_json_path)
    cached = _SIM_METADATA_CACHE.get(cache_key)
    if cached is not None:
        if verbose:
            print(f"cache hit: {cache_key}")
        return cached
    if verbose:
        print(f"cache miss: {cache_key}")

    object_count = 0

    # Stream only the objects list; ijson will stop as soon as the file ends.
    with simulation_json_path.open("rb") as f:
        data = orjson.loads(f.read())
        object_count = len(data["objects"])

    result = {
        "object_count": object_count,
        "objects": data["objects"],
        "simulation": data["simulation"],
    }
    _SIM_METADATA_CACHE[cache_key] = result
    return result


def find_interested_object_pixels_count(
    simulation_json_path: str | Path,
    question: str,
    file_names: List,
    verbose: bool = False,
) -> dict:
    last_file_name = file_names[-1]
    render_name = Path(last_file_name).stem.split("_")[0]  # We split by "_" to handle ablation filenames (eg, "001211_F_MASS_make-a-match.png")
    final_timestep = get_timestep_from_idx(int(render_name))

    simulation_json_path = Path(SIM_PATH_MODIFIER(simulation_json_path))
    cache_key = str(simulation_json_path)
    cached = _SIM_METADATA_CACHE.get(cache_key)
    if cached is not None:
        if verbose:
            print(f"cache hit: {cache_key}")
    else:
        if verbose:
            print(f"cache miss: {cache_key}")

        # Stream only the objects list; ijson will stop as soon as the file ends.
        with simulation_json_path.open("rb") as f:
            data = orjson.loads(f.read())
            object_count = len(data["objects"])

        result = {
            "object_count": object_count,
            "objects": data["objects"],
            "simulation": data["simulation"],
        }
        _SIM_METADATA_CACHE[cache_key] = result
        cached = result

    # find the interested objects from the question
    for object_id, obj in cached["objects"].items():
        if question.find(str(obj["description"]["object_name"])) != -1:
            # print(
            #     f"Found interested object: {obj['description']['object_name']} with id {object_id}"
            # )

            return {
                "visible_pixels_interested_object": cached["simulation"][
                    final_timestep
                ]["objects"][object_id]["infov_pixels"]
            }

    return {"visible_pixels_interested_object": None}


def _read_json_dataframe(path: Path) -> pd.DataFrame:
    try:
        return pd.read_json(path)
    except ValueError:
        import json

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data)


def _load_cached_df(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_pickle(path)


def _save_cached_df(df: pd.DataFrame, path: Path) -> None:
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_pickle(path)


def load_model_answers(results_dir: str | Path, wide: bool = False) -> pd.DataFrame:
    results_dir = Path(results_dir)
    frames = []

    for path in sorted(results_dir.glob("*_val.json")):
        model = path.stem.replace("_val", "")
        df = pd.read_json(path)
        if df.empty:
            print(f"Skipping {model} at path {path} has an empty answers file.")
            continue
        df["model"] = model
        df["og_answer"] = df["answer"]
        df["answer"] = df["answer"].apply(
            lambda a: _sanitize_answer(a)
        )
        frames.append(df)

        assert df["answer"].isna().sum() == 0, f"Error: Found NaN answers in model {model} at path {path}"

        unanswered, invalid, total = df["answer"].eq("").sum(), df["answer"].eq("?").sum(), len(df)
        print(f"{model}: loaded {total} answers{'' if unanswered+invalid == 0 else f' => [WARNING]'}{'' if unanswered == 0 else f' {100 * unanswered / total:.2f}% unanswered/empty'}{'' if invalid == 0 else f', {100 * invalid / total:.2f}% invalid/jibberish'}")

    if not frames:
        return pd.DataFrame()

    df_all = pd.concat(frames, ignore_index=True)
    if not wide:
        return df_all

    # Check if idx are standards
    if not all(df_all["idx"].apply(lambda x: re.fullmatch(_IDX_RE, x))):
        if any(df_all["idx"].apply(lambda x: re.fullmatch(_LEVEL_RE, x))):
            print("Level answers detected. Reformatting the idx column to extract levels.")

            dups = df_all.duplicated(subset=["idx", "model"], keep=False)
            if dups.any():
                print(f"Found {dups.sum()} duplicate (idx, model) pairs found in model answers. This is a known bug, attempting to auto-fix.")
                dups_answers = df_all.duplicated(subset=["idx", "model", "answer", "og_answer"], keep=False)
                if dups.sum() == dups_answers.sum():
                    print(" Auto-fix worked: duplicates have the same answer. Keeping only one entry per (idx, model) pair.")
                    df_all = df_all.drop_duplicates(
                        subset=["idx", "model", "answer", "og_answer"],
                        keep="first",
                    ).copy().reset_index()
            
    # Check for duplicates before pivoting
    dups = df_all.duplicated(subset=["idx", "model"], keep=False)
    if dups.any():
        print(f"Found {dups.sum()} duplicate (idx, model) pairs found in model answers:")
        print(df_all[dups])
        raise ValueError("There are duplicates (idx, model) rows, printed above. Cannot pivot to wide format.")

    return df_all.pivot_table(
        index="idx", columns="model", values="answer", aggfunc="first"
    )


def get_timestep_from_idx(idx: int) -> str:
    return f"{TIMESTART + float(idx) * RENDER_STEP:08.3f}"


def _sanitize_answer(answer: object) -> str | None:
    if answer is None or (isinstance(answer, float) and pd.isna(answer)):
        return ""
    match = _ANSWER_RE.search(str(answer))
    if not match:
        return "?"
    answer = next((group for group in match.groups() if group), None)
    return answer.upper()

def iter_mode_slices(eval_df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    # if "model_mode" not in eval_df.columns:
    #     return [("all", eval_df)]

    slices = []
    for mode in ("image-only", "general"):
        subset = eval_df[eval_df["model_mode"] == mode]
        if not subset.empty:
            slices.append((mode, subset))

    # unknown = eval_df[eval_df["model_mode"] == "unknown"]
    # if not unknown.empty:
    #     slices.append(("unknown", unknown))

    return slices or [("all", eval_df)]

def select_eval_df(
    eval_df: pd.DataFrame, *, mode: str = "all"
) -> list[tuple[str, pd.DataFrame]]:
    if mode == "all":
        slices = iter_mode_slices(eval_df)
        slices.insert(0, ("mixed", eval_df))
        return slices
    
    if mode == "mixed":
        return [("mixed", eval_df)]
    
    # Otherwise, select the specified mode
    subset = eval_df[eval_df["model_mode"] == mode]
    return [(mode, subset)]


GROUPINGS = [
    "family_best", 
    "family_bestmat", 
    "family_biggest", 
    "model", 
    # "family"  # not making sense to average by family
    # "model_best10"
]

def apply_group(df: pd.DataFrame, group_by: str) -> pd.DataFrame:
    # Best average across question (no balancing)
    # model_accuracy = df.groupby(['model_family', 'model_id'])['accuracy'].mean().reset_index()
    # best_models = model_accuracy.loc[model_accuracy.groupby('model_family')['accuracy'].idxmax()]
    # df = df[df['model_id'].isin(best_models['model_id'])]
    # group_by = "model_id"cat_acc_df
    model_best_re = r"model_best([0-9]+)"
    model_bestmat_re = r"model_bestmat([0-9]+)"

    if group_by in ["family_best", "family_bestmat"]:
        cat_acc_df = macro_accuracy(df, level="category", group_by=["model_family", "model_id"])
        
        if group_by == "family_bestmat":
            cat_acc_df = cat_acc_df[cat_acc_df["category"] == "material_understanding"]
        
        # Macro-accuracy: average across categories
        model_accuracy = cat_acc_df.groupby(['model_family', 'model_id'])['accuracy'].mean().reset_index()
        best_models = model_accuracy.loc[model_accuracy.groupby('model_family')['accuracy'].idxmax()]
        df = df[df['model_id'].isin(best_models['model_id'])]
        group_by = "model_id"
    elif group_by in ["family_biggest"]:
        # We select the biggest (primary) most recent (tie-breaker) model per family, 
        # and also check that there are no duplicates (same family, same params, same priority) 
        # among the winners.

        # No need to keep all entries, we just look at model metadata
        df_mod = (
                    df[["model_family", "model_id", "model_params_b", "model_priority"]]
                    .drop_duplicates()
                )
        df_mod.loc[df_mod["model_priority"].isna(), "model_priority"] = 0  # Set unknown priorities

        # Retrieve per-family model with highest params
        max_params = df_mod.groupby("model_family")["model_params_b"].transform("max")
        candidates = df_mod[df_mod["model_params_b"].eq(max_params)]

        # Retrieve per-family model with highest priority, among the candidates
        candidates_maxpriority = candidates.groupby("model_family")["model_priority"].transform("max")
        family_biggest_all = candidates[candidates["model_priority"].eq(candidates_maxpriority)]

        # Check if there are groups (family, params, priority) of size not equal to 1 (ie, duplicates)
        mask = family_biggest_all.groupby("model_family")["model_id"].transform("nunique").ne(1)
        assert family_biggest_all[mask].empty, (
            "Expected exactly 1 highest parameters model per model_family, found more:\n"
            + family_biggest_all[mask].to_string()
        )

        df = df[df['model_id'].isin(family_biggest_all["model_id"])]
        group_by = "model_id"
    elif group_by == "model":
        group_by = "model_id"
    elif re.fullmatch(model_best_re, group_by) is not None or re.fullmatch(model_bestmat_re, group_by) is not None:
        match_best = re.fullmatch(model_best_re, group_by)
        match_bestmat = re.fullmatch(model_bestmat_re, group_by)
        if match_best is not None:
            top_k = int(match_best.group(1))
        elif match_bestmat is not None:
            top_k = int(match_bestmat.group(1))
            df = df[df["category"] == "material_understanding"]
        
        cat_acc_df = macro_accuracy(df, level="model_id", group_by=["model_family", "model_id"])
        
        best_models = cat_acc_df.sort_values("accuracy", ascending=False).iloc[0:top_k]
        df = df[df['model_id'].isin(best_models['model_id'])]
        group_by = "model_id"
    elif group_by == "family":
        print("(i) Model family grouping may be misleading, as it averages models with different capabilities and performance. Consider using 'family_best' or 'family_biggest' to select a single representative model per family.")
        group_by = "model_family"
    else:
        raise ValueError(f"Unknown group_by: {group_by}")

    return df, group_by

def balanced_split_df(df: pd.DataFrame, 
                      group_by: list[str], 
                      balance_col: list[str], 
                      max_size: int | None = None) -> pd.DataFrame:
    group_by = list(group_by)
    balance_col = list(balance_col)
    strata = group_by + balance_col

    # counts per (group, question_id)
    c = df.groupby(strata, observed=True).size().rename("n").reset_index()

    # target per question_id = minimum count across groups
    target = (
        c.groupby(balance_col, observed=True)["n"]
        .min()
        .rename("target_n")
        .reset_index()
    )
    print(f"Max {group_by} size after balancing:", target["target_n"].sum())

    # keep only strata with target_n > 0
    c = c.merge(target, on=balance_col, how="inner")
    c = c[c["target_n"] > 0]


    # sample target_n within each (group_by, question_id) stratum
    merged = df.merge(c[strata + ["target_n"]], on=strata, how="inner")

    sampled_parts = []
    for target_n, chunk in merged.groupby("target_n", observed=True):
        n = int(target_n)
        if n <= 0:
            continue
        sampled_parts.append(
            chunk.groupby(strata, group_keys=False, observed=True)
            .sample(n=n, random_state=42)
        )

    out = pd.concat(sampled_parts, ignore_index=True).drop(columns=["target_n"])
    return out

if __name__ == "__main__":
    df, paths = load_results(
        "/data0/sebastian.cavada/compositional-physics/tiny_vqa_deterministic/output/run_15_general"
    )
    print(df.head())
