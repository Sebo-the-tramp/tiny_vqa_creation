from pathlib import Path
import re

import pandas as pd

_ANSWER_RE = re.compile(r"(?i)^\s*([a-d])(?:[^a-z0-9]|$)")
# _ANSWER_RE = re.compile(r"\b([A-D])\s*[\.\,\:\)]")

try:
    import orjson
except ImportError as exc:
    raise ImportError(
        "orjson is required for streaming simulation.json; install it first."
    ) from exc

_SIM_METADATA_CACHE: dict[str, dict] = {}

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
) -> pd.DataFrame:
    base = Path(base_path)

    test_path = base / run_folder / f"test_{run_folder}_10K.json"
    val_path = base / run_folder / f"val_answer_{run_folder}.json"

    if cache_path is None:
        cache_path = base / run_folder / "merged_results.pkl"
    else:
        cache_path = Path(cache_path)

    if cache and cache_path.exists():
        print("Cache found at ", cache_path)
        df_cached = _load_cached_df(cache_path)
        required_cols = []
        if add_sim_metadata:
            required_cols.append("object_count")
        if merge_model_answers:
            results_dir = (
                Path(model_results_dir)
                if model_results_dir is not None
                else base / run_folder / f"results_{run_folder}"
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

    drop_cols = [
        "scene", "source", "file_name"
    ]

    if drop_cols and keep_cols:
        raise ValueError("Use only one of drop_cols or keep_cols.")
    if keep_cols is not None:
        df_test = df_test[keep_cols]
    elif drop_cols is not None:
        df_test = df_test.drop(columns=drop_cols, errors="ignore")

    # Keep all test questions; val answers may include extra idx.
    df = df_test.merge(df_val, on="idx", how="left", suffixes=("_test", "_val"))

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
            else base / run_folder / f"results_{run_folder}"
        )
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
) -> pd.DataFrame:
    base = Path(base_path)

    test_path = base / run_folder / f"test_{run_folder}_10K.json"
    val_path = base / run_folder / f"val_answer_{run_folder}.json"

    print(f"Loading test data from: {test_path}")
    print(f"Loading val data from: {val_path}")

    if cache_path is None:
        cache_path = base / run_folder / "merged_results.pkl" if run_folder else base / "merged_results.pkl"
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
                else base / run_folder / f"results_{run_folder}"
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

    drop_cols = [
        "scene", "source", "file_name"
    ]

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
            else base / run_folder / f"results_{run_folder}"
        )
        df_models = load_model_answers(results_dir, wide=model_answers_wide)
        if model_answers_wide:
            df = df.merge(df_models, on="idx", how="left")
        else:
            raise ValueError("model_answers_wide=False not supported in load_results.")

    if cache:
        _save_cached_df(df, cache_path)

    return df

def read_simulation_metadata(simulation_json_path: str | Path, verbose: bool = False) -> dict:    
    simulation_json_path = Path(simulation_json_path.replace("simulation.json", "simulation_min.json"))
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
    }
    _SIM_METADATA_CACHE[cache_key] = result
    return result


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
        df = pd.read_json(path)
        df["model"] = path.stem.replace("_val", "")
        df["answer"] = df["answer"].apply(_sanitize_answer)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    df_all = pd.concat(frames, ignore_index=True)
    if not wide:
        return df_all

    return df_all.pivot_table(index="idx", columns="model", values="answer", aggfunc="first")


def _sanitize_answer(answer: object, max_prefix_chars: int | None = 10) -> str | None:
    if answer is None or (isinstance(answer, float) and pd.isna(answer)):
        return None
    if max_prefix_chars is None or max_prefix_chars < 0:
        text = str(answer)
    else:
        text = str(answer)[:max_prefix_chars]
    match = _ANSWER_RE.match(text)
    if not match:
        return "?"
    return match.group(1).upper()


if __name__ == "__main__":
    df = load_results(
        "/data0/sebastian.cavada/compositional-physics/tiny_vqa_deterministic/output/run_15_general"
    )
    print(df.head())
