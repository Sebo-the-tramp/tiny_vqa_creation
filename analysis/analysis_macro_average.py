from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.utils_read import load_results, _sanitize_answer
from utils.utils_graph import make_balanced_matrix


BASE_PATH = Path(
    "/data0/sebastian.cavada/compositional-physics/tiny_vqa_creation/output/"
)

# Map label -> run_name (folder under BASE_PATH).
RUNS = {
    "run_25_roi_ablation_baseline": "run_25_roi_ablation_baseline",
    "run_25_roi_circling_no_text": "run_25_roi_circling_no_text",
    "run_25_roi_circling_no_text_layout_position": "run_25_roi_circling_no_text_layout_position",         
}

# If non-empty, compute metrics per model_id in this list.
MODELS: list[str] = []


def _build_eval_df(base_path: Path, run_name: str) -> pd.DataFrame | None:
    run_folder = Path(run_name)
    results_dir = base_path / run_folder / f"results_{run_folder}_sanitized"
    model_cols = sorted(p.stem.replace("_val", "") for p in results_dir.glob("*_val.json"))
    if not model_cols:
        print(f"Skipping {run_name}: no model results in {results_dir}")
        return None

    df = load_results(
        base_path,
        run_folder,
        merge_model_answers=True,
        model_answers_wide=True,
        cache=True,
    )

    model_cols = [c for c in model_cols if c in df.columns]
    if not model_cols:
        print(f"Skipping {run_name}: no matching model columns in data")
        return None

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

    return eval_df


def macro_average_for_run(base_path: Path, run_name: str, model_id: str | None = None):
    eval_df = _build_eval_df(base_path, run_name)
    if eval_df is None:
        return None

    if model_id is not None:
        eval_df = eval_df[eval_df["model_id"] == model_id]
        if eval_df.empty:
            print(f"Skipping {run_name}: model_id {model_id} not found")
            return None

    _, breakdown = make_balanced_matrix(eval_df, by="model_id")
    overall = breakdown["overall"].set_index("model_id")["balanced_overall"]

    exact_cat_acc = (
        eval_df.groupby("category", observed=True, dropna=False)["is_correct"]
        .mean()
        .to_dict()
    )
    return float(overall.mean()), float(overall.std(ddof=0)), exact_cat_acc


def main() -> None:
    for label, run_name in RUNS.items():
        target_models = MODELS or [None]
        for model_id in target_models:
            result = macro_average_for_run(BASE_PATH, run_name, model_id=model_id)
            if result is None:
                continue
            mean, std, exact_cat_acc = result

            header = f"Run: {label} ({run_name})"
            if model_id is not None:
                header = f"{header} | {model_id}"
            print(header)
            print(f"Macro-average accuracy: {mean:.4f} ± {std:.4f}")

            if exact_cat_acc:
                print("per_category exact accuracy:")
                for cat in sorted(exact_cat_acc):
                    print(f"  {cat}: {exact_cat_acc[cat]:.4f}")


if __name__ == "__main__":
    main()
