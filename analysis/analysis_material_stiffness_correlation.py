from __future__ import annotations

import argparse
from pathlib import Path
import re

import pandas as pd

from utils.utils_read import load_results, _sanitize_answer, read_simulation_metadata
import utils.utils_graph as utils_graph
import utils.utils_graph_correlation as utils_graph_correlation
from utils.utils_graph_correlation import (
    create_material_stiffness_violin_grid,
)


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

    if "object_yms_num" not in df.columns:
        df = add_object_yms_num_from_metadata(df)

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
            "object_yms_num",
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


def _yms_level_from_name(name: str) -> int | None:
    label = name.lower()
    if "yms-softer" in label:
        return 1
    if "yms-medium" in label:
        return 2
    if "yms-stiffer" in label:
        return 3
    return None


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _extract_choice_map(question: str) -> dict[str, str]:
    choices: dict[str, str] = {}
    for match in re.finditer(
        r"(?s)\b([A-D])\.\s*(.+?)(?=\n[A-D]\.\s*|$)", question
    ):
        choices[match.group(1)] = match.group(2).strip()
    return choices


def add_object_yms_num_from_metadata(
    df: pd.DataFrame,
    *,
    sim_path_col: str = "simulation_id",
    question_col: str = "question",
) -> pd.DataFrame:
    if sim_path_col not in df.columns or question_col not in df.columns:
        return df

    def _lookup_yms(row: pd.Series) -> object:
        sim_path = row.get(sim_path_col)
        if pd.isna(sim_path):
            return pd.NA
        question = str(row.get(question_col, ""))
        answer = row.get("answer")
        meta = read_simulation_metadata(str(sim_path))
        objects = meta.get("objects", {})
        if not objects:
            return pd.NA

        def _yms_level_from_obj(obj: dict) -> int | None:
            props = obj.get("props", {})
            return _yms_level_from_name(str(props.get("name", "")))

        norm_question = _normalize_text(question)
        obj_entries = []
        for obj in objects.values():
            obj_name = str(obj.get("description", {}).get("object_name", ""))
            if not obj_name:
                continue
            norm_name = _normalize_text(obj_name)
            if not norm_name:
                continue
            obj_entries.append((obj, norm_name))

        matches = [entry for entry in obj_entries if entry[1] in norm_question]
        if len(matches) == 1:
            level = _yms_level_from_obj(matches[0][0])
            return level if level is not None else pd.NA

        if len(matches) > 1:
            answer_letter = (
                _sanitize_answer(str(answer), max_prefix_chars=None)
                if answer is not None
                else None
            )
            if answer_letter:
                choice_map = _extract_choice_map(question)
                chosen = choice_map.get(answer_letter)
                if chosen:
                    norm_choice = _normalize_text(chosen)
                    for obj, norm_name in matches:
                        if norm_name in norm_choice:
                            level = _yms_level_from_obj(obj)
                            return level if level is not None else pd.NA

        if len(objects) == 1:
            obj = next(iter(objects.values()))
            level = _yms_level_from_obj(obj)
            return level if level is not None else pd.NA

        if re.search(r"\b(stiffest|most stiff|most rigid)\b", question, re.I):
            levels = [
                _yms_level_from_obj(obj)
                for obj, _ in obj_entries
                if _yms_level_from_obj(obj) is not None
            ]
            return max(levels) if levels else pd.NA
        if re.search(
            r"\b(softest|least stiff|most soft|most deformable)\b", question, re.I
        ):
            levels = [
                _yms_level_from_obj(obj)
                for obj, _ in obj_entries
                if _yms_level_from_obj(obj) is not None
            ]
            return min(levels) if levels else pd.NA

        return pd.NA

    df = df.copy()
    df["object_yms_num"] = df.apply(_lookup_yms, axis=1)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-path",
        default="/data0/sebastian.cavada/compositional-physics/tiny_vqa_creation/output/",
    )
    parser.add_argument("--run-name", default="run_24_general_yms-variations")
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
        "--family-marker-mode",
        choices=["distinct", "rotated"],
        default="distinct",
        help="Use distinct shapes per family or rotate a base shape per family.",
    )
    parser.add_argument(
        "--family-marker-base",
        default="^",
        help="Base marker to rotate when --family-marker-mode=rotated.",
    )
    parser.add_argument(
        "--stiffness-col",
        default="object_yms_num",
        help="Column for material stiffness levels.",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis/output",
        help="Base directory for saving plots.",
    )
    args = parser.parse_args()

    utils_graph.RUN_NAME = args.run_name
    utils_graph_correlation.RUN_NAME = args.run_name

    output_dir = Path(args.output_dir) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_df = build_eval_df(args.base_path)
    eval_df = add_model_mode(eval_df)

    for mode_label, mode_df in select_eval_df(
        eval_df, mode=args.mode, split_by_mode=args.split_by_mode
    ):
        create_material_stiffness_violin_grid(
            mode_df,
            stiffness_col=args.stiffness_col,
            group_by="model_id",
            output_dir=output_dir,
            run_name=args.run_name,
            save_per_category=True,
            per_category_dirname=f"material_stiffness_per_model_{mode_label}",
            save_grid=True,
            filename=f"material_stiffness_grid_models_{mode_label}.png",
            save_legend=True,
            legend_filename=f"material_stiffness_legend_models_{mode_label}.png",
            legend_cols=6,
            y_limit_mode="fixed",
            family_marker_mode=args.family_marker_mode,
            family_marker_base=args.family_marker_base,
        )

        create_material_stiffness_violin_grid(
            mode_df,
            stiffness_col=args.stiffness_col,
            group_by="family",
            output_dir=output_dir,
            run_name=args.run_name,
            save_per_category=True,
            per_category_dirname=f"material_stiffness_per_family_{mode_label}",
            save_grid=True,
            filename=f"material_stiffness_grid_families_{mode_label}.png",
            save_legend=True,
            legend_filename=f"material_stiffness_legend_families_{mode_label}.png",
            legend_cols=4,
            family_marker_mode=args.family_marker_mode,
            family_marker_base=args.family_marker_base,
        )


if __name__ == "__main__":
    main()
