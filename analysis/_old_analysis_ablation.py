from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from utils.utils_read import load_results, _sanitize_answer
import utils.utils_graph as utils_graph
from utils.utils_graph import create_graph_from_eval_balanced

run = "23"

DEFAULT_ABLATIONS = [
    f"run_{run}_roi_ablation_baseline",
    f"run_{run}_roi_circling_no_text",
    f"run_{run}_roi_circling_no_text_layout_position",
    f"run_{run}_roi_circling_text",
    f"run_{run}_roi_circling_text_layout_position",
    f"run_{run}_no_roi_circling_yes_text_no_layout_position",
]

TARGET_CATEGORIES = [
    "mass",
    "density",
    "young_modulus",
    "poisson_ratio",
    "material_identification",
]
DISPLAY_NAMES = ["Mass", "Density", "Young Modulus", "Poisson Ratio", "Mat. ID"]

EXCLUDE_MODELS: list[str] = []

TABLE_ROWS = [
    (f"run_{run}_roi_ablation_baseline", "\\checkmark", "-", "-"),
    # (f"run_{run}_no_roi_circling_yes_text_no_layout_position", "\\checkmark", "\\checkmark", "-",), # -> this is fundamentally the ablation    
    (f"run_{run}_roi_circling_text_layout_position", "\\checkmark", "\\checkmark", "\\checkmark"),
    (f"run_{run}_roi_circling_text", "\\checkmark", "-", "\\checkmark"),
    (f"run_{run}_roi_circling_no_text_layout_position", "-", "\\checkmark", "\\checkmark"),
    (f"run_{run}_roi_circling_no_text", "-", "-", "\\checkmark"),
]


def _get_results_dir(base: Path, run_name: str) -> Path:
    sanitized = base / run_name / f"results_{run_name}_sanitized"
    if sanitized.exists():
        return sanitized
    raise FileNotFoundError(
        f"Missing sanitized results dir: {sanitized}. Please generate sanitized outputs."
    )


def build_eval_df(base_path: str | Path, run_name: str) -> pd.DataFrame | None:
    base = Path(base_path)
    results_dir = _get_results_dir(base, run_name)
    model_files = sorted(results_dir.glob("*_val.json"))
    if not model_files:
        print(f"Skipping {run_name}: no model results in {results_dir}")
        return None

    df = load_results(
        base,
        run_folder=run_name,
        merge_model_answers=True,
        model_answers_wide=True,
        model_results_dir=results_dir,
        cache=True,
    )

    model_cols = sorted(p.stem.replace("_val", "") for p in model_files)
    model_cols = [c for c in model_cols if c in df.columns and c not in EXCLUDE_MODELS]
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
    parser.add_argument("--ablations", nargs="*", default=DEFAULT_ABLATIONS)
    parser.add_argument("--output-run-name", default="run_23_roi_ablation")
    args = parser.parse_args()

    utils_graph.RUN_NAME = args.output_run_name

    eval_dfs: list[pd.DataFrame] = []
    run_names: list[str] = []
    for run_name in args.ablations:
        eval_dfs.append(build_eval_df(args.base_path, run_name))
        if eval_dfs[-1] is not None:
            run_names.append(run_name)
        else:
            eval_dfs.pop()

    acc_mats: dict[str, pd.DataFrame] = {}
    acc_mats_sub: dict[str, pd.DataFrame] = {}
    for run_name, eval_df in zip(run_names, eval_dfs):
        eval_df_multi_frame = eval_df[eval_df["idx"].astype(str).str.contains("_g")]
        print("number of questions:", len(eval_df_multi_frame))
        print(run_name)

        acc_mat, _ = create_graph_from_eval_balanced(
            eval_base=eval_df_multi_frame,
            index_to_use="question_id",
            title=(
                "Balanced accuracy by question_id and general models - "
                f"multi-frame - {run_name}"
            ),
            color_by_mode=True,
        )

        acc_mats[run_name] = acc_mat

        acc_mat_sub, _ = create_graph_from_eval_balanced(
            eval_base=eval_df_multi_frame,
            index_to_use="sub_category",
            title=(
                "Balanced accuracy by sub_category and general models - "
                f"multi-frame - {run_name}"
            ),
            color_by_mode=True,
        )

        acc_mats_sub[run_name] = acc_mat_sub

    if not acc_mats or not acc_mats_sub:
        print("No ablation results found to summarize.")
        return

    def _format_mean_std(values: pd.Series) -> tuple[float, float, str]:
        series = pd.to_numeric(values, errors="coerce").dropna()
        series = series.drop(index=["Average"], errors="ignore")
        if series.empty:
            return float("nan"), float("nan"), "--"
        mean_val = series.mean()
        std_val = series.std(ddof=1) if len(series) > 1 else 0.0
        return mean_val, std_val, f"{mean_val:.1f} $\\pm$ {std_val:.1f}"

    colors_balanced = ["#E57373", "#F6E6B3", "#8BC87A"]
    cmap_balanced = LinearSegmentedColormap.from_list("soft_r2g", colors_balanced)

    def _get_heatmap_color(val: float, min_val: float, max_val: float) -> str:
        if max_val <= min_val:
            norm = 0.0
        else:
            norm = (val - min_val) / (max_val - min_val)
        norm = max(0.0, min(1.0, norm))
        r_f, g_f, b_f, _ = cmap_balanced(norm)
        r, g, b = int(r_f * 255), int(g_f * 255), int(b_f * 255)
        return f"{r:02X}{g:02X}{b:02X}"

    def _colorize(mean_val: float, std_val: float, min_val: float, max_val: float) -> str:
        if pd.isna(mean_val):
            return "--"
        color_hex = _get_heatmap_color(mean_val, min_val, max_val)
        return f"\\cellcolor[HTML]{{{color_hex}}}{{{mean_val:.1f} $\\pm$ {std_val:.1f}}}"

    table_data: dict[str, dict[str, object]] = {}
    for run_name in run_names:
        acc_row = acc_mats[run_name].iloc[-1]
        mean_acc, std_acc, _ = _format_mean_std(acc_row[1:] * 100)

        sub_cat_values: list[str] = []
        sub_cat_means: list[float] = []
        sub_cat_stds: list[float] = []
        for cat in TARGET_CATEGORIES:
            try:
                row = acc_mats_sub[run_name].loc[cat]
                mean_cat, std_cat, _ = _format_mean_std(row[1:] * 100)
                sub_cat_means.append(mean_cat)
                sub_cat_stds.append(std_cat)
            except KeyError:
                sub_cat_means.append(float("nan"))
                sub_cat_stds.append(float("nan"))
            except Exception as exc:
                print(f"Error processing {cat} for {run_name}: {exc}")
                sub_cat_means.append(float("nan"))
                sub_cat_stds.append(float("nan"))

        table_data[run_name] = {
            "main_mean": mean_acc,
            "main_std": std_acc,
            "sub_means": sub_cat_means,
            "sub_stds": sub_cat_stds,
        }

    all_vals = []
    for run_name in run_names:
        data = table_data.get(run_name, {})
        all_vals.extend(
            v for v in [data.get("main_mean")] + list(data.get("sub_means", []))
            if v is not None and not pd.isna(v)
        )
    min_val = min(all_vals) if all_vals else 0.0
    max_val = max(all_vals) if all_vals else 100.0

    def _row_values(run_key: str) -> list[str]:
        data = table_data.get(run_key)
        if not data:
            return ["--"] * (len(TARGET_CATEGORIES) + 1)
        subs = [
            _colorize(m, s, min_val, max_val)
            for m, s in zip(data["sub_means"], data["sub_stds"])
        ]
        main_val = _colorize(data["main_mean"], data["main_std"], min_val, max_val)
        return subs + [main_val]

    row_lines = []
    for idx, (run_key, obj_name, location, circling) in enumerate(TABLE_ROWS):
        row_vals = " & ".join(_row_values(run_key))
        row_lines.append(
            f"         {obj_name} & {location} & {circling} & {row_vals} \\\\"
        )
        if idx == 2:
            row_lines.append("         \\midrule")

    latex_table = (
        "    \\resizebox{\\linewidth}{!}{%\n"
        "    \\begin{tabular}{cc|c|ccccc|c}\n"
        "         \\toprule\n"
        "         \\multicolumn{2}{c|}{\\textit{Textual cues}} & \\textit{Visual cues} \\\\\n"
        "         Obj. name & Location & Circling & Mass & Density & Young Modulus & Poisson Ratio & Mat. ID & Avg. (\\%)\\\\\n"
        "         \\midrule          \n"
        + "\n".join(row_lines)
        + "\n"
        "         \\bottomrule\n"
        "    \\end{tabular}%\n"
        "    }"
    )

    output_dir = Path("output") / args.output_run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ablation_table.tex").write_text(latex_table)

    header = ["Run", "Acc"] + DISPLAY_NAMES
    widths = [max(len(h), 16) for h in header]
    line = "  ".join(h.ljust(w) for h, w in zip(header, widths))
    print(line)
    print("-" * len(line))
    for run_key, _, _, _ in TABLE_ROWS:
        data = table_data.get(run_key)
        if not data:
            row = [run_key, "--"] + ["--"] * len(DISPLAY_NAMES)
            print("  ".join(cell.ljust(w) for cell, w in zip(row, widths)))
            continue
        main_plain = f"{data['main_mean']:.1f} +/- {data['main_std']:.1f}"
        sub_plain = [
            f"{m:.1f} +/- {s:.1f}"
            for m, s in zip(data["sub_means"], data["sub_stds"])
        ]
        row = [run_key, main_plain] + sub_plain
        print("  ".join(cell.ljust(w) for cell, w in zip(row, widths)))


if __name__ == "__main__":
    main()
