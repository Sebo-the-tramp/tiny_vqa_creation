from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils.utils_read import load_results, _sanitize_answer
import utils.utils_graph as utils_graph
from utils.utils_graph import create_graph_from_eval_balanced

DEFAULT_ABLATIONS = [
    "run_23_ablation_baseline",
    "run_23_roi_circling_no_text",
    "run_23_roi_circling_no_text_layout_position",
    "run_23_roi_circling_text",
    "run_23_roi_circling_text_layout_position",
    "run_23_black",
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
    (
        "run_23_ablation_baseline",
        "\\checkmark",
        "\\checkmark",
        "$\\times$",
        "$\\times$",
    ),
    (
        "run_23_roi_circling_no_text",
        "\\checkmark",
        "$\\times$",
        "\\checkmark",
        "$\\times$",
    ),
    (
        "run_23_roi_circling_no_text_layout_position",
        "\\checkmark",
        "$\\times$",
        "\\checkmark",
        "\\checkmark",
    ),
    (
        "run_23_roi_circling_text",
        "\\checkmark",
        "\\checkmark",
        "\\checkmark",
        "$\\times$",
    ),
    (
        "run_23_roi_circling_text_layout_position",
        "\\checkmark",
        "\\checkmark",
        "\\checkmark",
        "\\checkmark",
    ),
    (
        "run_23_black",
        "$\\times$",
        "$\\times$",
        "$\\times$",
        "$\\times$",
    ),
]


def _get_results_dir(base: Path, run_name: str) -> Path:
    sanitized = base / run_name / f"results_{run_name}_sanitized"
    if sanitized.exists():
        return sanitized
    return base / run_name / f"results_{run_name}"


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
        eval_df_single_image = eval_df[eval_df["idx"].astype(str).str.contains("_i")]
        print("number of questions:", len(eval_df_single_image))
        print(run_name)

        acc_mat, _ = create_graph_from_eval_balanced(
            eval_base=eval_df_single_image,
            index_to_use="question_id",
            title=(
                "Balanced accuracy by question_id and general models - "
                f"single-image - {run_name}"
            ),
            color_by_mode=True,
        )

        acc_mats[run_name] = acc_mat

        acc_mat_sub, _ = create_graph_from_eval_balanced(
            eval_base=eval_df_single_image,
            index_to_use="sub_category",
            title=(
                "Balanced accuracy by sub_category and general models - "
                f"single-image - {run_name}"
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

    def _get_heatmap_color(val: float) -> str:
        val = max(0.0, min(100.0, val))
        if val < 50:
            norm = val / 50.0
            r, g, b = 255, int(255 * norm), 0
        else:
            norm = (val - 50.0) / 50.0
            r, g, b = int(255 * (1 - norm)), 255, 0
        mix_white = 0.4
        r = int(r * (1 - mix_white) + 255 * mix_white)
        g = int(g * (1 - mix_white) + 255 * mix_white)
        b = int(b * (1 - mix_white) + 255 * mix_white)
        return f"{r:02X}{g:02X}{b:02X}"

    def _colorize(mean_val: float, std_val: float) -> str:
        if pd.isna(mean_val):
            return "--"
        color_hex = _get_heatmap_color(mean_val)
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
                sub_cat_values.append(_colorize(mean_cat, std_cat))
                sub_cat_means.append(mean_cat)
                sub_cat_stds.append(std_cat)
            except KeyError:
                sub_cat_values.append("--")
                sub_cat_means.append(float("nan"))
                sub_cat_stds.append(float("nan"))
            except Exception as exc:
                print(f"Error processing {cat} for {run_name}: {exc}")
                sub_cat_values.append("err")
                sub_cat_means.append(float("nan"))
                sub_cat_stds.append(float("nan"))

        table_data[run_name] = {
            "main_res": _colorize(mean_acc, std_acc),
            "subs": sub_cat_values,
            "main_mean": mean_acc,
            "main_std": std_acc,
            "sub_means": sub_cat_means,
            "sub_stds": sub_cat_stds,
        }

    overall_main_vals = pd.Series(
        [table_data[run]["main_mean"] for run in run_names], dtype="float64"
    )
    overall_main_mean = overall_main_vals.mean()
    overall_main_std = overall_main_vals.std(ddof=1) if len(overall_main_vals) > 1 else 0.0
    overall_main_str = f"{overall_main_mean:.1f} $\\pm$ {overall_main_std:.1f}"

    overall_subs = []
    for idx in range(len(TARGET_CATEGORIES)):
        vals = pd.Series(
            [table_data[run]["sub_means"][idx] for run in run_names],
            dtype="float64",
        )
        mean_val = vals.mean()
        std_val = vals.std(ddof=1) if len(vals) > 1 else 0.0
        overall_subs.append(f"{mean_val:.1f} $\\pm$ {std_val:.1f}")

    latex_header_subs = " & ".join(DISPLAY_NAMES)
    latex_col_def = "cccc|c|" + ("c" * len(DISPLAY_NAMES))

    def get_subs_latex(run_key: str) -> str:
        return " & ".join(table_data.get(run_key, {}).get("subs", ["--"] * len(DISPLAY_NAMES)))

    def get_main_latex(run_key: str) -> str:
        return table_data.get(run_key, {}).get("main_res", "--")

    latex_table = (
        "\\begin{table}[t]\n"
        "    \\centering\n"
        "    % IMPORTANT: Requires \\\\usepackage[table]{xcolor} in preamble\n"
        "    \\resizebox{\\linewidth}{!}{%\n"
        f"    \\begin{{tabular}}{{{latex_col_def}}}\n"
        "         \\toprule\n"
        f"         Image & Text & ROI & Layout & Acc. (\\%) & {latex_header_subs} \\\\\n"
        "         \\midrule          \n"
        f"         % VQA Baseline\n"
        f"         {TABLE_ROWS[0][1]} & {TABLE_ROWS[0][2]} & {TABLE_ROWS[0][3]} & {TABLE_ROWS[0][4]} & {get_main_latex(TABLE_ROWS[0][0])} & {get_subs_latex(TABLE_ROWS[0][0])} \\\\\n"
        "         \\midrule\n"
        f"         % Visual Only (ROI)\n"
        f"         {TABLE_ROWS[1][1]} & {TABLE_ROWS[1][2]} & {TABLE_ROWS[1][3]} & {TABLE_ROWS[1][4]} & {get_main_latex(TABLE_ROWS[1][0])} & {get_subs_latex(TABLE_ROWS[1][0])} \\\\\n"
        f"         % Visual Only (ROI + Layout)\n"
        f"         {TABLE_ROWS[2][1]} & {TABLE_ROWS[2][2]} & {TABLE_ROWS[2][3]} & {TABLE_ROWS[2][4]} & {get_main_latex(TABLE_ROWS[2][0])} & {get_subs_latex(TABLE_ROWS[2][0])} \\\\\n"
        "         \\midrule\n"
        f"         % Visual + Text (ROI)\n"
        f"         {TABLE_ROWS[3][1]} & {TABLE_ROWS[3][2]} & {TABLE_ROWS[3][3]} & {TABLE_ROWS[3][4]} & {get_main_latex(TABLE_ROWS[3][0])} & {get_subs_latex(TABLE_ROWS[3][0])} \\\\\n"
        f"         % Visual + Text (ROI + Layout)\n"
        f"         {TABLE_ROWS[4][1]} & {TABLE_ROWS[4][2]} & {TABLE_ROWS[4][3]} & {TABLE_ROWS[4][4]} & {get_main_latex(TABLE_ROWS[4][0])} & {get_subs_latex(TABLE_ROWS[4][0])} \\\\\n"
        "         \\midrule\n"
        f"         % No Cue\n"
        f"         {TABLE_ROWS[5][1]} & {TABLE_ROWS[5][2]} & {TABLE_ROWS[5][3]} & {TABLE_ROWS[5][4]} & {get_main_latex(TABLE_ROWS[5][0])} & {get_subs_latex(TABLE_ROWS[5][0])} \\\\\n"
        "         \\bottomrule\n"
        "    \\end{tabular}%\n"
        "    }\n"
        "    \\caption{Ablation matrix. Ablation matrix of grounding cues on Physics Properties questions, divided by sub-categories. Main accuracy reported as mean $\\pm$ std. Colors indicate performance (Red=Low, Green=High).}\n"
        "    \\label{tab:grounding_matrix}\n"
        "\\end{table}\n"
    )

    output_dir = Path("output") / args.output_run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ablation_table.tex").write_text(latex_table)

    header = ["Run", "Acc"] + DISPLAY_NAMES
    widths = [max(len(h), 16) for h in header]
    line = "  ".join(h.ljust(w) for h, w in zip(header, widths))
    print(line)
    print("-" * len(line))
    for run_key, _, _, _, _ in TABLE_ROWS:
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
