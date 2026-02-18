from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from utils.utils_read import _read_json_dataframe, _sanitize_answer, load_model_answers


def load_results_test_pa(
    base_path: str | Path,
    run_name: str,
    test_json: str | Path | None,
    val_json: str | Path | None,
    results_dir: str | Path | None,
) -> pd.DataFrame:
    base = Path(base_path) / run_name
    if test_json:
        test_path = Path(test_json)
        if not test_path.is_absolute():
            test_path = base / test_path
    else:
        test_path = base / f"test_{run_name}_10K.json"

    if val_json:
        val_path = Path(val_json)
        if not val_path.is_absolute():
            val_path = base / val_path
    else:
        val_path = base / f"val_answer_{run_name}.json"

    if results_dir:
        results_path = Path(results_dir)
        if not results_path.is_absolute():
            results_path = base / results_path
    else:
        results_path = base / f"results_{run_name}"

    df_test = _read_json_dataframe(test_path)
    df_val = _read_json_dataframe(val_path)

    df_test = df_test.drop(columns=["scene", "source"], errors="ignore")
    df = df_test.merge(df_val, on="idx", how="left", suffixes=("_test", "_val"))

    df_models = load_model_answers(results_path, wide=True)
    if df_models.empty:
        raise ValueError(f"No model answer files found in {results_path}")
    df = df.merge(df_models, on="idx", how="left")

    return df


def build_eval_df(df: pd.DataFrame, results_path: Path) -> pd.DataFrame:
    model_cols = sorted(p.stem.replace("_val", "") for p in results_path.glob("*_val.json"))
    model_cols = [c for c in model_cols if c in df.columns]
    if not model_cols:
        raise ValueError(f"No model answer columns found in {results_path}")

    df["answer"] = df["answer"].apply(
        lambda a: _sanitize_answer(a, max_prefix_chars=None)
    )

    if "mode_val" in df.columns:
        df["mode_y"] = df["mode_val"]
    elif "mode_test" in df.columns:
        df["mode_y"] = df["mode_test"]
    elif "mode" in df.columns:
        df["mode_y"] = df["mode"]

    id_cols = [
        c
        for c in [
            "idx",
            "question_id",
            "category",
            "sub_category",
            "mode_y",
            "answer",
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
    if "mode_y" in eval_df.columns:
        eval_df = eval_df.rename(columns={"mode_y": "mode"})
    return eval_df


def question_group_from_id(question_id: str) -> str:
    if question_id.endswith("_METRIC_PREFIX"):
        return "METRIC_PREFIX"
    if question_id.endswith("_SCIENTIFIC_NOTATION"):
        return "SCIENTIFIC_NOTATION"
    return "BASELINE"


def report_question_text_differences(df_test: pd.DataFrame) -> None:
    if "question_id" not in df_test.columns or "mode" not in df_test.columns:
        print("Question text diff check skipped; missing question_id or mode.")
        return
    diff_qids = []
    for qid, group in df_test.groupby("question_id"):
        modes = group.groupby("mode")["question"].apply(lambda s: set(s)).to_dict()
        if len(modes) < 2:
            continue
        items = list(modes.items())
        base_set = items[0][1]
        for _mode, qset in items[1:]:
            if qset != base_set:
                diff_qids.append(qid)
                break
    print(
        f"Question text differs across modes for {len(diff_qids)} question_id(s)."
    )
    if diff_qids:
        print("Sample question_id with differences:", diff_qids[:5])


def plot_violin_by_question(
    per_model_question: pd.DataFrame,
    per_question_avg: pd.DataFrame,
    output_path: Path,
) -> None:
    order = ["BASELINE", "METRIC_PREFIX", "SCIENTIFIC_NOTATION"]
    order = [o for o in order if o in per_question_avg["question_group"].unique()]
    avg_map = per_question_avg.set_index("question_group")["avg_acc"]
    use_hue = "mode" in per_model_question.columns and per_model_question["mode"].nunique() > 1

    figsize = (max(10, 1.8 * len(order) + 4), 7)
    fig, ax = plt.subplots(figsize=figsize)
    sns.violinplot(
        data=per_model_question,
        x="question_group",
        y="acc",
        order=order,
        cut=0,
        inner=None,
        color="#a6cee3",
        hue="mode" if use_hue else None,
        dodge=use_hue,
        ax=ax,
    )
    sns.stripplot(
        data=per_model_question,
        x="question_group",
        y="acc",
        order=order,
        color="#1f77b4",
        size=2,
        alpha=0.4,
        jitter=0.2,
        hue="mode" if use_hue else None,
        dodge=use_hue,
        ax=ax,
    )
    ax.scatter(
        range(len(order)),
        [avg_map.get(group, 0.0) for group in order],
        color="#d62728",
        s=24,
        label="Mean",
        zorder=3,
    )
    ax.set_title("Accuracy by question group across models")
    ax.set_xlabel("question group")
    ax.set_ylabel("Accuracy")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.tick_params(axis="x", rotation=0)
    if use_hue:
        handles, labels = ax.get_legend_handles_labels()
        seen = {}
        for h, l in zip(handles, labels):
            if l not in seen:
                seen[l] = h
        ax.legend(seen.values(), seen.keys(), loc="upper right", title="mode")
    else:
        ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_avg_bar(
    per_question_avg: pd.DataFrame,
    per_question_avg_by_mode: pd.DataFrame | None,
    output_path: Path,
) -> None:
    order = ["BASELINE", "METRIC_PREFIX", "SCIENTIFIC_NOTATION"]
    order = [o for o in order if o in per_question_avg["question_group"].unique()]
    figsize = (10, 6)
    fig, ax = plt.subplots(figsize=figsize)
    if per_question_avg_by_mode is not None:
        sns.barplot(
            data=per_question_avg_by_mode,
            x="question_group",
            y="avg_acc",
            hue="mode",
            order=order,
            ax=ax,
        )
        ax.legend(loc="upper right", title="mode")
    else:
        sns.barplot(
            data=per_question_avg,
            x="question_group",
            y="avg_acc",
            order=order,
            color="#b7e4a8",
            ax=ax,
        )
    ax.set_title("Average accuracy by question group (across models)")
    ax.set_xlabel("question group")
    ax.set_ylabel("Average accuracy")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-path",
        default="/data0/sebastian.cavada/compositional-physics/tiny_vqa_deterministic/output/",
    )
    parser.add_argument("--run-name", default="run_20_test_pa")
    parser.add_argument(
        "--test-json",
        default="test_run_20_test_pa_general.json",
        help="Optional explicit test JSON path.",
    )
    parser.add_argument(
        "--val-json",
        default="val_answer_run_20_test_pa_general.json",
        help="Optional explicit val JSON path.",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Optional explicit results directory.",
    )
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    base = Path(args.base_path) / args.run_name
    results_path = Path(args.results_dir) if args.results_dir else base / f"results_{args.run_name}"

    output_dir = Path("output") / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_results_test_pa(
        base_path=args.base_path,
        run_name=args.run_name,
        test_json=args.test_json,
        val_json=args.val_json,
        results_dir=results_path,
    )
    report_question_text_differences(df)
    eval_df = build_eval_df(df, results_path)
    eval_df = eval_df[eval_df["is_correct"].notna()]

    per_question = (
        eval_df.groupby(["question_id", "model_id", "mode"], observed=True)["is_correct"]
        .mean()
        .reset_index(name="acc")
    )
    per_question["question_group"] = per_question["question_id"].apply(
        question_group_from_id
    )
    per_model_question = (
        per_question.groupby(["question_group", "model_id", "mode"], observed=True)[
            "acc"
        ]
        .mean()
        .reset_index(name="acc")
    )
    per_question_avg = (
        per_model_question.groupby("question_group", observed=True)["acc"]
        .mean()
        .reset_index(name="avg_acc")
    )
    per_question_avg_by_mode = None
    if "mode" in per_model_question.columns and per_model_question["mode"].nunique() > 1:
        per_question_avg_by_mode = (
            per_model_question.groupby(["question_group", "mode"], observed=True)["acc"]
            .mean()
            .reset_index(name="avg_acc")
        )
        per_question_avg_by_mode["question_group"] = pd.Categorical(
            per_question_avg_by_mode["question_group"],
            categories=per_question_avg["question_group"].tolist(),
            ordered=True,
        )
        per_question_avg_by_mode = per_question_avg_by_mode.sort_values("question_group")

    per_model_question.to_csv(
        output_dir / "accuracy_by_question_group_per_model.csv", index=False
    )
    per_question_avg.to_csv(
        output_dir / "avg_accuracy_by_question_group.csv", index=False
    )
    if per_question_avg_by_mode is not None:
        per_question_avg_by_mode.to_csv(
            output_dir / "avg_accuracy_by_question_group_mode.csv", index=False
        )

    plot_violin_by_question(
        per_model_question,
        per_question_avg,
        output_dir / "violin_accuracy_by_question_group.png",
    )
    plot_avg_bar(
        per_question_avg,
        per_question_avg_by_mode,
        output_dir / "avg_accuracy_by_question_group.png",
    )

    print("Average accuracy by question group:")
    print(per_question_avg.to_string(index=False))


if __name__ == "__main__":
    main()
