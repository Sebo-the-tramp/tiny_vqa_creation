from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.markers as mmarkers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", message=".*edgecolor.*unfilled marker.*")

# /data0/sebastian.cavada/compositional-physics/tiny_vqa_deterministic/output/run_11_general_levels

def _safe_filename(label: str) -> str:
    return label.replace("/", "_").replace("\\", "_").replace(" ", "_")


def _macro_avg_by_question(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if "question_id" not in df.columns:
        return df.groupby(group_cols, observed=True)["accuracy"].mean().reset_index()
    q_acc = (
        df.groupby(group_cols + ["question_id"], observed=True)["accuracy"]
        .mean()
        .reset_index()
    )
    return q_acc.groupby(group_cols, observed=True)["accuracy"].mean().reset_index()


def _load_model_metadata(metadata_path: str | Path) -> pd.DataFrame:
    path = Path(metadata_path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    if "id" in df.columns:
        df = df.rename(columns={"id": "model_id"})
    return df


def _build_model_style(
    eval_df: pd.DataFrame,
    metadata_path: str | Path | None,
    *,
    group_by: str = "model_id",
) -> tuple[dict[str, tuple[str, str, float]], dict[str, str]]:
    assert False and "Replace with refactorize _build_model_style"
    group_ids = []
    palette = []

    metadata_df = None
    if metadata_path is not None and Path(metadata_path).exists():
        metadata_df = _load_model_metadata(metadata_path)

    families = []
    params = []
    family_map = {}
    for model_id in pd.unique(eval_df["model_id"]):
        if metadata_df is not None and model_id in set(metadata_df["model_id"]):
            row = metadata_df[metadata_df["model_id"] == model_id].iloc[0]
            family = str(row.get("family", "Other"))
            params_b = pd.to_numeric(row.get("params_b", np.nan), errors="coerce")
        else:
            family = "Other"
            params_b = np.nan
        family_map[str(model_id)] = family
        if group_by == "model_id":
            families.append(family)
            params.append(params_b)

    if group_by == "family":
        group_ids = pd.unique(pd.Series(list(family_map.values())))
        families = list(group_ids)
        params = [np.nan] * len(group_ids)
    else:
        group_ids = pd.unique(eval_df["model_id"])

    unique_families = list(dict.fromkeys(families)) if families else ["Other"]
    palette = sns.color_palette("tab20", len(group_ids))
    markers = list(_DEFAULT_MARKERS)
    for marker in list(mmarkers.MarkerStyle.markers.keys()):
        if marker not in markers:
            markers.append(marker)

    family_markers = {
        fam: markers[i % len(markers)] for i, fam in enumerate(unique_families)
    }

    params = np.array(params, dtype=float)
    valid_params = params[~np.isnan(params)]
    fallback = float(np.nanmedian(valid_params)) if valid_params.size else 5.0
    params = np.where(np.isnan(params), fallback, params)
    params = np.clip(params, 2.0, 15.0)
    sizes = (params - 2.0) / (15.0 - 2.0)

    model_style = {}
    for group_id, color, fam, size in zip(group_ids, palette, families, sizes):
        model_style[str(group_id)] = (color, family_markers.get(fam, "o"), float(size))

    return model_style, family_map


def create_scatter_by_family(
    eval_df: pd.DataFrame,
    *,
    metadata_path: str | Path | None = "utils/metadata.json",
    levels_sorted: list[str] | None = None,
    split_by_mode: bool = False,
    modes: list[str] | None = None,
    show: bool = True,
    output_dir: str | Path | None = None,
    run_name: str | None = None,
    filename: str = "levels_by_family.png",
) -> plt.Figure:
    raise NotImplementedError("This should be updated to use the new macro accuracies")
    levels_sorted = levels_sorted or [
        "baseline",
        "child",
        "teen",
        "undegrad",
        "graduate",
        "expert",
    ]

    if "model_id" not in eval_df.columns:
        raise KeyError("eval_df must include 'model_id'.")
    if "level" not in eval_df.columns:
        raise KeyError("eval_df must include 'level'.")
    if "sub_category" not in eval_df.columns:
        raise KeyError("eval_df must include 'sub_category'.")

    question_col = (
        "question_id_base" if "question_id_base" in eval_df.columns else "question_id"
    )
    if question_col not in eval_df.columns:
        raise KeyError("eval_df must include 'question_id' or 'question_id_base'.")

    if "accuracy" in eval_df.columns:
        eval_df = eval_df.copy()
        eval_df["accuracy"] = pd.to_numeric(eval_df["accuracy"], errors="coerce")
    elif "is_correct" in eval_df.columns:
        eval_df = eval_df.copy()
        eval_df["accuracy"] = pd.to_numeric(eval_df["is_correct"], errors="coerce")
    else:
        raise KeyError("eval_df must include 'accuracy' or 'is_correct'.")
    
    eval_df["accuracy"] *= 100

    family_style, family_map = _build_model_style(
        eval_df,
        metadata_path=metadata_path,
        group_by="family",
    )
    eval_df = eval_df.copy()
    eval_df["family"] = eval_df["model_id"].map(family_map).fillna("Other")

    print(eval_df["family"].unique())
    print(eval_df["sub_category"].unique())
    
    metadata_df = None
    if metadata_path is not None and Path(metadata_path).exists():
        metadata_df = _load_model_metadata(metadata_path)
    mode_map = {}
    if metadata_df is not None and "mode" in metadata_df.columns:
        mode_map = (
            metadata_df.dropna(subset=["model_id"])
            .set_index("model_id")["mode"]
            .to_dict()
        )

    line_styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 3))]
    rng = np.random.default_rng(0)

    def _plot_subset(
        plot_df: pd.DataFrame, title_suffix: str, output_name: str
    ) -> plt.Figure:
        q_scores = (
            plot_df.groupby(
                ["family", question_col, "level", "sub_category"], observed=True
            )["accuracy"]
            .mean()
            .reset_index()
            .dropna()
        )

        final_plot_df = (
            q_scores.groupby(["family", "level"], observed=True)["accuracy"]
            .agg(mean_accuracy="mean", band_width="sem")
            .reset_index()
            .dropna()
        )

        level_map = {lvl: i for i, lvl in enumerate(levels_sorted)}
        final_plot_df["level_idx"] = final_plot_df["level"].map(level_map)

        fig, ax = plt.subplots(figsize=(12, 6))
        # plot_df_model = (
        #     plot_df.groupby(
        #         ["model_id", question_col, "level", "sub_category"], observed=True
        #     )["accuracy"]
        #     .mean()
        #     .reset_index()
        #     .dropna()
        # )
        # final_plot_df_model = (
        #     plot_df_model.groupby(["model_id", "level"], observed=True)["accuracy"]
        #     .agg(mean_accuracy="mean", band_width="sem")
        #     .reset_index()
        #     .dropna()
        # )
        # final_plot_df_model["level_idx"] = final_plot_df_model["level"].map(level_map)
        # sns.violinplot(
        #     data=final_plot_df_model,
        #     x="level_idx",
        #     y="mean_accuracy",
        #     ax=ax,
        #     color="0.85",
        #     # inner="box",
        #     inner=None,
        #     cut=0,
        #     width=1.0,
        #     linewidth=0.5,
        #     order=list(range(len(levels_sorted))),
        # )

        for i, (family, fam_data) in enumerate(final_plot_df.groupby("family")):
            fam_data = fam_data.sort_values("level_idx")
            if fam_data.empty:
                continue
            
            jitter = rng.uniform(-0.15, 0.15, size=fam_data["level_idx"].values[1:].size)
            x = fam_data["level_idx"].values[1:] + jitter
            y = fam_data["mean_accuracy"].values[1:]
            baseline_acc =fam_data[fam_data["level_idx"] == 0]["mean_accuracy"].values
            # y = y - baseline_acc

            y_err = fam_data["band_width"].values
            color, marker, _size = family_style.get(family, ("black", "o", 0.5))
            ls = line_styles[i % len(line_styles)]
            # ax.plot(
            #     x, y, color=color, linestyle=ls, linewidth=2, alpha=0.85, label=family
            # )
            ax.scatter(
                x, y, color=color, marker=marker, s=_size**2, edgecolor="white", linewidth=1, zorder=4
            )
            # if np.isfinite(y_err).any():
            #     ax.fill_between(x, y - y_err, y + y_err, color=color, alpha=0.12)
        
        # n = 10
        # y_min, y_max = ax.get_ylim()
        # for i in range(n):
        #     ax.axhspan(y_min*(i+1)/n, y_min*i/n, facecolor="#ffcccc", alpha=0.5*i/n, zorder=1)
        # for i in range(n):
        #     ax.axhspan(y_max*i/n, y_max*(i+1)/n, facecolor="#ccffcc", alpha=0.5*i/n, zorder=1)
        # ax.set_ylim(y_min, y_max)
        # ax.axhline(0, color="#555555", alpha=1, zorder=3, linewidth=2)

        ax.set_xticks(list(range(len(levels_sorted)))[1:])
        nice_labels = [level.capitalize() for level in levels_sorted]
        ax.set_xticklabels(nice_labels[1:], fontsize=11, fontweight="bold", rotation=30)
        ax.tick_params(axis='x', pad=-2)
        # ax.set_xlabel("Difficulty Level", fontsize=12)
        ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.set_title(f"Performance by Family{title_suffix}", fontsize=14)
        # ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, linestyle="--", alpha=0.5)

        fig.tight_layout()
        utils_graph.paperformat(ax, figsize=(4, 3.5), grid=["y"])

        run = run_name or globals().get("RUN_NAME", "default")
        out_dir = Path(output_dir) if output_dir is not None else Path("output") / run
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / output_name, dpi=300, bbox_inches="tight", pad_inches=0.05)

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def _plot_baseline_improvement(
        plot_df: pd.DataFrame, title_suffix: str, output_name: str
    ) -> plt.Figure:
        q_scores = (
            plot_df.groupby(
                ["family", question_col, "level", "sub_category"], observed=True
            )["accuracy"]
            .mean()
            .reset_index()
            .dropna()
        )

        final_plot_df = (
            q_scores.groupby(["family", "level"], observed=True)["accuracy"]
            .agg(mean_accuracy="mean", band_width="sem")
            .reset_index()
            .dropna()
        )

        level_map = {lvl: i for i, lvl in enumerate(levels_sorted)}
        final_plot_df["level_idx"] = final_plot_df["level"].map(level_map)

        fig, ax = plt.subplots(figsize=(12, 6))
        # plot_df_model = (
        #     plot_df.groupby(
        #         ["model_id", question_col, "level", "sub_category"], observed=True
        #     )["accuracy"]
        #     .mean()
        #     .reset_index()
        #     .dropna()
        # )
        # final_plot_df_model = (
        #     plot_df_model.groupby(["model_id", "level"], observed=True)["accuracy"]
        #     .agg(mean_accuracy="mean", band_width="sem")
        #     .reset_index()
        #     .dropna()
        # )
        # final_plot_df_model["level_idx"] = final_plot_df_model["level"].map(level_map)
        # sns.violinplot(
        #     data=final_plot_df_model,
        #     x="level_idx",
        #     y="mean_accuracy",
        #     ax=ax,
        #     color="0.85",
        #     # inner="box",
        #     inner=None,
        #     cut=0,
        #     width=1.0,
        #     linewidth=0.5,
        #     order=list(range(len(levels_sorted))),
        # )

        for i, (family, fam_data) in enumerate(final_plot_df.groupby("family")):
            fam_data = fam_data.sort_values("level_idx")
            if fam_data.empty:
                continue
            
            jitter = rng.uniform(-0.15, 0.15, size=fam_data["level_idx"].values[1:].size)
            x = fam_data["level_idx"].values[1:] + jitter
            y = fam_data["mean_accuracy"].values[1:]
            baseline_acc =fam_data[fam_data["level_idx"] == 0]["mean_accuracy"].values
            y = y - baseline_acc

            y_err = fam_data["band_width"].values
            color, marker, _size = family_style.get(family, ("black", "o", 0.5))
            ls = line_styles[i % len(line_styles)]
            # ax.plot(
            #     x, y, color=color, linestyle=ls, linewidth=2, alpha=0.85, label=family
            # )
            ax.scatter(
                x, y, color=color, marker=marker, s=_size**2, edgecolor="white", linewidth=1, zorder=4, label=family
            )
            # if np.isfinite(y_err).any():
            #     ax.fill_between(x, y - y_err, y + y_err, color=color, alpha=0.12)
        
        n = 10
        y_min, y_max = ax.get_ylim()
        for i in range(n):
            ax.axhspan(y_min*(i+1)/n, y_min*i/n, facecolor="#ffcccc", alpha=0.5*i/n, zorder=1)
        for i in range(n):
            ax.axhspan(y_max*i/n, y_max*(i+1)/n, facecolor="#ccffcc", alpha=0.5*i/n, zorder=1)
        ax.set_ylim(y_min, y_max)
        ax.axhline(0, color="#555555", alpha=1, zorder=3, linewidth=2)

        ax.set_xticks(list(range(len(levels_sorted)))[1:])
        nice_labels = [level.capitalize() for level in levels_sorted]
        ax.set_xticklabels(nice_labels[1:], fontsize=11, fontweight="bold", rotation=30)
        ax.tick_params(axis='x', pad=-2)
        # ax.set_xlabel("Difficulty Level", fontsize=12)
        ax.set_ylabel("Change in accuracy", fontsize=12)
        ax.set_title(f"Performance by Family{title_suffix}", fontsize=14)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, linestyle="--", alpha=0.5)

        fig.tight_layout()
        utils_graph.paperformat(ax, figsize=(4.5, 3), grid=["y"], minor=False)

        yticks = np.arange(-2, 10, 2)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{'+' if i>0 else ''}{int(i)}" for i in yticks])
        colors = ["black" if y==0 else ("green" if y > 0 else "red") for y in yticks]
        for ticklabel, color in zip(ax.get_yticklabels(), colors):
            ticklabel.set_color(color)

        run = run_name or globals().get("RUN_NAME", "default")
        out_dir = Path(output_dir) if output_dir is not None else Path("output") / run
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / output_name, dpi=300, bbox_inches="tight", pad_inches=0.05)

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    if split_by_mode and mode_map:
        modes = modes or ["image-only", "general"]
        figures = []
        for mode in modes:
            mode_models = [mid for mid, m in mode_map.items() if m == mode]
            subset = eval_df[eval_df["model_id"].isin(mode_models)]
            if subset.empty:
                continue
            suffix = f" ({mode})"
            out_name = filename.replace(".png", f"_{_safe_filename(mode)}.png")
            figures.append(_plot_subset(subset, suffix, out_name))

            figures.append(_plot_baseline_improvement(subset, suffix, out_name.replace(".png", "_baselinerelative.png")))
        if figures:
            return figures[0]

    return _plot_subset(eval_df, "", filename)
