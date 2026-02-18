from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import matplotlib.markers as mmarkers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
import utils.utils_mapping

warnings.filterwarnings("ignore", message=".*edgecolor.*unfilled marker.*")

def paperformat(ax, figsize=(4, 3.1), ylim=None, ticks_step=10, grid=["x", "y"], minor=True):
    fig = ax.get_figure()
    if figsize is not None:
        fig.set_size_inches(*figsize)

    ax.set_title("")
    for label in ax.get_xticklabels():
        label.set_fontsize(13)
        label.set_ha('center')
        label.set_fontweight('bold')  # or 'normal', 'light', etc.
    
    for label in ax.get_yticklabels():
        label.set_fontsize(13)
        label.set_fontweight('bold')  # or 'normal', 'light', etc.

    for label in [ax.xaxis.label, ax.yaxis.label]:
        label.set_fontsize(14)
        label.set_fontweight("bold")

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if ylim is not None:
        ax.set_ylim(ylim)
    
    import matplotlib.ticker as mticker
    ax.yaxis.set_major_locator(mticker.MultipleLocator(ticks_step))
    if minor:
        ax.yaxis.set_minor_locator(mticker.MultipleLocator(ticks_step//2))

    ax.grid(False)
    if grid:
        for axis in grid:
            ax.grid(axis=axis, which="major", linestyle="-", alpha=0.5)
            ax.grid(axis=axis, which="minor", linestyle="-", alpha=0.1)
        

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



def create_num_objects_category_curve(
    eval_df: pd.DataFrame,
    *,
    metadata_path: str | Path | None = "utils/metadata.json",
    num_objects_col: str | None = None,
    category_column: str = "category",
    categories: list[str] | None = None,
    show: bool = False,
    output_dir: str | Path | None = None,
    run_name: str | None = None,
    filename: str = "num_objects_category_curve.png",
    y_pad: float = 0.05,
    y_limit_mode: str = "zero_to_max",
    legend_loc: str = "best",
    scatter: bool = False,
    colors: list[str] | None = None,
) -> plt.Figure:
    """
    Plot accuracy (in %) vs num_objects, one curve per category (6 total).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from pathlib import Path

    if category_column not in eval_df.columns or "model_id" not in eval_df.columns:
        raise KeyError(f"eval_df must include '{category_column}' and 'model_id' columns.")

    if num_objects_col is None:
        if "num_objects" in eval_df.columns:
            num_objects_col = "num_objects"
        elif "object_count" in eval_df.columns:
            num_objects_col = "object_count"
        else:
            raise KeyError("eval_df must include 'num_objects' or 'object_count'.")

    plot_df = eval_df.copy()
    plot_df["num_objects"] = pd.to_numeric(plot_df[num_objects_col], errors="coerce")

    if "accuracy" in plot_df.columns:
        plot_df["accuracy"] = pd.to_numeric(plot_df["accuracy"], errors="coerce")
    elif "is_correct" in plot_df.columns:
        plot_df["accuracy"] = pd.to_numeric(plot_df["is_correct"], errors="coerce")
    else:
        raise KeyError("eval_df must include 'accuracy' or 'is_correct'.")

    plot_df = plot_df.dropna(subset=["num_objects", "accuracy", category_column])

    # Convert accuracy to percentage
    plot_df["accuracy"] = plot_df["accuracy"] * 100

    # Get categories
    categories = list(plot_df[category_column].unique())
    if not categories:
        raise ValueError("No categories found in data.")

    # Aggregate: mean accuracy per (category, num_objects)
    agg_df = (
        plot_df.groupby([category_column, "num_objects"], observed=True)["accuracy"]
        .mean()
        .reset_index()
    )

    # Prepare plot
    fig, ax = plt.subplots(figsize=(7, 3.5))
    # Color palette
    # if colors is None:
    #     palette = sns.color_palette("tab10", len(categories))
    # else:
    #     palette = colors

    categories_sorted = sorted(categories, key=lambda x: utils.utils_mapping.mapping_cat_order.get(x, float('inf')))
    for i, cat in enumerate(categories_sorted):
        df_cat = agg_df[agg_df[category_column] == cat]
        if df_cat.empty:
            continue
        x = df_cat["num_objects"].values
        y = df_cat["accuracy"].values
        cat_label = utils.utils_mapping.mapping_cat_short.get(cat)
        cat_color = utils.utils_mapping.mapping_cat_colors.get(cat)
        if scatter:
            ax.scatter(x, y, color=cat_color, s=60, alpha=0.8, label=cat_label)
        else:
            ax.plot(x, y, marker="o", color=cat_color, linewidth=2, alpha=0.85, label=cat_label)

    ax.set_xlabel("Number of Objects", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    # ax.set_title("Accuracy vs Number of Objects by Category", fontsize=14, fontweight="bold")
    legend = ax.legend(  loc=legend_loc, 
                fontsize=9.5, 
                ncol=2, 
                markerscale=0.5,
                handletextpad=0.2,
                columnspacing=0.5,
                borderpad=0.2,
                frameon=True,
                handlelength=1.3 )
    for text, handle in zip(legend.get_texts(), legend.legend_handles):
        if hasattr(handle, "get_color"):
            text.set_color(handle.get_color())
            text.set_fontweight("bold")
        elif hasattr(handle, "get_facecolor"):
            text.set_color(handle.get_facecolor()[0])
    ax.grid(axis="y", alpha=0.3)

    # Set y limits
    y_min = agg_df["accuracy"].min()
    y_max = agg_df["accuracy"].max()
    if y_limit_mode == "zero_to_max":
        ax.set_ylim(0, y_max + y_pad * max(1e-6, y_max))
    elif y_limit_mode == "fit":
        pad = y_pad * max(1e-6, y_max - y_min)
        ax.set_ylim(y_min - pad, y_max + pad)
    elif y_limit_mode == "fixed":
        ax.set_ylim(15, 55)

    fig.tight_layout()
    paperformat(ax, grid=["y"], minor=True)

    ax.set_xticks(range(1, 11))
    ax.set_xticklabels([str(v) for v in range(1, 11)])


    # Save
    if output_dir is not None:
        run = run_name or "default"
        out_dir = Path(output_dir) / run
        out_dir.mkdir(parents=True, exist_ok=True)
        f_out = out_dir / filename
        fig.savefig(f_out, dpi=150, bbox_inches="tight", pad_inches=0.05)
        print(f"Plot saved to: {f_out}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig

def create_num_objects_violin_grid(
    eval_df: pd.DataFrame,
    *,
    metadata_path: str | Path | None = "utils/metadata.json",
    num_objects_col: str | None = None,
    count_model_substring: str | None = None,
    subcategory_label_map: dict[str, str] | None = None,
    n_cols: int = 4,
    seed: int = 0,
    show: bool = False,
    output_dir: str | Path | None = None,
    run_name: str | None = None,
    filename: str = "num_objects_violin_grid.png",
    group_by: str = "model_id",
    save_per_category: bool = False,
    per_category_dirname: str | None = None,
    save_grid: bool = False,
    save_legend: bool = False,
    legend_filename: str | None = None,
    legend_cols: int = 4,
    legend_figsize: tuple[float, float] | None = None,
    show_legend: bool = True,
    grid_hspace: float | None = 0.6,
    grid_wspace: float | None = None,
    use_tight_layout: bool = True,
    grid_top: float | None = None,
    grid_bottom: float | None = None,
    y_limit_mode: str = "zero_to_max",
    y_pad: float = 0.0,
    n_label_offset: float = 0.08,
    split_values: tuple[float, float] | None = (3, 4),
    split_line_kwargs: dict | None = None,
    sample_frac: float | None = None,
    sample_seed: int | None = None,
    family_marker_mode: str = "distinct",
    family_marker_base: str = "^",
    category_column: str = "sub_category",
) -> plt.Figure:
    if category_column not in eval_df.columns or "model_id" not in eval_df.columns:
        raise KeyError(f"eval_df must include '{category_column}' and 'model_id' columns.")

    if num_objects_col is None:
        if "num_objects" in eval_df.columns:
            num_objects_col = "num_objects"
        elif "object_count" in eval_df.columns:
            num_objects_col = "object_count"
        else:
            raise KeyError("eval_df must include 'num_objects' or 'object_count'.")

    plot_df = eval_df.copy()
    plot_df["num_objects"] = pd.to_numeric(plot_df[num_objects_col], errors="coerce")

    if "accuracy" in plot_df.columns:
        plot_df["accuracy"] = pd.to_numeric(plot_df["accuracy"], errors="coerce")
    elif "is_correct" in plot_df.columns:
        plot_df["accuracy"] = pd.to_numeric(plot_df["is_correct"], errors="coerce")
    else:
        raise KeyError("eval_df must include 'accuracy' or 'is_correct'.")

    plot_df["accuracy"] *= 100

    plot_df = plot_df.dropna(subset=["num_objects", "accuracy"])

    if sample_frac is not None and 0 < sample_frac < 1:
        sample_key = "question_id" if "question_id" in plot_df.columns else "idx"
        if sample_key in plot_df.columns:
            rng = np.random.default_rng(
                sample_seed if sample_seed is not None else seed
            )
            sampled_rows = []
            for (_cat, _num), df_group in plot_df.groupby(
                [category_column, "num_objects"], observed=True
            ):
                unique_ids = pd.unique(df_group[sample_key])
                if unique_ids.size == 0:
                    continue
                keep_n = max(1, int(math.ceil(sample_frac * unique_ids.size)))
                keep_ids = rng.choice(unique_ids, size=keep_n, replace=False)
                sampled_rows.append(df_group[df_group[sample_key].isin(keep_ids)])
            if sampled_rows:
                plot_df = pd.concat(sampled_rows, ignore_index=True)

    if count_model_substring:
        counts_source = plot_df[
            plot_df["model_id"]
            .astype(str)
            .str.contains(count_model_substring, na=False)
        ]
    else:
        dedup_cols = [
            c for c in ["idx", category_column, "num_objects"] if c in plot_df.columns
        ]
        counts_source = (
            plot_df.drop_duplicates(subset=dedup_cols) if dedup_cols else plot_df
        )

    sub_categories = pd.unique(plot_df[category_column])
    if sub_categories.size == 0:
        raise ValueError(f"No {category_column} values found after filtering.")

    model_style, family_map = utils.utils_mapping._build_model_style(
        # plot_df,
        metadata_path,
        group_by=group_by,
        family_marker_mode=family_marker_mode,
        # family_marker_base=family_marker_base,
    )
    rng = np.random.default_rng(seed)

    cols = max(1, n_cols)
    rows = math.ceil(len(sub_categories) / cols)

    use_constrained_layout = (not show_legend) and (grid_hspace is None)
    fig, axes = plt.subplots(
        rows, cols, figsize=(5 * cols, 4 * rows), constrained_layout=use_constrained_layout
    )
    axes = np.array(axes).flatten()

    global_limits = None
    if y_limit_mode == "zero_to_max":
        if group_by == "model_family":
            plot_df["group_id"] = plot_df["model_id"].map(family_map).fillna("Other")
        else:
            plot_df["group_id"] = plot_df["model_id"].astype(str)
        agg_all = _macro_avg_by_question(plot_df, ["group_id", "num_objects"])
        if not agg_all.empty:
            global_limits = (
                float(agg_all["accuracy"].min()),
                float(agg_all["accuracy"].max()),
            )

    for i, cat in enumerate(sub_categories):
        ax = axes[i]
        df_cat = plot_df[plot_df[category_column] == cat].copy()
        if df_cat.empty:
            ax.set_visible(False)
            continue

        if group_by == "model_family":
            df_cat["group_id"] = df_cat["model_id"].map(family_map).fillna("Other")
        else:
            df_cat["group_id"] = df_cat["model_id"].astype(str)

        agg_df = _macro_avg_by_question(df_cat, ["group_id", "num_objects"])
        if agg_df.empty:
            ax.set_visible(False)
            continue

        num_values = sorted(pd.unique(agg_df["num_objects"]))
        num_to_pos = {value: idx for idx, value in enumerate(num_values)}
        agg_df["num_objects_pos"] = agg_df["num_objects"].map(num_to_pos)

        sns.violinplot(
            data=agg_df,
            x="num_objects_pos",
            y="accuracy",
            ax=ax,
            color="0.85",
            inner=None,
            cut=0,
            order=list(range(len(num_values))),
            zorder=3
        )
        ax.axhline(y=25, color="gray", linestyle="--", linewidth=1)
        if (
            split_values
            and split_values[0] in num_to_pos
            and split_values[1] in num_to_pos
        ):
            left = num_to_pos[split_values[0]]
            right = num_to_pos[split_values[1]]
            line_x = (left + right) / 2
            ax.axvline(
                x=line_x,
                color="gray",
                linestyle=":",
                linewidth=1,
                **(split_line_kwargs or {}),
            )

        counts_series = (
            counts_source[counts_source[category_column] == cat]
            .groupby("num_objects", observed=True)["num_objects"]
            .count()
        )
        if global_limits:
            label_y = global_limits[0] - n_label_offset * max(
                1e-6, global_limits[1] - global_limits[0]
            )
        else:
            label_y = -0.075
        # for num_obj, count in counts_series.items():
        #     x_pos = num_to_pos.get(num_obj)
        #     if x_pos is None:
        #         continue
        #     ax.text(
        #         x_pos,
        #         label_y,
        #         f"N:{int(count)}",
        #         ha="center",
        #         va="bottom",
        #         fontsize=9,
        #         color="black",
        #     )

        for group_id, df_m in agg_df.groupby("group_id"):
            x_vals = df_m["num_objects_pos"].to_numpy()
            y_vals = df_m["accuracy"].to_numpy()
            if x_vals.size == 0:
                continue
            jitter = rng.uniform(-0.2, 0.2, size=x_vals.size)
            x_jittered = x_vals + jitter
            color, marker, size = model_style[group_id]
            ax.scatter(
                x_jittered,
                y_vals,
                color=color,
                s=50+ 100 * size,
                alpha=0.8,
                edgecolor="white",
                linewidth=1,
                marker=marker,
                zorder=4
            )

        label = (
            subcategory_label_map.get(str(cat), str(cat))
            if subcategory_label_map
            else str(cat)
        )
        ax.set_title(label)
        ax.set_xlabel("Number of Objects")
        # ax.set_ylabel("Accuracy")
        ax.grid(axis="y")
        if y_limit_mode == "fit":
            y_min = agg_df["accuracy"].min()
            y_max = agg_df["accuracy"].max()
            if pd.isna(y_min) or pd.isna(y_max):
                ax.set_ylim(-10, 110)
            else:
                pad = y_pad * max(1e-6, y_max - y_min)
                ax.set_ylim(y_min - pad, y_max + pad)
        elif y_limit_mode == "fixed":
            ax.set_ylim(-10, 110)
        elif y_limit_mode == "zero_to_max" and global_limits:
            y_min, y_max = global_limits
            pad = y_pad * max(1e-6, y_max - y_min)
            ax.set_ylim(y_min - n_label_offset * max(1e-6, y_max - y_min), y_max + pad)
        else:
            y_min = agg_df["accuracy"].min()
            y_max = agg_df["accuracy"].max()
            if pd.isna(y_min) or pd.isna(y_max):
                ax.set_ylim(-10, 110)
            else:
                pad = y_pad * max(1e-6, y_max - y_min)
                ax.set_ylim(y_min - pad, y_max + pad)
        
        cap_min, cap_max = -100/100, 100+100/100
        if ax.get_ylim()[0] < cap_min:
            ax.set_ylim(cap_min, ax.get_ylim()[1])
        if ax.get_ylim()[1] > cap_max:
            ax.set_ylim(ax.get_ylim()[0], cap_max)
            
        ax.set_xticks(range(len(num_values)))
        ax.set_xticklabels([str(v) for v in num_values])

        paperformat(ax)

    for j in range(len(sub_categories), len(axes)):
        axes[j].set_visible(False)

    legend_handles = []
    legend_labels = []
    for group_id, (color, marker, size) in model_style.items():
        marker_style = mmarkers.MarkerStyle(marker)
        marker_face = color if marker_style.is_filled() else "none"
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                color="none",
                markerfacecolor=marker_face,
                markeredgecolor=color,
                markersize=8 + 6 * size,
                linestyle="None",
            )
        )
        legend_labels.append(group_id)

    if show_legend:
        fig.legend(
            legend_handles,
            legend_labels,
            title="Model",
            loc="upper center",
            ncol=5,
            bbox_to_anchor=(0.5, 1.02),
        )
        if use_tight_layout and y_limit_mode != "fit":
            plt.tight_layout(rect=[0, 0, 1, 0.96])
    elif not use_constrained_layout and use_tight_layout and y_limit_mode != "fit":
        plt.tight_layout()
    if grid_hspace is not None or grid_wspace is not None:
        fig.subplots_adjust(
            hspace=grid_hspace if grid_hspace is not None else 0.2,
            wspace=grid_wspace if grid_wspace is not None else 0.2,
        )
    if y_limit_mode == "fit":
        fig.subplots_adjust(
            top=grid_top if grid_top is not None else (0.94 if show_legend else 0.96),
            bottom=grid_bottom if grid_bottom is not None else 0.1,
        )
    
    paperformat(fig.gca())

    run = run_name or globals().get("RUN_NAME", "default")
    out_dir = Path(output_dir) if output_dir is not None else Path("output") / run
    seed_tag = f"_seed_{sample_seed}" if sample_seed is not None else ""
    out_dir.mkdir(parents=True, exist_ok=True)
    if save_grid:
        f_out = out_dir / f"{Path(filename).stem}{seed_tag}{Path(filename).suffix}"
        fig.savefig(
            f_out,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.05,
        )
        print("Plot saved to:", f_out)

    if save_legend:
        fig_legend = plt.figure(figsize=legend_figsize or (5 * legend_cols, 1.0))
        fig_legend.legend(
            legend_handles,
            legend_labels,
            title="Model",
            loc="center",
            ncol=legend_cols,
            frameon=False,
        )
        fig_legend.tight_layout()
        legend_name = legend_filename or filename.replace(".png", "_legend.png")
        f_out = out_dir / f"{Path(legend_name).stem}{seed_tag}{Path(legend_name).suffix}"
        fig_legend.savefig(
            f_out,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.05,
        )
        print("Plot saved to:", f_out)
        plt.close(fig_legend)

    if save_per_category:
        sub_dirname = per_category_dirname or filename.replace(".png", "")
        per_cat_dir = out_dir / sub_dirname
        per_cat_dir.mkdir(parents=True, exist_ok=True)
        for cat in sub_categories:
            fig_cat, ax_cat = plt.subplots(1, 1, figsize=(4, 3.1))
            df_cat = plot_df[plot_df[category_column] == cat].copy()
            if df_cat.empty:
                plt.close(fig_cat)
                continue

            if group_by == "model_family":
                df_cat["group_id"] = df_cat["model_id"].map(family_map).fillna("Other")
            else:
                df_cat["group_id"] = df_cat["model_id"].astype(str)

            agg_df = _macro_avg_by_question(df_cat, ["group_id", "num_objects"])
            if agg_df.empty:
                plt.close(fig_cat)
                continue

            num_values = sorted(pd.unique(agg_df["num_objects"]))
            num_to_pos = {value: idx for idx, value in enumerate(num_values)}
            agg_df["num_objects_pos"] = agg_df["num_objects"].map(num_to_pos)

            sns.violinplot(
                data=agg_df,
                x="num_objects_pos",
                y="accuracy",
                ax=ax_cat,
                color="0.85",
                inner=None,
                cut=0,
                order=list(range(len(num_values))),
                zorder=3
            )
            ax_cat.axhline(y=25, color="gray", linestyle="--", linewidth=1)
            if (
                split_values
                and split_values[0] in num_to_pos
                and split_values[1] in num_to_pos
            ):
                left = num_to_pos[split_values[0]]
                right = num_to_pos[split_values[1]]
                line_x = (left + right) / 2
                ax_cat.axvline(
                    x=line_x,
                    color="gray",
                    linestyle=":",
                    linewidth=1,
                    **(split_line_kwargs or {}),
                )

            counts_series = (
                counts_source[counts_source[category_column] == cat]
                .groupby("num_objects", observed=True)["num_objects"]
                .count()
            )
            label_y = agg_df["accuracy"].min() - n_label_offset * max(
                1e-6, agg_df["accuracy"].max() - agg_df["accuracy"].min()
            )
            # for num_obj, count in counts_series.items():
            #     x_pos = num_to_pos.get(num_obj)
            #     if x_pos is None:
            #         continue
            #     ax_cat.text(
            #         x_pos,
            #         label_y,
            #         f"N:{int(count)}",
            #         ha="center",
            #         va="bottom",
            #         fontsize=9,
            #         color="black",
            #     )

            for group_id, df_m in agg_df.groupby("group_id"):
                x_vals = df_m["num_objects_pos"].to_numpy()
                y_vals = df_m["accuracy"].to_numpy()
                if x_vals.size == 0:
                    continue
                jitter = rng.uniform(-0.2, 0.2, size=x_vals.size)
                x_jittered = x_vals + jitter
                color, marker, size = model_style[group_id]
                # print(f"Plotting group '{group_id}' with color {color}, marker {marker}, size {size}")
                ax_cat.scatter(
                    x_jittered,
                    y_vals,
                    color=color,
                    s=50+ 100 * size,
                    alpha=0.8,
                    edgecolor="white",
                    linewidth=1,
                    marker=marker,
                    zorder=4
                )

            label = (
                subcategory_label_map.get(str(cat), str(cat))
                if subcategory_label_map
                else str(cat)
            )
            # ax_cat.set_title(label)
            ax_cat.set_xlabel("Number of objects")
            # ax_cat.set_ylabel("Accuracy")
            if cat != "all":
                ylabel = utils.utils_mapping.mapping_cat_short.get(cat)
                ylabel_color = utils.utils_mapping.mapping_cat_colors.get(cat)+"CC"
            else:
                ylabel = "Overall accuracy"
                ylabel_color = "black"
            ax_cat.set_ylabel(ylabel.capitalize(), color=ylabel_color)
            ax_cat.grid(axis="y")
            if y_limit_mode == "fit":
                y_min = agg_df["accuracy"].min()
                y_max = agg_df["accuracy"].max()
                if pd.isna(y_min) or pd.isna(y_max):
                    ax_cat.set_ylim(-10, 110)
                else:
                    pad = y_pad * max(1e-6, y_max - y_min)
                    ax_cat.set_ylim(y_min - pad, y_max + pad)
            elif y_limit_mode == "fixed":
                ax_cat.set_ylim(-10, 110)
            elif y_limit_mode == "zero_to_max":
                y_min = float(agg_df["accuracy"].min())
                y_max = float(agg_df["accuracy"].max())
                pad = y_pad * max(1e-6, y_max - y_min)
                ax_cat.set_ylim(
                    y_min - n_label_offset * max(1e-6, y_max - y_min), y_max + pad
                )
            else:
                y_min = agg_df["accuracy"].min()
                y_max = agg_df["accuracy"].max()
                if pd.isna(y_min) or pd.isna(y_max):
                    ax_cat.set_ylim(-10, 110)
                else:
                    pad = y_pad * max(1e-6, y_max - y_min)
                    ax_cat.set_ylim(y_min - pad, y_max + pad)

            cap_min, cap_max = 0, 100+100/100
            if ax_cat.get_ylim()[0] < cap_min:
                ax_cat.set_ylim(cap_min, ax_cat.get_ylim()[1])
            if ax_cat.get_ylim()[1] > cap_max:
                ax_cat.set_ylim(ax_cat.get_ylim()[0], cap_max)

            ax_cat.set_xticks(range(len(num_values)))
            ax_cat.set_xticklabels([str(v) for v in num_values])

            fig_cat.tight_layout()
            paperformat(ax_cat)

            safe_cat = _safe_filename(str(cat))
            if category_column == "sub_category":
                assert plot_df[plot_df[category_column] == cat]["category"].unique().size == 1
                safe_cat = plot_df[plot_df[category_column] == cat]["category"].unique()[0] + "_" + safe_cat
            f_out = per_cat_dir / f"num_objects_{safe_cat}{seed_tag}.png"
            fig_cat.savefig(
                f_out,
                dpi=300,
                bbox_inches="tight",
            )
            print("Plot saved to:", f_out)
            plt.close(fig_cat)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def create_category_accuracy(
    eval_df: pd.DataFrame,
    *,
    metadata_path: str | Path | None = "utils/metadata.json",
    category_col: str = "category",
    category_label_map: dict[str, str] | None = None,
    seed: int = 0,
    show: bool = False,
    output_dir: str | Path | None = None,
    run_name: str | None = None,
    filename: str = "accuracy_by_category.png",
    figsize: tuple[float, float] = (8.5, 4),
    show_legend: bool = False,
    y_limit_mode: str = "fixed",
    y_pad: float = 0.05,
    family_marker_mode: str = "distinct",
    group_by: str = "model_id",
    bars: bool = False,
) -> plt.Figure:
    """
    Create a violin plot showing model accuracy by category.
    
    Parameters:
    -----------
    eval_df : pd.DataFrame
        DataFrame with columns: model_id, category (or specified category_col), is_correct/accuracy
    category_col : str
        Name of the column containing categories
    category_label_map : dict, optional
        Mapping from category names to display labels
    """
    if category_col not in eval_df.columns or "model_id" not in eval_df.columns:
        raise KeyError(f"eval_df must include '{category_col}' and 'model_id' columns.")

    plot_df = eval_df.copy()

    plot_df["accuracy"] *= 100

    # Compute accuracy for each model-category combination (keeping model_family)
    group_by_cols = np.unique(["model_id", group_by, category_col])
    agg_cat_accuracy = (
        plot_df.groupby(list(group_by_cols))["accuracy"]
        .mean()
        .reset_index()
    )

    # Compute mean, min, max per family-category combination
    agg_df = (
        agg_cat_accuracy.groupby([group_by, category_col])["accuracy"]
        .agg(['mean', 'min', 'max', 'std', 'count'])
        .reset_index()
    )

    if agg_df.empty:
        raise ValueError("No data available after aggregation.")
    
    # Get unique categories and sort them
    # categories = sorted(pd.unique(agg_df[category_col]))
    categories = ["material_understanding", "mechanics", "spatial_reasoning", "view_point", "persistence", "temporal"]
    categories = [cat for cat in categories if cat in plot_df[category_col].unique()]
    cat_to_pos = {cat: idx for idx, cat in enumerate(categories)}
    agg_df["category_pos"] = agg_df[category_col].map(cat_to_pos)

    # Build model style
    model_style, family_map = utils.utils_mapping._build_model_style(
        # plot_df,
        metadata_path,
        group_by=group_by,
        family_marker_mode=family_marker_mode,
    )
    rng = np.random.default_rng(seed)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create violin plot (commented out)
    sns.violinplot(
        data=agg_df,
        x="category_pos",
        y="mean",
        ax=ax,
        color="0.85",
        # inner="box",
        inner=None,
        cut=0,
        width=1.0,
        linewidth=0.5,
        order=list(range(len(categories))),
    )
    ax.set_xlabel("")

    # Add reference line at chance level
    # ax.axhline(y=25, color="gray", linestyle="--", linewidth=1, label="Chance")

    # Plot scatter points with error bars for each family
    for group_name, df_m in agg_df.groupby(group_by):
        x_vals = df_m["category_pos"].to_numpy()
        y_vals = df_m["mean"].to_numpy()
        y_min = df_m["min"].to_numpy()
        y_max = df_m["max"].to_numpy()
        
        if x_vals.size == 0:
            continue
        
        # Calculate asymmetric error bars: [lower_error, upper_error]
        y_err = np.array([y_vals - y_min, y_max - y_vals])
        
        jitter = rng.uniform(-0.15, 0.15, size=x_vals.size)
        x_jittered = x_vals + jitter
        color, marker, size = model_style[group_name]
        
        # Plot error bars (min to max range)
        # ax.errorbar(
        #     x_jittered,
        #     y_vals,
        #     yerr=y_err,
        #     fmt='none',
        #     ecolor=color,
        #     alpha=0.5,
        #     capsize=3,
        #     capthick=3.0,
        #     linewidth=2.0,
        #     zorder=2
        # )
        
        # Plot scatter points
        ax.scatter(
            x_jittered,
            y_vals,
            color=color,
            s=75+ 200 * size,
            alpha=0.8,
            edgecolor="white",
            linewidth=1,
            marker=marker,
            zorder=4
        )

    # "#AED6F1"
    # "#A9DFBF"
    for r_x_start, r_x_end, r_color, r_label in ((-0.5,3.5, None, "Image and Video Models"),
                                                (3.5,5.6, None, "Video models")):
        # Add semi-transparent background
        mid_x = (r_x_start + r_x_end) / 2  # arithmetic mean

        if r_color is not None:
            ax.axvspan(r_x_start, r_x_end, alpha=0.15, color=r_color, zorder=-1, linewidth=0)
        
        # Add vertical lines at boundaries
        if r_color is None:
            r_color =  "#999999"
        ax.axvline(r_x_start, color=r_color, linewidth=2, alpha=0.5, zorder=-1, linestyle='--')
        ax.axvline(r_x_end, color=r_color, linewidth=2, alpha=0.5, zorder=-1, linestyle='--')
        ax.text(mid_x, ax.get_ylim()[1] * 0.95, r_label, 
                ha='center', va='top', fontsize=12, color='#555555', fontweight='bold')

    ax.set_xlim(0.-0.5, len(categories) - 1 + 0.5)
    # Set labels
    # ax.set_xlabel(fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ax.set_xlabel(fontsize=12)
    # ax.spines['left'].set_visible(False)

    # ax.set_title("Model Accuracy by Category", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Set y-limits
    # if y_limit_mode == "fixed":
    # ax.set_ylim(-0.05, 1.05)
    # else:
    #     y_min = agg_df["accuracy"].min()
    #     y_max = agg_df["accuracy"].max()
    #     if pd.isna(y_min) or pd.isna(y_max):
    #         ax.set_ylim(-0.05, 1.05)
    #     else:
    #         pad = y_pad * max(1e-6, y_max - y_min)
    #         ax.set_ylim(y_min - pad, y_max + pad)

    # Set x-axis labels
    # Compute mean accuracy per category (averaged across all families)

    cat_accuracy = (
        plot_df.groupby([category_col])["accuracy"]
        .mean()
        .reset_index()
    )

    question_df = plot_df.groupby("question_id")["accuracy"].mean()

    def get_cat_label(cat):
        # print(cat)
        q_ids = plot_df[plot_df.loc[:, category_col] == cat]["question_id"].unique()

        label = utils.utils_mapping.mapping_cat.get(cat).replace(" ", "\n")
        label += f"\n"
        label += f"({cat_accuracy.loc[cat_accuracy[category_col] == cat, 'accuracy'].values[0]:.1f}%)"
        # perf = []
        # for q_id in q_ids:
        #     perf+= [question_df.loc[q_id]]
        #     print(q_id, question_df.loc[q_id])
        
        # perf = np.array(perf)
        # label += " ["
        # label += f", {np.std(perf):.2f}]"
        return label

    category_labels = [ get_cat_label(cat) for cat in categories ]
    # print(category_labels)

    # ax.set_xticklabels(category_labels, rotation=45, ha="right")
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(category_labels, ha='center', fontweight='bold', fontsize=9.5)

    for ticklabel, cat in zip(ax.get_xticklabels(), categories):
        ticklabel.set_color(utils.utils_mapping.mapping_cat_colors.get(cat)+"CC")  # Adding transparency
    # ax.set_xticklabels(category_labels, ha="center")


    for label in [ax.xaxis.label, ax.yaxis.label]:
        label.set_fontweight("bold")

    # Add legend if requested
    if show_legend:
        legend_handles = []
        legend_labels = []
        groups = plot_df[group_by].unique()
        for group_name, (color, marker, size_val) in model_style.items():
            if group_name not in groups:
                continue

            marker_style = mmarkers.MarkerStyle(marker)
            marker_face = color if marker_style.is_filled() else "none"

            models_num = len(plot_df[plot_df[group_by] == group_name]["model_id"].unique())
            if models_num > 1:
                group_name = f"{group_name} (x{models_num})"
            legend_handles.append(
                Line2D(
                    [0], [0],
                    marker=marker,
                    color="none",
                    markerfacecolor=marker_face,
                    markeredgecolor=color,
                    markersize=8 + 6 * size_val,
                    linestyle="None",
                )
            )
            legend_labels.append(group_name)

        title_str = {"model_id": "Model", 
                     "model_family": "Model Family"}.get(group_by)
        ax.legend(legend_handles, legend_labels, title=title_str, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, title_fontsize=9, markerscale=0.7)

    plt.tight_layout()

    # Save figure
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        f_out = out_dir / filename
        fig.savefig(f_out, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {f_out}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def create_material_stiffness_violin_grid(
    eval_df: pd.DataFrame,
    *,
    metadata_path: str | Path | None = "utils/metadata.json",
    stiffness_col: str = "object-yms",
    count_model_substring: str | None = None,
    subcategory_label_map: dict[str, str] | None = None,
    n_cols: int = 4,
    col_width: float = 5.0,
    row_height: float = 4.5,
    seed: int = 0,
    show: bool = False,
    output_dir: str | Path | None = None,
    run_name: str | None = None,
    filename: str = "yms_violin.png",
    group_by: str = "model_id",
    save_per_category: bool = False,
    per_category_dirname: str | None = None,
    save_grid: bool = False,
    save_legend: bool = False,
    legend_filename: str | None = None,
    legend_cols: int = 4,
    legend_figsize: tuple[float, float] | None = None,
    show_legend: bool = True,
    grid_hspace: float | None = 0.5,
    grid_wspace: float | None = None,
    use_tight_layout: bool = True,
    y_limit_mode: str = "zero_to_max",
    category_col: str = "category",
    y_pad: float = 0.0,
    n_label_offset: float = 0.08,
    stiffness_labels: tuple[str, str, str] = ("Soft\n($\\text{yms} \leq 2e4$)", "Medium\n($2e4 > \\text{yms} \leq 1e6$)", "Stiff\n($\\text{yms} > 1e6$)"),
    family_marker_mode: str = "distinct",
    family_marker_base: str = "^",
) -> plt.Figure:
    if "sub_category" not in eval_df.columns or "model_id" not in eval_df.columns:
        raise KeyError("eval_df must include 'sub_category' and 'model_id' columns.")
    if stiffness_col not in eval_df.columns:
        raise KeyError(f"eval_df must include '{stiffness_col}'.")

    plot_df = eval_df.copy()
    # plot_df["stiffness_level"] = pd.to_numeric(
    #     plot_df[stiffness_col], errors="coerce"
    # )
    mapping = {"soft": 0, "medium": 1, "stiff": 2}
    plot_df["stiffness_level"] = plot_df[stiffness_col].map(mapping)

    if "accuracy" in plot_df.columns:
        plot_df["accuracy"] = pd.to_numeric(plot_df["accuracy"], errors="coerce")
    elif "is_correct" in plot_df.columns:
        plot_df["accuracy"] = pd.to_numeric(plot_df["is_correct"], errors="coerce")
    else:
        raise KeyError("eval_df must include 'accuracy' or 'is_correct'.")

    plot_df["accuracy"] *= 100
    plot_df = plot_df.dropna(subset=["stiffness_level", "accuracy"])

    if count_model_substring:
        counts_source = plot_df[
            plot_df["model_id"]
            .astype(str)
            .str.contains(count_model_substring, na=False)
        ]
    else:
        dedup_cols = [
            c for c in ["idx", "sub_category", "stiffness_level"] if c in plot_df.columns
        ]
        counts_source = (
            plot_df.drop_duplicates(subset=dedup_cols) if dedup_cols else plot_df
        )
    
    categories = pd.unique(plot_df[category_col])
    if categories.size == 0:
        raise ValueError("No sub_category values found after filtering.")

    model_style, family_map = utils.utils_mapping._build_model_style(
        # plot_df,
        metadata_path,
        group_by=group_by,
        family_marker_mode=family_marker_mode,
        # family_marker_base=family_marker_base,
    )
    rng = np.random.default_rng(seed)

    cols = max(1, n_cols)
    rows = math.ceil(len(categories) / cols)

    fig, axes = plt.subplots(
        rows, cols, figsize=(col_width * cols, row_height * rows)
    )
    axes = np.array(axes).flatten()

    global_limits = None
    if y_limit_mode == "zero_to_max":
        if group_by == "model_family":
            plot_df["group_id"] = plot_df["model_id"].map(family_map).fillna("Other")
        else:
            plot_df["group_id"] = plot_df["model_id"].astype(str)
        agg_all = _macro_avg_by_question(plot_df, ["group_id", "stiffness_level"])
        if not agg_all.empty:
            global_limits = (
                float(agg_all["accuracy"].min()),
                float(agg_all["accuracy"].max()),
            )

    for i, cat in enumerate(categories):
        ax = axes[i]
        df_cat = plot_df[plot_df[category_col] == cat].copy()
        if df_cat.empty:
            ax.set_visible(False)
            continue

        if group_by == "model_family":
            df_cat["group_id"] = df_cat["model_id"].map(family_map).fillna("Other")
        else:
            df_cat["group_id"] = df_cat["model_id"].astype(str)

        agg_df = _macro_avg_by_question(df_cat, ["group_id", "stiffness_level"])
        if agg_df.empty:
            ax.set_visible(False)
            continue

        stiffness_values = sorted(pd.unique(agg_df["stiffness_level"]))
        stiffness_to_pos = {value: idx for idx, value in enumerate(stiffness_values)}
        agg_df["stiffness_pos"] = agg_df["stiffness_level"].map(stiffness_to_pos)

        sns.violinplot(
            data=agg_df,
            x="stiffness_pos",
            y="accuracy",
            ax=ax,
            color="0.85",
            # inner="box",
            inner=None,
            cut=0,
            width=1.0,
            linewidth=0.5,
            order=list(range(len(stiffness_values))),
            zorder=3
        )
        ax.axhline(y=25, color="gray", linestyle="--", linewidth=1, zorder=2)

        counts_series = (
            counts_source[counts_source[category_col] == cat]
            .groupby("stiffness_level", observed=True)["stiffness_level"]
            .count()
        )
        if y_limit_mode == "zero_to_max":
            label_y = -0.08
            label_transform = ax.get_xaxis_transform()
        elif global_limits:
            label_y = global_limits[0] - n_label_offset * max(
                1e-6, global_limits[1] - global_limits[0]
            )
            label_transform = None
        else:
            label_y = -0.075
            label_transform = None
        for level, count in counts_series.items():
            x_pos = stiffness_to_pos.get(level)
            if x_pos is None:
                continue
            # ax.text(
            #     x_pos,
            #     label_y,
            #     f"N:{int(count)}",
            #     ha="center",
            #     va="bottom",
            #     fontsize=9,
            #     color="black",
            #     transform=label_transform,
            # )

        for group_id, df_m in agg_df.groupby("group_id"):
            x_vals = df_m["stiffness_pos"].to_numpy()
            y_vals = df_m["accuracy"].to_numpy()
            if x_vals.size == 0:
                continue
            jitter = rng.uniform(-0.2, 0.2, size=x_vals.size)
            x_jittered = x_vals + jitter
            color, marker, size = model_style[group_id]
            ax.scatter(
                x_jittered,
                y_vals,
                color=color,
                s=75+ 200 * size,
                alpha=0.8,
                edgecolor="white",
                linewidth=1,
                marker=marker,
                zorder=4,
            )

        label = (
            subcategory_label_map.get(str(cat), str(cat))
            if subcategory_label_map
            else str(cat)
        )
        ax.set_title(label)
        # ax.set_xlabel("Object Stiffness")
        ax.set_xlabel("")
        ax.set_ylabel("Accuracy")
        ax.grid(axis="y")
        if y_limit_mode == "fixed":
            ax.set_ylim(-10, 110)
        elif y_limit_mode == "zero_to_max" and global_limits:
            y_min, y_max = global_limits
            pad = y_pad * max(1e-6, y_max - y_min)
            ax.set_ylim(y_min - n_label_offset * max(1e-6, y_max - y_min), y_max + pad)
        elif y_limit_mode == "fit":            
            y_min = agg_df["accuracy"].min()
            y_max = agg_df["accuracy"].max()
            pad = y_pad * max(1e-6, y_max - y_min)
            ax.set_ylim(y_min - pad, y_max + pad)
        else:
            y_min = agg_df["accuracy"].min()
            y_max = agg_df["accuracy"].max()
            if pd.isna(y_min) or pd.isna(y_max):
                ax.set_ylim(-10, 110)
            else:
                pad = y_pad * max(1e-6, y_max - y_min)
                ax.set_ylim(y_min - pad, y_max + pad)
        ax.set_xticks(range(len(stiffness_values)))
        
        if len(stiffness_values) == len(stiffness_labels):
            ax.set_xticklabels(list(stiffness_labels))
        else:
            ax.set_xticklabels([str(v) for v in stiffness_values])

        paperformat(ax)

    for j in range(len(categories), len(axes)):
        axes[j].set_visible(False)

    legend_handles = []
    legend_labels = []
    for group_id, (color, marker, size) in model_style.items():
        marker_style = mmarkers.MarkerStyle(marker)
        marker_face = color if marker_style.is_filled() else "none"
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                color="none",
                markerfacecolor=marker_face,
                markeredgecolor=color,
                markersize=8 + 6 * size,
                linestyle="None",
            )
        )
        legend_labels.append(group_id)

    if show_legend:
        fig.legend(
            legend_handles,
            legend_labels,
            title="Model",
            loc="upper center",
            ncol=5,
            bbox_to_anchor=(0.5, 1.02),
        )
        if use_tight_layout:
            plt.tight_layout(rect=[0, 0, 1, 0.96])
    elif use_tight_layout:
        plt.tight_layout()
    if grid_hspace is not None or grid_wspace is not None:
        fig.subplots_adjust(
            hspace=grid_hspace if grid_hspace is not None else 0.2,
            wspace=grid_wspace if grid_wspace is not None else 0.2,
        )

    run = run_name or globals().get("RUN_NAME", "default")
    out_dir = Path(output_dir) if output_dir is not None else Path("output") / run / group_by
    # out_dir = out_dir / category_col
    out_dir.mkdir(parents=True, exist_ok=True)
    if save_grid:
        bbox = None if y_limit_mode == "fit" else "tight"
        print("Saving grid plot to:", out_dir / filename.replace(".png", f"_{category_col}.png"))
        fig.savefig(
            out_dir / filename.replace(".png", f"_{category_col}.png"),
            dpi=300,
            bbox_inches=bbox,
            pad_inches=0.05,
        )

    if save_legend:
        fig_legend = plt.figure(figsize=legend_figsize or (5 * legend_cols, 1.0))
        fig_legend.legend(
            legend_handles,
            legend_labels,
            title="Model",
            loc="center",
            ncol=legend_cols,
            frameon=False,
        )
        fig_legend.tight_layout()
        legend_name = legend_filename or filename.replace(".png", f"_{category_col}_legend.png")
        print("Saving legend to:", out_dir / legend_name)
        fig_legend.savefig(
            out_dir / legend_name,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.05,
        )
        plt.close(fig_legend)

    if save_per_category:
        sub_dirname = per_category_dirname or filename.replace(".png", "")
        per_cat_dir = out_dir / sub_dirname
        per_cat_dir.mkdir(parents=True, exist_ok=True)
        for cat in categories:
            fig_cat, ax_cat = plt.subplots(1, 1, figsize=(4, 3.1))
            df_cat = plot_df[plot_df[category_col] == cat].copy()
            if df_cat.empty:
                plt.close(fig_cat)
                continue

            if group_by == "model_family":
                df_cat["group_id"] = df_cat["model_id"].map(family_map).fillna("Other")
            else:
                df_cat["group_id"] = df_cat["model_id"].astype(str)

            agg_df = _macro_avg_by_question(df_cat, ["group_id", "stiffness_level"])
            if agg_df.empty:
                plt.close(fig_cat)
                continue

            stiffness_values = sorted(pd.unique(agg_df["stiffness_level"]))
            stiffness_to_pos = {
                value: idx for idx, value in enumerate(stiffness_values)
            }
            agg_df["stiffness_pos"] = agg_df["stiffness_level"].map(stiffness_to_pos)

            sns.violinplot(
                data=agg_df,
                x="stiffness_pos",
                y="accuracy",
                ax=ax_cat,
                color="0.85",
                # inner="box",
                inner=None,
                cut=0,
                width=1.0,
                linewidth=0.5,
                order=list(range(len(stiffness_values))),
                zorder=3
            )
            ax_cat.axhline(y=25, color="gray", linestyle="--", linewidth=1, zorder=2)

            counts_series = (
                counts_source[counts_source[category_col] == cat]
                .groupby("stiffness_level", observed=True)["stiffness_level"]
                .count()
            )
            label_y = 0.02
            label_transform = ax_cat.get_xaxis_transform()
            # for level, count in counts_series.items():
            #     x_pos = stiffness_to_pos.get(level)
            #     if x_pos is None:
            #         continue
            #     ax_cat.text(
            #         x_pos,
            #         label_y,
            #         f"N:{int(count)}",
            #         ha="center",
            #         va="bottom",
            #         fontsize=9,
            #         color="black",
            #         transform=label_transform,
            #     )

            for group_id, df_m in agg_df.groupby("group_id"):
                x_vals = df_m["stiffness_pos"].to_numpy()
                y_vals = df_m["accuracy"].to_numpy()
                if x_vals.size == 0:
                    continue
                jitter = rng.uniform(-0.2, 0.2, size=x_vals.size)
                x_jittered = x_vals + jitter
                color, marker, size = model_style[group_id]
                ax_cat.scatter(
                    x_jittered,
                    y_vals,
                    color=color,
                    s=75+ 200 * size,
                    alpha=0.8,
                    edgecolor="white",
                    linewidth=1,
                    marker=marker,
                    zorder=4
                )

            label = (
                subcategory_label_map.get(str(cat), str(cat))
                if subcategory_label_map
                else str(cat)
            )
            # ax_cat.set_title(label)
            # ax_cat.set_xlabel("Material stiffness (Young's modulus level)")
            ax_cat.set_xlabel("")
            ax_cat.set_ylabel(utils.utils_mapping.mapping_cat.get(cat), color=utils.utils_mapping.mapping_cat_colors.get(cat))
            ax_cat.grid(axis="y")
            if y_limit_mode == "fixed":
                ax_cat.set_ylim(-10,110)
            elif y_limit_mode == "zero_to_max":
                y_min = float(agg_df["accuracy"].min())
                y_max = float(agg_df["accuracy"].max())
                pad = y_pad * max(1e-6, y_max - y_min)
                ax_cat.set_ylim(
                    y_min - n_label_offset * max(1e-6, y_max - y_min), y_max + pad
                )
            else:
                y_min = agg_df["accuracy"].min()
                y_max = agg_df["accuracy"].max()
                if pd.isna(y_min) or pd.isna(y_max):
                    ax_cat.set_ylim(-10,110)
                else:
                    pad = y_pad * max(1e-6, y_max - y_min)
                    ax_cat.set_ylim(y_min - pad, y_max + pad)
            ax_cat.set_xticks(range(len(stiffness_values)))
            if len(stiffness_values) == len(stiffness_labels):
                ax_cat.set_xticklabels(list(stiffness_labels))
            else:
                ax_cat.set_xticklabels([str(v) for v in stiffness_values])

            fig_cat.tight_layout()
            paperformat(ax_cat)
            safe_cat = _safe_filename(str(cat))
            if category_col == "sub_category":
                assert plot_df[plot_df[category_col] == cat]["category"].unique().size == 1
                safe_cat = plot_df[plot_df[category_col] == cat]["category"].unique()[0] + "_" + safe_cat
            bbox = None if y_limit_mode == "fit" else "tight"
            print("Saving per-category plot to:", per_cat_dir / f"yms_{safe_cat}.png")
            fig_cat.savefig(
                per_cat_dir / f"yms_{safe_cat}.png",
                dpi=300,
                bbox_inches=bbox,
                pad_inches=0.05,
            )
            plt.close(fig_cat)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def create_num_objects_violin_per_question_id(
    eval_df: pd.DataFrame,
    *,
    metadata_path: str | Path | None = "utils/metadata.json",
    num_objects_col: str | None = None,
    count_model_substring: str | None = None,
    question_label_map: dict[str, str] | None = None,
    seed: int = 0,
    show: bool = False,
    output_dir: str | Path | None = None,
    run_name: str | None = None,
    group_by: str = "model_id",
    per_question_dirname: str = "num_objects_per_question",
    save_legend: bool = False,
    legend_filename: str | None = None,
    legend_cols: int = 4,
    legend_figsize: tuple[float, float] | None = None,
    y_limit_mode: str = "zero_to_max",
    y_pad: float = 0.0,
    n_label_offset: float = 0.08,
    split_values: tuple[float, float] | None = (3, 4),
    split_line_kwargs: dict | None = None,
    family_marker_mode: str = "distinct",
    family_marker_base: str = "^",
) -> None:
    if "question_id" not in eval_df.columns or "model_id" not in eval_df.columns:
        raise KeyError("eval_df must include 'question_id' and 'model_id' columns.")

    if num_objects_col is None:
        if "num_objects" in eval_df.columns:
            num_objects_col = "num_objects"
        elif "object_count" in eval_df.columns:
            num_objects_col = "object_count"
        else:
            raise KeyError("eval_df must include 'num_objects' or 'object_count'.")

    plot_df = eval_df.copy()
    plot_df["num_objects"] = pd.to_numeric(plot_df[num_objects_col], errors="coerce")

    if "accuracy" in plot_df.columns:
        plot_df["accuracy"] = pd.to_numeric(plot_df["accuracy"], errors="coerce")
    elif "is_correct" in plot_df.columns:
        plot_df["accuracy"] = pd.to_numeric(plot_df["is_correct"], errors="coerce")
    else:
        raise KeyError("eval_df must include 'accuracy' or 'is_correct'.")

    plot_df = plot_df.dropna(subset=["num_objects", "accuracy"])

    if count_model_substring:
        counts_source = plot_df[
            plot_df["model_id"]
            .astype(str)
            .str.contains(count_model_substring, na=False)
        ]
    else:
        dedup_cols = [
            c for c in ["idx", "question_id", "num_objects"] if c in plot_df.columns
        ]
        counts_source = (
            plot_df.drop_duplicates(subset=dedup_cols) if dedup_cols else plot_df
        )

    question_ids = pd.unique(plot_df["question_id"])
    if question_ids.size == 0:
        raise ValueError("No question_id values found after filtering.")

    model_style, family_map = utils.utils_mapping._build_model_style(
        # plot_df,
        metadata_path,
        group_by=group_by,
        family_marker_mode=family_marker_mode,
        # family_marker_base=family_marker_base,
    )
    rng = np.random.default_rng(seed)

    global_limits = None
    if y_limit_mode == "zero_to_max":
        if group_by == "model_family":
            plot_df["group_id"] = plot_df["model_id"].map(family_map).fillna("Other")
        else:
            plot_df["group_id"] = plot_df["model_id"].astype(str)
        agg_all = _macro_avg_by_question(plot_df, ["group_id", "num_objects"])
        if not agg_all.empty:
            global_limits = (
                float(agg_all["accuracy"].min()),
                float(agg_all["accuracy"].max()),
            )

    run = run_name or globals().get("RUN_NAME", "default")
    out_dir = Path(output_dir) if output_dir is not None else Path("output") / run
    per_q_dir = out_dir / per_question_dirname
    per_q_dir.mkdir(parents=True, exist_ok=True)

    legend_handles = []
    legend_labels = []
    for group_id, (color, marker, size) in model_style.items():
        marker_style = mmarkers.MarkerStyle(marker)
        marker_face = color if marker_style.is_filled() else "none"
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                color="none",
                markerfacecolor=marker_face,
                markeredgecolor=color,
                markersize=8 + 6 * size,
                linestyle="None",
            )
        )
        legend_labels.append(group_id)

    if save_legend:
        fig_legend = plt.figure(figsize=legend_figsize or (5 * legend_cols, 1.0))
        fig_legend.legend(
            legend_handles,
            legend_labels,
            title="Model",
            loc="center",
            ncol=legend_cols,
            frameon=False,
        )
        fig_legend.tight_layout()
        legend_name = legend_filename or "num_objects_legend_question_id.png"
        f_out = out_dir / legend_name
        fig_legend.savefig(f_out, dpi=300, bbox_inches="tight")
        print("Plot saved to:", f_out)
        plt.close(fig_legend)

    for qid in question_ids:
        df_q = plot_df[plot_df["question_id"] == qid].copy()
        if df_q.empty:
            continue

        if group_by == "model_family":
            df_q["group_id"] = df_q["model_id"].map(family_map).fillna("Other")
        else:
            df_q["group_id"] = df_q["model_id"].astype(str)

        agg_df = _macro_avg_by_question(df_q, ["group_id", "num_objects"])
        if agg_df.empty:
            continue

        num_values = sorted(pd.unique(agg_df["num_objects"]))
        num_to_pos = {value: idx for idx, value in enumerate(num_values)}
        agg_df["num_objects_pos"] = agg_df["num_objects"].map(num_to_pos)

        fig_q, ax_q = plt.subplots(1, 1, figsize=(4, 3.1))
        sns.violinplot(
            data=agg_df,
            x="num_objects_pos",
            y="accuracy",
            ax=ax_q,
            color="0.85",
            inner=None,
            cut=0,
            order=list(range(len(num_values))),
        )
        ax_q.axhline(y=25, color="gray", linestyle="--", linewidth=1)
        if (
            split_values
            and split_values[0] in num_to_pos
            and split_values[1] in num_to_pos
        ):
            left = num_to_pos[split_values[0]]
            right = num_to_pos[split_values[1]]
            line_x = (left + right) / 2
            ax_q.axvline(
                x=line_x,
                color="gray",
                linestyle=":",
                linewidth=1,
                **(split_line_kwargs or {}),
            )

        counts_series = (
            counts_source[counts_source["question_id"] == qid]
            .groupby("num_objects", observed=True)["num_objects"]
            .count()
        )
        if global_limits:
            label_y = global_limits[0] - n_label_offset * max(
                1e-6, global_limits[1] - global_limits[0]
            )
        else:
            label_y = agg_df["accuracy"].min() - n_label_offset * max(
                1e-6, agg_df["accuracy"].max() - agg_df["accuracy"].min()
            )
        for num_obj, count in counts_series.items():
            x_pos = num_to_pos.get(num_obj)
            if x_pos is None:
                continue
            ax_q.text(
                x_pos,
                label_y,
                f"N:{int(count)}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="black",
            )

        for group_id, df_m in agg_df.groupby("group_id"):
            x_vals = df_m["num_objects_pos"].to_numpy()
            y_vals = df_m["accuracy"].to_numpy()
            if x_vals.size == 0:
                continue
            jitter = rng.uniform(-0.2, 0.2, size=x_vals.size)
            x_jittered = x_vals + jitter
            color, marker, size = model_style[group_id]
            ax_q.scatter(
                x_jittered,
                y_vals,
                color=color,
                s=75+ 200 * size,
                alpha=0.8,
                edgecolor="white",
                linewidth=1,
                marker=marker,
                zorder=4,
            )

        label = (
            question_label_map.get(str(qid), str(qid))
            if question_label_map
            else str(qid)
        )
        ax_q.set_title(label)
        ax_q.set_xlabel("Number of Objects")
        ax_q.set_ylabel("Accuracy")
        ax_q.grid(axis="y")
        if y_limit_mode == "fixed":
            ax_q.set_ylim(-0.1, 1.1)
        elif y_limit_mode == "zero_to_max" and global_limits:
            y_min, y_max = global_limits
            pad = y_pad * max(1e-6, y_max - y_min)
            ax_q.set_ylim(
                y_min - n_label_offset * max(1e-6, y_max - y_min), y_max + pad
            )
        else:
            y_min = agg_df["accuracy"].min()
            y_max = agg_df["accuracy"].max()
            if pd.isna(y_min) or pd.isna(y_max):
                ax_q.set_ylim(-0.1, 1.1)
            else:
                pad = y_pad * max(1e-6, y_max - y_min)
                ax_q.set_ylim(y_min - pad, y_max + pad)
        ax_q.set_xticks(range(len(num_values)))
        ax_q.set_xticklabels([str(v) for v in num_values])

        fig_q.tight_layout()
        safe_qid = _safe_filename(str(qid))
        f_out = per_q_dir / f"num_objects_{safe_qid}.png"
        fig_q.savefig(
            f_out, dpi=300, bbox_inches="tight"
        )
        print("Plot saved to:", f_out)
        if show:
            plt.show()
        else:
            plt.close(fig_q)
