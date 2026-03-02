import os
import json
import hashlib
import re
from typing import Any, Callable
from matplotlib.legend import Legend
import numpy as np
import pandas as pd
from prompt_toolkit import prompt
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_hex, to_rgba
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import matplotlib.markers as mmarkers
import utils.utils_mapping
import utils.utils_read

from pathlib import Path

from scipy.stats import pearsonr

try:
    from adjustText import adjust_text
except ImportError:  # adjustText is optional; plotting still works without it.
    adjust_text = None

colors_balanced = ["#E57373", "#F6E6B3", "#8BC87A"]
cmap_balanced = LinearSegmentedColormap.from_list("soft_r2g", colors_balanced)
_SUBCATEGORY_PALETTE = []
for _name in ("Dark2", "tab10"):
    _cmap = plt.get_cmap(_name)
    _SUBCATEGORY_PALETTE.extend(to_hex(_cmap(i)) for i in range(_cmap.N))


def _color_for_subcategory(sub_category: str, palette: list[str]) -> str:
    digest = hashlib.md5(sub_category.encode("utf-8")).hexdigest()
    idx = int(digest, 16) % len(palette)
    return palette[idx]

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
        ax.yaxis.set_minor_locator(mticker.MultipleLocator(ticks_step//2 if isinstance(ticks_step, int) else ticks_step/2))

    ax.grid(False)
    if grid:
        for axis in grid:
            ax.grid(axis=axis, which="major", linestyle="-", alpha=0.5)
            ax.grid(axis=axis, which="minor", linestyle="-", alpha=0.1)
    
    for artist in ax.get_children() + list(ax.get_figure().legends):
        if not artist or not isinstance(artist, Legend):
            continue
        
        leg = artist
        leg.get_title().set_fontsize(12)
        leg.get_title().set_fontweight("bold")
        plt.setp(leg.get_texts(), fontsize=10)      # all entry labels
        leg.get_frame().set_alpha(0.9)              # frame style
        leg.get_frame().set_linewidth(0.8)

def get_level_label(level, name):
    if level == "category":
        return utils.utils_mapping.categories.get(name).replace(" ", "\n")
    elif level == "sub_category":
        return utils.utils_mapping.subcategories.get(name).replace(" ", "\n")
    elif level == "question_id":
        return name

# def get_level_color(level, name):
#     cat_key = get_category(level, name)
#     return utils.utils_mapping.mapping_cat_colors.get(cat_key)

# def get_category(level, name):
#     if level == "category":
#         return name
#     elif level == "sub_category":
#         return utils.utils_mapping.mapping_sub.get(name)
#     elif level == "question_id":
#         return 


def get_benchmark_filepath(dir, cat, bench):
    o_dir = dir if bench == "all" else (dir / f"bench_{bench.lower().replace(' ', '_')}")
    o_dir.mkdir(parents=True, exist_ok=True)
    o_fname = f"cs_{cat}.png" if cat != "all" else "cs_correlation.png"
    return Path(o_dir) / o_fname

def _build_group_legend_items(
    plot_df: pd.DataFrame,
    group_by: str,
    metadata_path: str,
    *,
    legend_clustering: str = None
) -> tuple[list[Line2D], list[str], list[str], str | None]:
    model_style, family_map = utils.utils_mapping._build_model_style(
        # metadata_path,
        group_by=group_by,
        family_marker_mode="distinct",
        metadata_path=metadata_path,
    )

    legend_handles: list[Line2D] = []
    legend_labels: list[str] = []
    legend_groups: list[str] = []

    items = model_style.items()

    groups = set(plot_df[group_by].astype(str).unique())
    for group, (color, marker, size_val, edge) in items:
        if group not in groups:
            continue

        marker_style = mmarkers.MarkerStyle(marker)
        marker_face = color if marker_style.is_filled() else "none"

        if group_by == "model_id":
            label = utils.utils_mapping.model_name_format(group)
        else:
            label = group
        
        models_num = len(plot_df[plot_df[group_by] == group]["model_id"].unique())
        if models_num > 1:
            label = f"{label} (x{models_num})"

        if np.allclose(to_rgba(edge)[:3], (1.0, 1.0, 1.0), atol=1e-3):
            edge = marker_face

        handle = Line2D(
                [0],
                [0],
                marker=marker,
                color="none",
                markerfacecolor=marker_face,
                markeredgecolor=edge,
                markersize=size_val,
                linestyle="None",
            )
        
        legend_handles.append(handle)
        legend_labels.append(label)
        legend_groups.append(group)

    title_str = {"model_id": "Model", "model_family": "Model Family"}.get(group_by)
    
    legend_handles, legend_labels, legend_groups = sort_group_legend_items_posthoc(legend_handles, legend_labels, legend_groups, group_by=group_by, metadata_path=metadata_path)
    # if legend_clustering is not None:
    #     if legend_clustering == "mode":
            
    return legend_handles, legend_labels, legend_groups, title_str


def sort_group_legend_items_posthoc(
    legend_handles: list[Line2D],
    legend_labels: list[str],
    legend_groups: list[str],
    *,
    group_by: str,
    metadata_df: pd.DataFrame | None = None,
    metadata_path: str | Path | None = "utils/metadata.json",
) -> tuple[list[Line2D], list[str], list[str]]:
    metadata_df = utils.utils_read._load_model_metadata(metadata_path)

    def _sort_key(entry: tuple[Line2D, str, str]) -> tuple[str, float, str]:
        _, label, group = entry
        group_str = str(group)

        # If plotting individual models, sort by family first, then by size, then alphabetical model
        if group_by == "model_id":
            current_group = metadata_df[metadata_df[group_by] == group_str]

            family_name = str(current_group["family"].iloc[0])
            params_b = float(current_group["params_b"].iloc[0])
            model_id = str(current_group["model_id"].iloc[0])

            return family_name, params_b, model_id
        
        # If plotting families, sort by family name
        if group_by == "model_family":
            return group_str, None, None

    ordered_entries = sorted(
        zip(legend_handles, legend_labels, legend_groups),
        key=_sort_key,
    )
    
    sorted_handles = [h for h, _, _ in ordered_entries]
    sorted_labels = [l for _, l, _ in ordered_entries]
    sorted_groups = [g for _, _, g in ordered_entries]
    return sorted_handles, sorted_labels, sorted_groups


def create_benchmarks_violin(
    eval_df: pd.DataFrame,
    output_dir: str | Path,
    filename: str,
    benchmarks: list[str],
    *,
    figsize: tuple[float, float] = None,
):
    output_dir = Path(output_dir)
    # all_categories = np.hstack([eval_df["category"].unique(), "all"])
    categories = list(eval_df["category"].unique())

    bench_frames: list[pd.DataFrame] = []
    for bench in benchmarks:
        cat_frames: list[pd.DataFrame] = []
        for cat in categories:
            fpath = get_benchmark_filepath(output_dir, cat, bench).with_suffix(".json")

            df = pd.read_json(fpath)
            df["category"] = cat
            cat_frames.append(df)

        bench_df = pd.concat(cat_frames, ignore_index=True)
        bench_df["benchmark"] = bench
        bench_frames.append(bench_df)

    bench_df = pd.concat(bench_frames, ignore_index=True)

    # Compute pearson per (benchmark, category) group
    def _pearson(g: pd.DataFrame) -> float:
        x, y = g["our_accuracy"], g["cs_accuracy"]
        return pearsonr(x, y)[0]

    bench_cat_pearson_df = (
        bench_df.groupby(["benchmark", "category"], observed=True)[["our_accuracy", "cs_accuracy"]]
        .apply(_pearson)
        .reset_index(name="pearson")
    )

    # Add category positions for plotting
    categories_sorted = utils.utils_mapping.sort_categories(
        bench_cat_pearson_df["category"].unique()
    )
    categories_sorted = list(categories_sorted)
    cat_to_pos = {cat: idx for idx, cat in enumerate(categories_sorted)}
    bench_cat_pearson_df["cat_pos"] = bench_cat_pearson_df["category"].map(cat_to_pos)

    # Do the violin
    cats_mask = bench_cat_pearson_df["category"].ne("all")
    fig, ax = plt.subplots(figsize=(2.5+len(categories_sorted), 4))
    sns.violinplot(
        data=bench_cat_pearson_df[cats_mask],  # Exclude the "all" category from the violin plot
        x="cat_pos",
        y="pearson",
        ax=ax,
        color="0.90",
        inner=None,
        cut=0,
        width=1.0,
        linewidth=0.5,
        order=list(range(len(categories_sorted))),
    )

    ax.set_xticks(range(len(categories_sorted)))
    ax.set_xticklabels(categories_sorted)
    ax.set_xlabel("")
    ax.set_ylabel("Pearson correlation")
    
    rng = np.random.default_rng(40)
    palette = sns.color_palette("tab10", n_colors=bench_cat_pearson_df[cats_mask]["benchmark"].nunique())
    for g_idx, (group_name, df_m) in enumerate(bench_cat_pearson_df[cats_mask].groupby("benchmark", observed=True)):
        x_vals = df_m["cat_pos"].to_numpy()
        y_vals = df_m["pearson"].to_numpy()
        
        x_jittered = x_vals + rng.uniform(-0.15, 0.15, size=x_vals.size)
        
        if group_name == "all":
            color, marker, size = "#333333", "o", 12**2
        else:
            color, marker, size = palette[g_idx % len(palette)], "*", 24**2

        # Plot scatter points
        ax.scatter(
            x_jittered,
            y_vals,
            color=color,
            s=size,
            alpha=0.8,
            edgecolor="white",
            linewidth=1,
            marker=marker,
            zorder=4,
            label=group_name if group_name != "all" else "All benchmarks",
        )
    
    # Category labels
    category_labels = [ get_level_label("category", cat) if cat != "all" else "All benchmarks"
                    #    + f"\n({bench_cat_pearson_df[bench_cat_pearson_df['category'] == 'all']['pearson'].values[0]:.2f})" 
                       for cat in categories_sorted]
    ax.set_xticks(range(len(category_labels)))
    ax.set_xticklabels(category_labels, ha='center')
    
    for ticklabel, cat in zip(ax.get_xticklabels(), categories_sorted):
        if cat != "all":
            ticklabel.set_color(utils.utils_mapping.mapping_cat_colors.get(cat)+"CC")  # Adding transparency

    ax.legend(
        title="Benchmarks",
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
    )
    
    paperformat(ax, figsize=None, grid=["y"], minor=False, ticks_step=0.2)
    for ticklabel in ax.get_xticklabels():
        ticklabel.set_fontsize(ticklabel.get_fontsize()*0.85)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{output_dir}/{filename}", dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_dir}/{filename}")


def make_balanced_matrix(
    eval_df: pd.DataFrame,
    by="model_id",
    hierarchy=("category", "sub_category", "question_id"),
    index_to_use="sub_category",
):
    """
    Build a balanced accuracy matrix for plotting.

    Balanced definition:
      per question: mean(is_correct)
      per sub_category: mean of per-question accuracies
      per category: mean of per-sub_category accuracies
      overall per model: mean of per-category accuracies

    Parameters
    ----------
    eval_base : DataFrame with columns [by, category, sub_category, question_id, is_correct, ...]
    by        : key to pivot to columns, typically 'model_id'
    hierarchy : order of the taxonomy (cat, sub, question)
    index_to_use : row index for the heatmap; can be
                   'question_id', 'sub_category', 'category', or any other column,
                   or a tuple like ('category','sub_category') to keep rows unique

    Returns
    -------
    acc : DataFrame pivot with rows = index_to_use, cols = by, values = balanced accuracy
          plus an extra "Total" row that is the fully balanced overall per model
    breakdown : dict with q_acc, sub_acc, cat_acc, overall for optional debugging
    """
    by_cols = [by] if isinstance(by, str) else list(by)
    cat_col, sub_col, q_col = hierarchy

    level_df = utils.utils_read.macro_accuracy(eval_df, level=index_to_use, group_by=by_cols)

    # Choose which level to display on the heatmap rows
    idx_cols = index_to_use

    # Pivot into a matrix
    cols = by_cols[0] if len(by_cols) == 1 else by_cols
    mat = level_df.pivot(index=idx_cols, columns=cols, values="accuracy").sort_index()

    # Append a fully balanced Total row per model
    # okay the total is always aggregated by overall category accuracy
    overall_df = utils.utils_read.macro_accuracy(eval_df, level="model_id")
    tot = overall_df.set_index(by_cols)["accuracy"].to_frame().T
    tot.index = ["Total"]
    acc = pd.concat([mat, tot], axis=0)

    return acc


def create_graph_from_eval_balanced(
    eval_base: pd.DataFrame,
    index_to_use="sub_category",
    name_graph="heatmap_balanced",
    filename="matrix.png",
    color_by_mode=True,
    orientation="landscape",
    by="model_id",
    figsize=None,
    show=True,
    include_counts=False,
    color_question_id_by_subcategory=False,
    subcategory_palette=None,
    out_dir=None,
):
    """
    Plot a heatmap where every cell is a balanced accuracy derived from eval_base.

    The "Total" row is the fully balanced overall per model.
    The "Average" first column is the simple mean across models for each row,
    added for quick visual comparison.
    """
    # Convert accuracy to percentage
    # eval_base["accuracy"] = eval_base["accuracy"] * 100

    # Build the balanced matrix for the requested row index
    acc = make_balanced_matrix(
        eval_df=eval_base,
        by=by,
        index_to_use=index_to_use,
    )

    # Optional transpose to put models on rows
    if orientation != "landscape":
        acc = acc.T
        x_label, y_label = (
            (index_to_use, by)
            if isinstance(index_to_use, str)
            else (" × ".join(index_to_use), by)
        )
    else:
        x_label, y_label = (
            (by, index_to_use)
            if isinstance(index_to_use, str)
            else (by, " × ".join(index_to_use))
        )

    if (
        include_counts
        and isinstance(index_to_use, str)
        and index_to_use in eval_base.columns
    ):
        counts = eval_base.groupby(index_to_use, observed=True, dropna=False)[
            "idx"
        ].nunique()
        if orientation != "landscape":
            acc = acc.rename(
                columns={
                    k: f"{k} (n={counts.get(k, 0)})" for k in acc.columns if k in counts
                }
            )
        else:
            acc = acc.rename(
                index={
                    k: f"{k} (n={counts.get(k, 0)})" for k in acc.index if k in counts
                }
            )

    # Add an "Average" column across models for each row
    avg_col = acc.iloc[:-1].mean(
        axis=1
    )  # ignore the Total row when computing row means
    acc.insert(0, "Average", avg_col)
    acc.loc["Total", "Average"] = acc.loc["Total", acc.columns[1:]].mean()

    acc = acc.apply(pd.to_numeric, errors="coerce")

    # Labels
    labels = (acc * 100).round(2).astype("Float64").astype(str) + "%"

    # Plot
    if figsize is None:
        figsize = (
            max(28, 1.5 * acc.shape[1] + 4),
            max(6, 0.6 * acc.shape[0] + 2),
        )
    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        acc,
        vmin=0,
        vmax=1,
        cmap=cmap_balanced,
        annot=labels,
        fmt="",
        linewidths=0.5,
        linecolor="white",
        cbar=False,
        cbar_kws={"format": PercentFormatter(xmax=1)},
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"Accuracy by {index_to_use} and model", fontsize=32)
    plt.yticks(rotation=0)

    # Optional tick coloring by model mode
    # if color_by_mode and "mode_y" in eval_base.columns:
    #     model_mode_map = (
    #         eval_base[[by, "mode_y"]]
    #         .drop_duplicates(subset=[by])
    #         .set_index(by)["mode_y"]
    #     )
    #     mode_colors = {"image-only": "#208A00", "general": "#001C82"}
    #     ticklabels = (
    #         ax.get_xticklabels() if orientation == "landscape" else ax.get_yticklabels()
    #     )
    #     for label in ticklabels:
    #         model = label.get_text()
    #         if model in model_mode_map.index:
    #             label.set_color(mode_colors.get(model_mode_map[model], "black"))
    #     handles = [plt.Line2D([0], [0], color=c, lw=4) for c in mode_colors.values()]
    #     ax.legend(
    #         handles,
    #         list(mode_colors.keys()),
    #         title="Mode",
    #         loc="upper left",
    #         bbox_to_anchor=(1.02, 1),
    #     )

    # if color_question_id_by_subcategory and index_to_use == "question_id":
    #     if "sub_category" in eval_base.columns:
    #         q_to_sub = (
    #             eval_base[["question_id", "sub_category"]]
    #             .drop_duplicates()
    #             .set_index("question_id")["sub_category"]
    #             .to_dict()
    #         )
    #         palette = subcategory_palette or _SUBCATEGORY_PALETTE
    #         ticklabels = (
    #             ax.get_yticklabels()
    #             if orientation == "landscape"
    #             else ax.get_xticklabels()
    #         )
    #         for label in ticklabels:
    #             raw = label.get_text()
    #             qid = raw.split(" (n=")[0]
    #             sub = q_to_sub.get(qid)
    #             if sub:
    #                 label.set_color(_color_for_subcategory(str(sub), palette))

    # Highlight the first column and last row
    num_rows, num_columns = acc.shape
    rect_column = Rectangle(
        (0, 0), 1, num_rows, fill=False, edgecolor="white", linewidth=4
    )
    rect_rows = Rectangle(
        (0, num_rows - 1), num_columns, 1, fill=False, edgecolor="white", linewidth=4
    )
    ax.add_patch(rect_rows)
    ax.add_patch(rect_column)

    # Save
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f"{out_dir}/{filename}", dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {out_dir}/{filename}")

    return acc


def create_sub_categories_summary(
    acc_mat,
    title="Sub-category accuracy summary",
    show=True,
):
    raise NotImplementedError("This should be updated to use the new macro accuracies")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    # --- 1. PREPARE DATA ---
    # (Assuming acc_mat is already loaded in your environment)
    df = acc_mat.iloc[:-1, 1:].T.reset_index()  # Uncomment if you have the raw acc_mat
    df = df.rename(columns={"index": "model_id"})

    # Set index for easier filtering
    df_indexed = df.set_index("model_id")

    # --- 2. SEPARATE & SORT ROWS (The Key Logic) ---
    # We determine if a model is "Multi-frame" if it has valid data (not NaN)
    # in the 'camera_motion' column (or any other primary multi-frame column).
    is_multi_frame = df_indexed["camera_motion"].notna()

    # Split the dataframe
    df_multi = df_indexed.loc[is_multi_frame]
    df_single = df_indexed.loc[~is_multi_frame]

    # Optional: Sort internally within groups (e.g., by fewest NaNs)
    df_multi = df_multi.loc[
        df_multi.isna().sum(axis=1).sort_values(ascending=False).index
    ]
    df_single = df_single.loc[
        df_single.isna().sum(axis=1).sort_values(ascending=False).index
    ]

    # Recombine: Multi on TOP, Single on BOTTOM
    mat = pd.concat([df_single, df_multi])

    # Reorder columns by missingness (most missing first) for visibility
    mat = mat.reindex(columns=mat.isna().sum().sort_values(ascending=False).index)

    # Create mask for NaNs
    mask = mat.isna()

    # --- 3. PLOTTING ---
    plt.figure(figsize=(12, 12))  # Increased height slightly

    # Custom cmap (placeholder if you don't have cmap_balanced defined)
    # cmap_balanced = sns.diverging_palette(20, 220, as_cmap=True)

    ax = sns.heatmap(
        mat,
        annot=True,
        fmt=".2f",
        cmap=cmap_balanced,  # specific cmap or use your 'cmap_balanced'
        mask=mask,
        cbar_kws={"label": "Score"},
    )

    # --- 4. FILL BLANKS (N/A) ---
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mask.iat[i, j]:
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    "N/A",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8,
                )
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color="#F4EBEB", zorder=2))

    # --- 5. VISUAL SEPARATION (New Feature) ---

    # A. Add a Horizontal Line between the groups
    # The line position is exactly the number of rows in the first group
    split_pos = len(df_single)
    ax.axhline(
        y=split_pos, color="white", linewidth=6
    )  # Thick white line for gap effect
    ax.axhline(
        y=split_pos, color="black", linewidth=1, linestyle="--"
    )  # Thin dashed line for logic

    # B. Add Text Annotation for the groups
    ax.text(
        -7,
        split_pos / 2,
        "Single-Frame\nModels",
        fontsize=12,
        rotation=90,
        va="center",
        ha="center",
        color="#001C82",
        fontweight="bold",
    )
    ax.text(
        -7,
        split_pos + (len(df_single) / 2),
        "Multi-Frame\nModels",
        fontsize=12,
        rotation=90,
        va="center",
        ha="center",
        color="#016405",
        fontweight="bold",
    )

    # --- 6. COLOR LABELS ---

    # Column Colors mapping (main categories palette)
    main_colors = colors_balanced
    col_colors = {
        # Spatial
        "layout": main_colors[2],
        "distance": main_colors[2],
        "size": main_colors[2],
        "camera_characteristics": main_colors[2],
        # Temporal
        "camera_motion": main_colors[1],
        "event_ordering": main_colors[1],
        "persistence": main_colors[1],
        # Physical
        "kinematics": main_colors[0],
        "collision": main_colors[0],
        "material_identification": main_colors[0],
        "physics_property": main_colors[0],
        "mass": main_colors[0],
        "visibility": main_colors[0],
    }

    # Apply Column Colors (X-axis) - SAFER METHOD
    # Instead of zip(), we look up the color by the label text
    for tick_label in ax.get_xticklabels():
        lbl_text = tick_label.get_text()
        if lbl_text in col_colors:
            tick_label.set_color(col_colors[lbl_text])

    # Apply Row Colors (Y-axis) - NEW FEATURE
    # Identify names of multi-frame models
    multi_model_names = set(df_multi.index)

    for tick_label in ax.get_yticklabels():
        model_name = tick_label.get_text()
        if model_name in multi_model_names:
            tick_label.set_color("#016405")  # Light Green for Multi-frame
        else:
            tick_label.set_color("#001C82")  # Dark Blue for Multi-frame

    # Final Polish
    ax.set_ylabel("")  # Remove default label to clean up
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    run = globals().get("RUN_NAME", "default")
    os.makedirs(f"./output/{run}/", exist_ok=True)
    plt.savefig(f"./output/{run}/{title}.png", dpi=300, bbox_inches="tight")
    print(f"Plot saved to: ./output/{run}/{title}.png")


def create_correlation_common_sense(
    eval_df: pd.DataFrame,
    acc_mat: pd.DataFrame,
    title: str = "Correlation Matrix Sorted",
    show: bool = True,
):
    raise NotImplementedError("This should be updated to use the new macro accuracies")
    df = acc_mat.iloc[:-1, 1:].T.reset_index()  # Uncomment if you have the raw acc_mat
    df = df.rename(columns={"index": "model_id"})

    # Define column groups
    multi_frame_cols = ["camera_motion", "event_ordering", "kinematics"]
    # Identify all columns
    all_cols = df.columns.tolist()
    # Filter out model_id and multi_frame columns to find single_frame columns
    single_frame_cols = [
        c for c in all_cols if c not in multi_frame_cols and c != "model_id"
    ]

    # Set index for easier filtering
    df_indexed = df.set_index("model_id")

    # --- 2. SEPARATE & SORT ROWS (The Key Logic) ---
    # We determine if a model is "Multi-frame" if it has valid data (not NaN)
    # in the 'camera_motion' column (or any other primary multi-frame column).
    is_multi_frame = df_indexed["camera_motion"].notna()

    # Split the dataframe
    df_multi = df_indexed.loc[is_multi_frame]
    df_single = df_indexed.loc[~is_multi_frame]

    # Optional: Sort internally within groups (e.g., by fewest NaNs)
    df_multi = df_multi.loc[
        df_multi.isna().sum(axis=1).sort_values(ascending=False).index
    ]
    df_single = df_single.loc[
        df_single.isna().sum(axis=1).sort_values(ascending=False).index
    ]

    # Recombine: Multi on TOP, Single on BOTTOM
    mat = pd.concat([df_single, df_multi])

    # Reorder columns: Multi-frame cols first, then the rest
    mat = mat.reindex(columns=multi_frame_cols + single_frame_cols)

    with open("./utils/common_sense.json", "r") as file:
        common_sense_df = pd.DataFrame(json.load(file)).drop(columns=["OCRBench"])
    common_sense_df.drop(
        columns=[
            "Rank",
            "Eval Time",
            "Language Model",
            "Vision Model",
            "Avg. Score",
            "OCRBench",
        ],
        inplace=True,
        errors="ignore",
    )

    model_unique_ids = eval_df["model_id"].unique()

    matches = []

    for model_id in model_unique_ids:
        modified = model_id.replace("2_5", "2.5") if "2_5" in model_id else model_id

        mask = common_sense_df["Method"].str.contains(modified, case=False, regex=False)

        hits = common_sense_df[mask].copy()
        if hits.empty:
            continue

        hits["matched_model"] = model_id
        matches.append(hits)

    final_df = pd.concat(matches, ignore_index=True)

    norm_mat_df = mat.apply(lambda col: col * 100 if col.max() < 1.0 else col)

    joined_df = (
        final_df.merge(
            norm_mat_df.reset_index(),
            left_on="matched_model",
            right_on="model_id",
            how="left",
        )
        .drop(columns=["Method", "Params", "matched_model"])
        .set_index("model_id")
        .reset_index()
    )

    joined_df_index = joined_df.set_index("model_id")
    # print(joined_df_index)

    corr = joined_df_index.corr(method="pearson")

    # 1. Identify the column you want to be the "Standard" for sorting (e.g., 'Overall Score')
    # If you don't have one, we can sort by the sum of correlations to see generally 'connected' features first
    target_col = corr.columns[0]  # Or replace with specific name like 'accuracy'

    # 2. Sort the correlation matrix based on that column
    # ascending=False puts the high positive correlations (Greens) first
    sorted_index = corr.sort_values(by=target_col, ascending=False).index

    # 3. Re-index the correlation matrix with this new order
    corr_sorted = corr.reindex(index=sorted_index, columns=sorted_index)

    # 4. Plot the sorted heatmap
    plt.figure(figsize=(15, 12))
    sns.heatmap(corr_sorted, annot=True, cmap=cmap_balanced)
    plt.title(f"{title} by {target_col}")
    run = globals().get("RUN_NAME", "default")
    os.makedirs(f"./output/{run}/", exist_ok=True)
    plt.savefig(f"./output/{run}/{title}.png", dpi=300, bbox_inches="tight")
    print(f"Plot saved to: ./output/{run}/{title}.png")

    return corr_sorted


def create_accuracy_bench_vs_common_sense(
        eval_df: pd.DataFrame, 
        out_filename: str = "accuracy_vs_common_sense.png",
        show_legend: bool = True,
        family_marker_mode: str = "distinct",
        group_by: str = "model_family",
        ylabel: str = "Accuracy (%)",
        label_fontsize = 12,
        tick_fontsize = 12,
        legend_fontsize = 10,
        ylim = None,
        show_xlabel: bool = True,
        figsize: tuple = (4, 2.5),
        out_dir: str = None,
        benchmark: str = "all",
    ):
    def _standardize_model_label(model_id: str) -> str:
        label = model_id
        label = label.replace("2_5", "2.5")
        label = label.replace("V1-5-", "V1.5-")
        label = label.replace("InternVL2-76B", "InternVL2-Llama3-76B")
        label = label.replace("-quantable", "")
        label = label.replace("MiniCPM-V2.5", "MiniCPM-Llama3-V2.5")
        label = label.replace("MolmoE-7B-", "Molmo-7B-")
        label = label.replace("Phi-3-vision-128k-instruct", "Phi-3-Vision")
        label = label.replace("instructblip-vicuna-7b", "InstructBLIP-7B")
        label = label.replace("llava-interleave-qwen", "LLaVA-Next-Interleave")
        label = label.replace("llava-v1.6-", "LLaVA-Next-")
        label = label.replace("vila-1.5-8b", "Llama-3-VILA1.5-8B")
        label = label.removesuffix("-hf")
        label = label.replace("_", "-")
        label = re.sub(
            r"(\d+(?:\.\d+)?)b\b",
            lambda m: f"{m.group(1)}B",
            label,
            flags=re.IGNORECASE,
        )
        return label

    with open("./utils/common_sense.json", "r") as file:
        common_sense_df = pd.DataFrame(json.load(file))
    with open("./utils/metadata.json", "r") as file:
        metadata_models = pd.DataFrame(json.load(file))

    common_sense_df.head()

    model_unique_ids = eval_df["model_id"].unique()
    model_df = utils.utils_read.macro_accuracy(eval_df, level="model_id")

    benchmark_field = "Avg. Score" if benchmark == "all" else benchmark
    
    model_cs_mapping = {}
    cs_accuracy = {}
    cs_methods = common_sense_df["Method"].values
    for model_id in model_unique_ids:
        model_row = model_df[model_df["model_id"] == model_id].iloc[0]

        #Prioritize open, then close, then any match
        cs_mask = None
        for suffix in ["Open", "Close", ""]:
            cs_mask = np.array([(_standardize_model_label(model_id)+suffix).lower() in cs_model_id.lower() for cs_model_id in cs_methods])
            if sum(cs_mask) > 0:
                break
        
        if sum(cs_mask) != 0:
            assert sum(cs_mask) == 1, Exception(f"Multiple common sense matches found for {model_id}. Matches: {common_sense_df[cs_mask]['Method'].values}")
            
            cs_model_id = common_sense_df[cs_mask]["Method"].values[0]
            cs_row = common_sense_df[common_sense_df["Method"] == cs_model_id].iloc[0]
            # print(f"Mapping `{model_id}` (ours) with `{cs_model_id}` (common sense)")
            cs_accuracy[model_id] = float(cs_row[benchmark_field])
            if benchmark_field == "OCRBench":
                cs_accuracy[model_id] *= 0.1  # OCRBench is in [0, 1000], we want it in [0, 100]
        else:
            cs_model_id = None
        
        model_cs_mapping[model_id] = cs_model_id
    
    cs_found = [model for model, cs_model in model_cs_mapping.items() if cs_model is not None]
    cs_not_found = [model for model, cs_model in model_cs_mapping.items() if cs_model is None]
    print(f"CS Model mapping found ({len(cs_found)}/{len(model_cs_mapping)}): ", cs_found)
    print(f"CS Model mapping NOT found ({len(cs_not_found)}/{len(model_cs_mapping)}): ", cs_not_found)
    
    # Save mapping for reference
    pd.Series(model_cs_mapping, name="cs_model") \
    .rename_axis("model_id") \
    .reset_index() \
    .to_json( f"{out_dir}/model_cs_mapping.json", orient="records", indent=2, force_ascii=False)

    # Retrieve styles
    model_style, family_map = utils.utils_mapping._build_model_style(
        "./utils/metadata.json",
        group_by=group_by,
        family_marker_mode=family_marker_mode,
    )
    

    # accuracy_total_per_model = acc_mat.iloc[-1:, :]
    eval_df_accuracy_total_per_model = (
        eval_df.merge(
            model_df.reset_index().rename(columns={"accuracy": "Total"}),
            left_on="model_id",
            right_on="model_id",
            how="left",
        )
        .groupby("model_id")
        .first()
        .reset_index()
    )

    if "idx" in eval_df.columns and "model_id" in eval_df.columns:
        multi_models = set(
            eval_df[
                eval_df["idx"].astype(str).str.contains("_g")
                & eval_df["model_answer"].notna()
            ]["model_id"].unique()
        )
        eval_df_accuracy_total_per_model["mode_y"] = eval_df_accuracy_total_per_model[
            "model_id"
        ].apply(lambda m: "general" if m in multi_models else "image-only")

    base_cols = ["model_id", "Total", "mode_y"]
    if "params_b" in eval_df_accuracy_total_per_model.columns:
        base_cols.append("params_b")
    
    cs_df = pd.concat([
            pd.Series({m: cs_m for m, cs_m in model_cs_mapping.items () if m is not None}, name="cs_model"),
            pd.Series(cs_accuracy, name="cs_accuracy"),
        ],
        axis=1,
    ).rename_axis("model_id").reset_index()
    eval_df_accuracy_total_per_model = (
        eval_df_accuracy_total_per_model[base_cols]
        .rename(columns={"Total": "our_accuracy", "mode_y": "mode"})
        .merge(cs_df, on="model_id", how="left")
        .merge(
            metadata_models[["id", "params_b"]].rename(columns={"id": "model_id"}),
            on="model_id",
            how="left",
        )
        .dropna(subset=["cs_accuracy"])
    )

    if "params_b" not in eval_df_accuracy_total_per_model.columns:
        eval_df_accuracy_total_per_model["params_b"] = pd.NA
    eval_df_accuracy_total_per_model["cs_accuracy"] = pd.to_numeric(
        eval_df_accuracy_total_per_model["cs_accuracy"], errors="coerce"
    )

    plt.figure(figsize=figsize)
    sns.set_style("white")

    # Ensure numeric columns
    num_cols = ["cs_accuracy", "our_accuracy"]
    if "params_b" in eval_df_accuracy_total_per_model.columns:
        num_cols.append("params_b")
    for c in num_cols:
        eval_df_accuracy_total_per_model[c] = pd.to_numeric(
            eval_df_accuracy_total_per_model[c], errors="coerce"
        ).astype(float)

    balanced_max = eval_df_accuracy_total_per_model["our_accuracy"].max()
    common_max = eval_df_accuracy_total_per_model["cs_accuracy"].max()
    scale_balanced = pd.notna(balanced_max) and balanced_max <= 1.0 and common_max > 1.5
    scale_common = pd.notna(common_max) and common_max <= 1.0 and balanced_max > 1.5
    if scale_balanced:
        eval_df_accuracy_total_per_model["our_accuracy"] *= 100.0
    if scale_common:
        eval_df_accuracy_total_per_model["cs_accuracy"] *= 100.0

    # 1. Calculate Pearson Correlation
    # Drop NaNs to ensure accurate calculation
    assert eval_df_accuracy_total_per_model.isna().sum().sum() == 0, "NaN values found in correlation dataframe. Please check the data."
    corr_df = eval_df_accuracy_total_per_model.dropna(
        subset=["cs_accuracy", "our_accuracy"]
    )
    corr_df.to_json(f"{out_dir}/{Path(out_filename).stem}.json", orient="records", indent=2, force_ascii=False)

    r_val, p_val = pearsonr(
        corr_df["cs_accuracy"], corr_df["our_accuracy"]
    )
    
    # 2. Regression Plot
    sns.regplot(
        data=eval_df_accuracy_total_per_model,
        x="cs_accuracy",
        y="our_accuracy",
        scatter=False,
        ci=95,
        line_kws={"color": "red", "lw": 1.5, "ls": "--"},
    )

    # 3. Scatter Plot
    for model_id, df_m in eval_df_accuracy_total_per_model.groupby("model_id"):
        color, marker, size, edge = model_style[model_id]
        scatter_kwargs = dict(
            data=df_m,
            x="cs_accuracy",
            y="our_accuracy",
            # hue="mode",
            # marker="o",
            color=color,
            s=size**2,
            marker=marker,
            edgecolor=edge,
            alpha=0.9,
            legend=False,
        )

        ax = sns.scatterplot(**scatter_kwargs)

    if show_xlabel:
        ax.set_xlabel("Common Sense", fontsize=label_fontsize, fontweight="bold")
    else:
        ax.set_xlabel("", fontsize=label_fontsize, fontweight="bold")
    # ax.set_ylabel("Accuracy", fontsize=label_fontsize, fontweight="bold")
    # if scale_common:
    #     ax.set_xlabel("Common Sense (%)", fontsize=label_fontsize, fontweight="bold")
    # if scale_balanced:
    ylabel_color = "black"
    if eval_df["category"].nunique() == 1:
        ylabel_color = utils.utils_mapping.mapping_cat_colors.get(eval_df["category"].unique()[0])+"CC"

    ax.set_ylabel(ylabel, fontsize=label_fontsize, fontweight="bold", color=ylabel_color)
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    # ax.grid(False)
    
    import matplotlib.ticker as mticker
    # yticks = list(range(int(ax.get_ylim()[0])//5*5, int(ax.get_ylim()[1])//5*5 + 5, 5))  # [10, 20, ..., 100]
    # ax.set_yticks(yticks)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(5))
    # ax.grid(axis="y", which="major", alpha=0.3)
    ax.grid(axis="y", alpha=0.3)

    # 4. Display Pearson Correlation on the plot
    # transform=ax.transAxes uses relative coordinates (0,0 is bottom-left, 1,1 is top-right)
    cmap = plt.get_cmap("RdYlGn")
    ax.text(
        0.02,
        0.04,
        f"r = {r_val:.2f}",
        transform=ax.transAxes,
        fontsize=legend_fontsize,
        fontweight="bold",
        verticalalignment="bottom",
        horizontalalignment="left",
        bbox=dict(boxstyle="round,pad=0.3", fc=cmap(Normalize(0.2, 0.8, clip=True)(r_val)), ec="gray", alpha=0.8),
    )

    # 5. Modified Legend
    # legend_handles, legend_labels, legend_groups, title_str = _build_group_legend_items(
    #     plot_df,
    #     group_by=group_by,
    #     metadata_path=metadata_path
    # )

    # 6. Annotations
    annotate_df = eval_df_accuracy_total_per_model.dropna(
        subset=["cs_accuracy", "our_accuracy"]
    ).sort_values("our_accuracy", ascending=False)
    annotate_df["model_label"] = annotate_df["model_id"].map(_standardize_model_label)
    label_texts = []
    for _, r in annotate_df.iterrows():
        label = r["model_label"]
        versionname = metadata_models.loc[metadata_models["id"] == r["model_id"], "versionname"].values[0]
        # print(r["model_id"])
        offset = (0, 10)
        # ha = "left" if r["cs_accuracy"] < 45 else "right"
        if r["cs_accuracy"] < 45:
            offset = (offset[0]+10*(45-r["cs_accuracy"])/(45-35), offset[1])
        ha = "center"
        # if label in {"InternVL2.5-4B", "InternVL2.5-2B"}:
        #     offset = (-10, 0)
        #     ha = "right"
        label_texts.append(
            ax.annotate(
                versionname,
                xy=(r["cs_accuracy"], r["our_accuracy"]),
                xytext=offset,
                textcoords="offset points",
                va="center",
                ha=ha,
                fontsize=8,
            )
        )

    if adjust_text is not None and label_texts:
        adjust_text(
            label_texts,
            ax=ax,
            expand_points=(1.2, 1.2),
            expand_text=(1.1, 1.2),
            arrowprops=dict(arrowstyle="-", color="0.5", lw=0.5),
        )

    if "params_b_plot" in eval_df_accuracy_total_per_model.columns:
        assert False, "Outdated need to refactorize with the new plot code,(style, etc.)"
        params = eval_df_accuracy_total_per_model["params_b_plot"].dropna().astype(float)
        positive = params[params > 0]
        if not positive.empty:
            min_pos = float(positive.min())
            max_pos = float(positive.max())
            size_min, size_max = 40, 900
            size_norm = LogNorm(vmin=min_pos, vmax=max_pos)

            def _size_for_param(val: float) -> float:
                frac = float(size_norm(val))
                return size_min + (size_max - size_min) * frac

            size_refs = [1, 7, 20]
            size_refs = [suffix for suffix in size_refs if min_pos <= suffix <= max_pos]
            if not size_refs:
                size_refs = [
                    round(min_pos, 2),
                    round((min_pos + max_pos) / 2, 2),
                    round(max_pos, 2),
                ]
            size_handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    color="gray",
                    markerfacecolor="none",
                    markersize=_size_for_param(suffix) ** 0.5,
                    label=f"{suffix}B",
                )
                for suffix in size_refs
            ]
            if show_legend:
                legend_sizes = ax.legend(
                    handles=size_handles,
                    title="Size (params)",
                    loc="lower right",
                    fontsize=legend_fontsize,
                    title_fontsize=legend_fontsize,
                    frameon=True,
                )
                ax.add_artist(legend_mode)
                ax.add_artist(legend_sizes)

    sns.despine(ax=ax)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(
        f"{out_dir}/{out_filename}",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.0,
    )
    print(f"Plot saved to: {out_dir}/{out_filename}")
