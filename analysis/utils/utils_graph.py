import os
import json
import hashlib
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_hex
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import utils.utils_mapping

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


def make_balanced_matrix(
    eval_base: pd.DataFrame,
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

    # 1) per question accuracy
    q_acc = (
        eval_base.groupby(
            by_cols + [cat_col, sub_col, q_col], observed=True, dropna=False
        )["is_correct"]
        .mean()
        .reset_index(name="acc_q")
    )

    # 2) per sub_category (equal weight to questions)
    sub_acc = (
        q_acc.groupby(by_cols + [cat_col, sub_col], observed=True, dropna=False)[
            "acc_q"
        ]
        .mean()
        .reset_index(name="acc_sub")
    )

    # 3) per category (equal weight to sub_categories)
    cat_acc = (
        sub_acc.groupby(by_cols + [cat_col], observed=True, dropna=False)["acc_sub"]
        .mean()
        .reset_index(name="acc_cat")
    )

    # 4) overall per model (equal weight to categories)
    overall = (
        cat_acc.groupby(by_cols, observed=True, dropna=False)["acc_cat"]
        .mean()
        .reset_index(name="balanced_overall")
    )

    # Choose which level to display on the heatmap rows
    if index_to_use == q_col:
        base = q_acc.rename(columns={"acc_q": "value"})
        idx_cols = [q_col]
    elif index_to_use == sub_col:
        base = sub_acc.rename(columns={"acc_sub": "value"})
        idx_cols = [sub_col]
    elif index_to_use == cat_col:
        base = cat_acc.rename(columns={"acc_cat": "value"})
        idx_cols = [cat_col]
    else:
        # Generic dimension: take equal mean of per‑question accuracy inside that dimension
        idx_cols = (
            list(index_to_use)
            if isinstance(index_to_use, (list, tuple))
            else [index_to_use]
        )
        base = (
            q_acc.groupby(by_cols + idx_cols, observed=True, dropna=False)["acc_q"]
            .mean()
            .reset_index(name="value")
        )

    # Pivot into a matrix
    cols = by_cols[0] if len(by_cols) == 1 else by_cols
    mat = base.pivot(index=idx_cols, columns=cols, values="value").sort_index()

    # Append a fully balanced Total row per model
    # okay the total is always aggregated by overall category accuracy
    tot = overall.set_index(by_cols)["balanced_overall"].to_frame().T
    tot.index = ["Total"]
    acc = pd.concat([mat, tot], axis=0)

    return acc, {
        "q_acc": q_acc,
        "sub_acc": sub_acc,
        "cat_acc": cat_acc,
        "overall": overall,
    }


def create_graph_from_eval_balanced(
    eval_base: pd.DataFrame,
    index_to_use="sub_category",
    name_graph="heatmap_balanced",
    title=None,
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
    if out_dir is None:
        run = globals().get("RUN_NAME", "default")
        out_dir = f"./output/{run}/"

    # Build the balanced matrix for the requested row index
    acc, breakdown = make_balanced_matrix(
        eval_base=eval_base,
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
    ax.set_title(title or f"Balanced accuracy by {index_to_use} and model", fontsize=32)
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
    plt.savefig(f"{out_dir}/{title}.png", dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {out_dir}/{title}.png")

    return acc, breakdown


def create_sub_categories_summary(
    acc_mat,
    title="Sub-category accuracy summary",
    show=True,
):
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
        acc_mat: pd.DataFrame, 
        out_filename: str = "accuracy_vs_common_sense.png",
        show_legend: bool = True,
        family_marker_mode: str = "distinct",
        # family_marker_base: str = "^",
        group_by: str = "model_family",
        ylabel: str = "Accuracy (%)",
        label_fontsize = 12,
        tick_fontsize = 12,
        legend_fontsize = 10,
        show_xlabel: bool = True,
        figsize: tuple = (4, 2.5),
        out_dir: str = None,
    ):
    if out_dir is None:
        run = globals().get("RUN_NAME", "default")
        out_dir = f"./output/{run}/"
    
    def _standardize_model_label(model_id: str) -> str:
        label = model_id.replace("2_5", "2.5")
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

    common_sense_accuracy = {}

    for model_id in model_unique_ids:
        for cs_model_id in common_sense_df["Method"].values:
            if _standardize_model_label(model_id) in cs_model_id:
                # print(f"Model: {modified_model_id}, Common Sense Score: {cs_model_id}")
                row = common_sense_df[common_sense_df["Method"] == cs_model_id].iloc[0]
                common_sense_accuracy[model_id] = row.get("Avg. Score")

    model_style, family_map = utils.utils_mapping._build_model_style(
        "./utils/metadata.json",
        group_by=group_by,
        family_marker_mode=family_marker_mode,
    )
    
    accuracy_total_per_model = acc_mat.iloc[-1:, :]
    eval_df_accuracy_total_per_model = (
        eval_df.merge(
            accuracy_total_per_model.T.reset_index().rename(
                columns={0: "balanced_accuracy"}
            ),
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
    eval_df_accuracy_total_per_model = (
        eval_df_accuracy_total_per_model[base_cols]
        .rename(columns={"Total": "balanced_accuracy", "mode_y": "mode"})
        .merge(
            pd.DataFrame.from_dict(
                {
                    "common_sense_accuracy": common_sense_accuracy,
                }
            )
            .reset_index()
            .rename(columns={"index": "model_id"}),
            on="model_id",
            how="left",
        )
        .merge(
            metadata_models[["id", "params_b"]].rename(columns={"id": "model_id"}),
            on="model_id",
            how="left",
        )
        .dropna(subset=["common_sense_accuracy"])
    )

    if "params_b" not in eval_df_accuracy_total_per_model.columns:
        eval_df_accuracy_total_per_model["params_b"] = pd.NA
    eval_df_accuracy_total_per_model["common_sense_accuracy"] = pd.to_numeric(
        eval_df_accuracy_total_per_model["common_sense_accuracy"], errors="coerce"
    )

    plt.figure(figsize=figsize)
    sns.set_style("white")

    # Ensure numeric columns
    num_cols = ["common_sense_accuracy", "balanced_accuracy"]
    if "params_b" in eval_df_accuracy_total_per_model.columns:
        num_cols.append("params_b")
    for c in num_cols:
        eval_df_accuracy_total_per_model[c] = pd.to_numeric(
            eval_df_accuracy_total_per_model[c], errors="coerce"
        ).astype(float)

    balanced_max = eval_df_accuracy_total_per_model["balanced_accuracy"].max()
    common_max = eval_df_accuracy_total_per_model["common_sense_accuracy"].max()
    scale_balanced = pd.notna(balanced_max) and balanced_max <= 1.0 and common_max > 1.5
    scale_common = pd.notna(common_max) and common_max <= 1.0 and balanced_max > 1.5
    if scale_balanced:
        eval_df_accuracy_total_per_model["balanced_accuracy"] *= 100.0
    if scale_common:
        eval_df_accuracy_total_per_model["common_sense_accuracy"] *= 100.0

    # 1. Calculate Pearson Correlation
    # Drop NaNs to ensure accurate calculation
    corr_df = eval_df_accuracy_total_per_model.dropna(
        subset=["common_sense_accuracy", "balanced_accuracy"]
    )
    r_val, p_val = pearsonr(
        corr_df["common_sense_accuracy"], corr_df["balanced_accuracy"]
    )

    # 2. Regression Plot
    sns.regplot(
        data=eval_df_accuracy_total_per_model,
        x="common_sense_accuracy",
        y="balanced_accuracy",
        scatter=False,
        ci=95,
        line_kws={"color": "red", "lw": 1.5, "ls": "--"},
    )

    # 3. Scatter Plot
    for model_id, df_m in eval_df_accuracy_total_per_model.groupby("model_id"):
        color, marker, size = model_style[model_id]
        scatter_kwargs = dict(
            data=df_m,
            x="common_sense_accuracy",
            y="balanced_accuracy",
            # hue="mode",
            # marker="o",
            color=color,
            s=75+ 200 * size,
            marker=marker,
            edgecolor="w",
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
    # ax.set_ylim(0, 60)
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
    df_plot = eval_df_accuracy_total_per_model.query(
        "mode in ['image-only', 'general']"
    ).copy()

    hue_order = ["image-only", "general"]
    # Map old keys to new display names
    label_map = {"image-only": "single-frame", "general": "multi-frame"}

    palette = dict(zip(hue_order, sns.color_palette(n_colors=len(hue_order))))
    present = [k for k in hue_order if k in set(df_plot["mode"])]

    # Create handles with the new labels
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=palette[k],
            label=label_map.get(k, k),
        )
        for k in present
    ]
    # if show_legend:
    #     legend_mode = ax.legend(
    #         handles=handles,
    #         title="",
    #         loc="upper left",
    #         fontsize=legend_fontsize,
    #         title_fontsize=legend_fontsize,
    #         frameon=True,
    #     )

    # 6. Annotations
    annotate_df = eval_df_accuracy_total_per_model.dropna(
        subset=["common_sense_accuracy", "balanced_accuracy"]
    ).sort_values("balanced_accuracy", ascending=False)
    annotate_df["model_label"] = annotate_df["model_id"].map(_standardize_model_label)
    label_texts = []
    for _, r in annotate_df.iterrows():
        label = r["model_label"]
        versionname = metadata_models.loc[metadata_models["id"] == r["model_id"], "versionname"].values[0]
        # print(r["model_id"])
        offset = (0, 10)
        # ha = "left" if r["common_sense_accuracy"] < 45 else "right"
        if r["common_sense_accuracy"] < 45:
            offset = (offset[0]+10*(45-r["common_sense_accuracy"])/(45-35), offset[1])
        ha = "center"
        # if label in {"InternVL2.5-4B", "InternVL2.5-2B"}:
        #     offset = (-10, 0)
        #     ha = "right"
        label_texts.append(
            ax.annotate(
                versionname,
                xy=(r["common_sense_accuracy"], r["balanced_accuracy"]),
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
            size_refs = [s for s in size_refs if min_pos <= s <= max_pos]
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
                    markersize=_size_for_param(s) ** 0.5,
                    label=f"{s}B",
                )
                for s in size_refs
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
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.0,
    )
    print(f"Plot saved to: {out_dir}/{out_filename}")
