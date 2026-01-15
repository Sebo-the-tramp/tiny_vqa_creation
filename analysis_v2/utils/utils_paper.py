import numpy as np
import pandas as pd
import json


def print_summary_models_used(eval_df):
    # here we define Models_IDs and their definition

    eval_df_single_image_models_unique = eval_df[eval_df["idx"].str.contains("_i")][
        "model_id"
    ].unique()
    eval_df_multi_image_models_unique = eval_df[eval_df["idx"].str.contains("_g")][
        "model_id"
    ].unique()

    single_images = np.setdiff1d(
        eval_df_single_image_models_unique, eval_df_multi_image_models_unique
    )
    multi_images = eval_df_multi_image_models_unique

    # 1. Load Metadata
    try:
        with open(
            "/data0/sebastian.cavada/compositional-physics/tiny_vqa_deterministic/analysis_example/utils/metadata.json",
            "r",
        ) as f:
            metadata_list = json.load(f)
        meta_dict = {item["id"]: item for item in metadata_list}
    except FileNotFoundError:
        print("Error: 'metadata.json' not found.")
        meta_dict = {}

    # 2. Define Groups (using your previous logic)
    # Ensure you have your dataframes loaded as 'eval_df'
    # For this example, I will assume the lists are already generated in variables:
    # only_images = ... (from previous step)
    # multi_images = ... (from previous step)

    # Combine sorted lists for ID generation
    # (Single images get M1..Mx, Multi images get Mx..My)
    all_models = list(single_images) + list(multi_images)

    all_models = np.sort(all_models)

    # print(len(all_models), "models found for LaTeX generation.")
    # print(all_models)

    # 3. Create the Short ID Map (M1, M2, M3...)
    id_map = {}
    for idx, original_id in enumerate(all_models, 1):
        id_map[original_id] = f"M{idx}"

    # --- HELPER: Escape special LaTeX characters ---
    def tex_escape(text):
        """
        Escapes characters that break LaTeX:
        _ -> \_  (underscore)
        % -> \%  (percent)
        # -> \#  (hash)
        & -> \&  (ampersand)
        """
        if not isinstance(text, str):
            return str(text)

        return (
            text.replace("_", r"\_")
            .replace("%", r"\%")
            .replace("#", r"\#")
            .replace("&", r"\&")
        )

    # 4. Generator Function
    def get_model_latex(original_id):
        info = meta_dict.get(original_id, {})
        short_id = id_map[original_id]

        family = tex_escape(info.get("family", "Unknown Family"))
        source = tex_escape(info.get("source", "N/A"))

        if info.get("params_b"):
            params = f"{info['params_b']:.2f}B"
        else:
            params = "Proprietary/Unknown"

        license_ = tex_escape(info.get("license", "N/A"))
        year = info.get("release_year", "N/A")

        return f"""
    \\paragraph{{{tex_escape(original_id)} ({short_id})}} 
    \\label{{mod:{short_id}}}
    \\begin{{itemize}}
        \\item \\textbf{{Source:}} \\\\ \\texttt{{{source}}}
        \\item \\textbf{{Parameters:}} {params}
        \\item \\textbf{{Release Year:}} {year}
        \\item \\textbf{{License:}} {license_}
        \\item \\textbf{{Family:}} {family}
    \\end{{itemize}}
    % Context for {original_id}:
    """

    # 5. Generate Section
    latex_output = []
    latex_output.append(r"\section{Model Specifications}")
    latex_output.append(
        r"This section provides detailed specifications for each of the models evaluated in this study. Models are categorized based on their input capabilities: single-image models and multi-image/general models. Each model is identified by a unique short ID (M1, M2, etc.) for ease of reference throughout the document."
    )

    latex_output.append(r"\subsection{Single-Image Models}")
    latex_output.append(
        r"This subsection provides specifications for single-image models. Those models can only handle one single image as input."
    )
    if len(single_images) > 0:
        for model in single_images:
            latex_output.append(get_model_latex(model))
    else:
        latex_output.append(r"No single-image models found.")

    latex_output.append(r"\subsection{Multi-Image \& General Models}")
    latex_output.append(
        r"This subsection provides specifications for multi-image and general models. Those models can handle multiple images as input. In our setup we provide up to eight images sampled uniformly."
    )
    if len(multi_images) > 0:
        for model in multi_images:
            latex_output.append(get_model_latex(model))

    # 6. Print LaTeX
    print("\n".join(latex_output))

    # # 7. PRINT SORTED MAPPING
    # print("\n" + "="*40)
    # print("   SORTED MODEL ID MAPPING (A-Z)")
    # print("="*40)
    # for orig, short in id_map.items():
    #     print(f"{short} : {orig}")


def print_heatmap_table_latex(
    acc_mat: pd.DataFrame, output_path: str = "heatmap_table.txt"
) -> str:
    # 2. Clean Data
    acc_safe = acc_mat.apply(pd.to_numeric, errors="coerce").fillna(0).round(2)

    # 3. FIX ORIENTATION (Swap the crossed-over Avg/Total)
    # After transposing:
    # - The Column 'Total' holds the Question Averages. We want this as Column 'Average' at [0].
    # - The Row 'Average' holds the Model Averages. We want this as Row 'Total' at [-1].

    # A. Handle the First Column (Question Difficulty)
    # If we have a column named 'Total' (from the old row), rename it to 'Average' and move to front
    if "Total" in acc_safe.columns:
        # Rename 'Total' -> 'Average'
        acc_safe.rename(columns={"Total": "Average"}, inplace=True)

        # Move 'Average' to Index 0
        col_data = acc_safe.pop("Average")
        acc_safe.insert(0, "Average", col_data)

    # B. Handle the Last Row (Model Performance)
    # If we have a row named 'Average' (from the old col), rename it to 'Total'
    if "Average" in acc_safe.index:
        # Rename index 'Average' -> 'Total'
        acc_safe.rename(index={"Average": "Total"}, inplace=True)

        # Move 'Total' to the absolute bottom
        # (Extract, drop, append)
        row_data = acc_safe.loc["Total"]
        acc_safe.drop(index="Total", inplace=True)
        acc_safe.loc["Total"] = row_data

    # 4. Generate Mappings
    vlm_id_map = {}
    m_cnt = 1
    for col in acc_safe.columns:
        if col == "Average":
            vlm_id_map[col] = r"\textbf{Average}"
        else:
            vlm_id_map[col] = f"M{m_cnt}"
            m_cnt += 1

    question_id_map = {}
    q_cnt = 1
    for idx in acc_safe.index:
        if idx == "Total":
            question_id_map[idx] = r"\textbf{Total}"
        else:
            question_id_map[idx] = f"Q{q_cnt}"
            q_cnt += 1

    # 5. Rename Dataframe
    acc_display = acc_safe.rename(columns=vlm_id_map, index=question_id_map)

    # 6. LaTeX Function (Vertical Center + Lines)
    def latex_tables_by_vlm_chunks(df, chunk_size=10, cmap="RdYlGn"):
        latex_tables = []
        n_cols = df.shape[1]

        for start in range(0, n_cols, chunk_size):
            end = start + chunk_size
            chunk = df.iloc[:, start:end]

            cols = chunk.columns.tolist()
            caption_text = f"Accuracy for {cols[0]} to {cols[-1]}"

            # --- A. Column Format (Vertical Lines) ---
            # If 'Average' is the first column in this chunk
            is_first_chunk = start == 0

            if is_first_chunk:
                # Index | Avg | M1 ...
                col_fmt = "l|c|" + "c" * (chunk.shape[1] - 1)
            else:
                # Index | M_x ...
                col_fmt = "l|" + "c" * chunk.shape[1]

            # --- B. Style ---
            styled = (
                chunk.style.format("{:.2f}")
                .background_gradient(cmap=cmap, axis=None)
                .set_properties(**{"border": "0pt none"})
            )

            inner_latex = styled.to_latex(
                convert_css=True,
                hrules=True,
                clines=None,
                column_format=col_fmt,
                position=None,
            )

            # --- C. Inject LaTeX Fixes ---
            inner_latex = inner_latex.replace(
                r"\begin{tabular}", r"\small\begin{tabular}"
            )
            inner_latex = inner_latex.replace(
                r"\textbf{Total}", r"\midrule \textbf{Total}"
            )

            # --- D. Wrap with [p] for Vertical Centering ---
            full_latex_float = (
                "\\begin{table}[p]\n"
                f"\\caption{{{caption_text}}}\n"
                "\\centering\n"
                f"{inner_latex}"
                "\\end{table}"
            )

            latex_tables.append(full_latex_float)

        return latex_tables

    # --- Execution ---
    tables = latex_tables_by_vlm_chunks(acc_display, chunk_size=15)

    sections = ["% ================= HEATMAP TABLES =================\n"]
    for t in tables:
        sections.append(t)
        sections.append(r"\clearpage")
        sections.append("\n")

    latex_text = "\n".join(sections)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex_text)

    return latex_text
