import json
import pandas as pd
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
# Point these to your files
RUN_NAME = "_run_06_1K_roi_circling.json"
PATH = "./output/"
ANSWERS_PATH = f"{PATH}test{RUN_NAME}"
TEST_PATH    = f"{PATH}val_answer{RUN_NAME}"

# If your join key is not 'question_ID', set it here or leave as None to auto-detect
JOIN_KEY = "idx"

# -----------------------------
# IO
# -----------------------------
def load_json_records(path):
    path = Path(path)
    with path.open("r") as f:
        data = json.load(f)
    # Accept both list-of-dicts and dict-of-lists
    if isinstance(data, list):
        return pd.json_normalize(data)
    elif isinstance(data, dict):
        try:
            return pd.DataFrame(data)
        except Exception:
            return pd.json_normalize(data)
    else:
        raise ValueError(f"Unsupported JSON structure in {path}")

answers_df = load_json_records(ANSWERS_PATH)
test_df    = load_json_records(TEST_PATH)

# -----------------------------
# Join key detection
# -----------------------------
def pick_join_key(a_df, b_df, preferred=None):
    candidates = [preferred] if preferred else []
    candidates += ["idx"]
    for c in candidates:
        if c and c in a_df.columns and c in b_df.columns:
            return c
    # Heuristic: choose the most likely common key among intersections
    common = list(set(a_df.columns).intersection(set(b_df.columns)))
    if not common:
        raise KeyError("No common key found to merge the two files")
    # Prefer columns that look like ids
    common_sorted = sorted(common, key=lambda x: (not x.lower().endswith("id"), x))
    return common_sorted[0]

key = pick_join_key(answers_df, test_df, preferred=JOIN_KEY)

# -----------------------------
# Merge
# -----------------------------
merged = pd.merge(answers_df, test_df, on=key, how="inner")
if merged.empty:
    raise ValueError("Merged dataframe is empty. Check JOIN_KEY or input files")

print(f"Merged dataframes on key '{key}'. Resulting shape: {merged.shape}")
print(f"Columns in merged dataframe: {merged.columns.tolist()}")

# -----------------------------
# Mode inference helper
# -----------------------------
def infer_mode(df):
    """
    Returns a Series 'mode_inferred' with values:
    - 'image-only' if no question text or flagged so
    - 'general' otherwise
    Uses existing columns if available: 'mode' or 'question_mode'
    Falls back to heuristic based on 'question' text presence.
    """
    # If explicit mode exists, normalize and use it
    for col in ["mode_x", "question_mode", "type"]:
        if col in df.columns:
            s = df[col].astype(str).str.strip().str.lower()
            # Normalize some common variants
            s = s.replace({
                "image_only": "image-only",
                "imageonly": "image-only",
                "image": "image-only",
                "text": "general",
                "text_only": "general",
                "textonly": "general"
            })
            return s.rename("mode_inferred")

    # Heuristic: if 'question' column exists and is empty or NA -> image-only
    if "question" in df.columns:
        q = df["question"].fillna("").astype(str).str.strip()
        return q.apply(lambda x: "image-only" if x == "" else "general").rename("mode_inferred")

    # If we cannot infer, label as unknown
    return pd.Series(["unknown"] * len(df), index=df.index, name="mode_inferred")

merged["mode_inferred"] = infer_mode(merged)

# -----------------------------
# Pretty histogram printer
# -----------------------------
def print_histogram(series, title=None, width=40, label_width=24):
    counts = series.value_counts(dropna=False)
    total = int(counts.sum())
    if total == 0:
        return
    if title:
        print(f"\n=== {title} ===")
    for val, cnt in counts.items():
        pct = cnt / total
        bar = "█" * int(pct * width)
        label = str(val)[:label_width]
        print(f"{label:<{label_width}} | {bar:<{width}} {cnt} ({pct:.1%})")

# -----------------------------
# General summary
# -----------------------------
total_qs = merged[key].nunique()
unique_answers = merged["answer"].nunique() if "answer" in merged.columns else 0

print("=== Summary ===")
print(f"Total questions: {total_qs}")
print(f"Unique answers: {unique_answers}")

# Scene counts
def pick_scene_column(df):
    priority = ["scene", "scene_id", "scene_idx", "scene_index", "scene_name"]
    priority.extend([c for c in df.columns if "scene" in c.lower() and c not in priority])
    for col in priority:
        if col in df.columns:
            return col
    return None

scene_col = pick_scene_column(merged)
if scene_col:
    print_histogram(merged[scene_col], title=f"Questions per {scene_col}")
else:
    print("\nNo scene column found in the merged data. Available columns:")
    print(", ".join(merged.columns))

# Mode breakdown
print_histogram(merged["mode_x"], title="Mode breakdown")

# Category and sub_category counts if present
if "category" in merged.columns:
    print_histogram(merged["category"], title="Category counts")
if "sub_category" in merged.columns:
    print_histogram(merged["sub_category"], title="Sub-category counts")

# Missingness quick check for key cols
important_cols = [c for c in [key, "answer", "category", "sub_category", "mode_x"] if c in merged.columns]
if important_cols:
    na_counts = merged[important_cols].isna().sum()
    if na_counts.any():
        print("\n=== Missing values (selected columns) ===")
        for c, n in na_counts.items():
            if n > 0:
                print(f"{c}: {n}")

# -----------------------------
# Answer distributions
# -----------------------------
if "answer" in merged.columns:
    # Overall
    print_histogram(merged["answer"], title="Overall answer distribution")

    # By category
    if "category" in merged.columns:
        for cat, sub in merged.groupby("category"):
            print_histogram(sub["answer"], title=f"Answer distribution — category: {cat}")

    # By sub_category
    if "sub_category" in merged.columns:
        for subcat, sub in merged.groupby("sub_category"):
            print_histogram(sub["answer"], title=f"Answer distribution — sub_category: {subcat}")

    # # # By question_ID
    # for qid_col in ["question_id"] if key else []:
    #     for qid, sub in merged.groupby(qid_col):
    #         print_histogram(sub["answer"], title=f"Answer distribution — {qid_col}: {qid}")
else:
    print("\nNo 'answer' column found in the merged data. Check your answers file.")


# python show_answer_distributions.py 
