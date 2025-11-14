import pandas as pd

# main function for analysis
def summarize_accuracy(df, by=("model_id",), sort=("accuracy","n"), ascending=(False,False)):
    g = df.groupby(list(by), observed=True, dropna=False).agg(
        n=("idx","count"),
        correct=("is_correct","sum")
    ).reset_index()
    g["accuracy"] = g["correct"] / g["n"]
    return g.sort_values(list(sort), ascending=list(ascending))

# Confusion (rows: truth, cols: pred) per model
def confusion(df, model_id):
    sub = df[(df["model_id"]==model_id) & (df["split"]=="val")]
    return pd.crosstab(sub["correct_answer"], sub["answer"], dropna=False).fillna(0).astype(int)

# Error list for a model (to inspect failures)
def error_table(df, model_id, k=50):
    sub = df[(df["model_id"]==model_id) & (df["split"]=="val") & (~df["is_correct"])]
    cols = ["idx","scene","difficulty","ability_type","sub_type","question","answer","correct_answer","file_name"]
    cols = [c for c in cols if c in sub.columns]
    return sub[cols].head(k).sort_values("scene")

def summarize_overall_accuracy(df):
    n = len(df)
    correct = df["is_correct"].sum()
    accuracy = correct / n if n > 0 else 0
    return {"n": n, "correct": correct, "accuracy": accuracy}
