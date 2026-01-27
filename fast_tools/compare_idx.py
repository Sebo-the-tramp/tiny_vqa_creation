#!/usr/bin/env python3
import json

FILE_A = "/data0/sebastian.cavada/compositional-physics/tiny_vqa_creation/output/run_25_roi_ablation_baseline/test_run_25_roi_ablation_baseline_karo_10K.json"
FILE_B = "/data0/sebastian.cavada/compositional-physics/tiny_vqa_creation/output/run_25_roi_circling_no_text/test_run_25_roi_circling_no_text_karo_10K.json"
VAL_A = "/data0/sebastian.cavada/compositional-physics/tiny_vqa_creation/output/run_25_roi_ablation_baseline/val_answer_run_25_roi_ablation_baseline.json"
VAL_B = "/data0/sebastian.cavada/compositional-physics/tiny_vqa_creation/output/run_25_roi_circling_no_text/val_answer_run_25_roi_circling_no_text.json"


def load_idx_set(path: str) -> set:
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}, got {type(data).__name__}")
    return {item["idx"] for item in data if "idx" in item}

def load_idx_answer_map(path: str) -> dict:
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}, got {type(data).__name__}")
    idx_to_answer = {}
    for item in data:
        if "idx" not in item or "answer" not in item:
            continue
        idx = item["idx"]
        ans = item["answer"]
        if idx in idx_to_answer and idx_to_answer[idx] != ans:
            raise ValueError(f"Conflicting answers for idx {idx} in {path}")
        idx_to_answer[idx] = ans
    return idx_to_answer


def main() -> None:
    idx_a = load_idx_set(FILE_A)
    idx_b = load_idx_set(FILE_B)

    common = idx_a & idx_b
    total_unique = idx_a | idx_b

    print(f"{FILE_A}: {len(idx_a)} idx values")
    print(f"{FILE_B}: {len(idx_b)} idx values")
    print(f"Total unique idx values: {len(total_unique)}")
    print(f"Idx values in common: {len(common)}")

    val_a = load_idx_answer_map(VAL_A)
    val_b = load_idx_answer_map(VAL_B)
    val_common = set(val_a) & set(val_b)
    same_answer = sum(1 for idx in val_common if val_a[idx] == val_b[idx])
    diff_answer = len(val_common) - same_answer

    print(f"{VAL_A}: {len(val_a)} idx values")
    print(f"{VAL_B}: {len(val_b)} idx values")
    print(f"Val idx values in common: {len(val_common)}")
    print(f"Val idx with same answer: {same_answer}")
    print(f"Val idx with different answer: {diff_answer}")


if __name__ == "__main__":
    main()
