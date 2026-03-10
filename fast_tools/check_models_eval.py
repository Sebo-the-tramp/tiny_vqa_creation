#!/usr/bin/env python3
import re
from pathlib import Path

RUN_PARALLEL_PATH = Path("/Users/sebastiancavada/Desktop/tmp_paris/PhysBench-Sebfork/run_parallel.py")

# Hardcoded inputs: each item can be either a run folder (contains results_* dirs)
# or a direct results folder.
TARGET_INPUT_PATHS = [
    # Path("/Users/sebastiancavada/Desktop/tmp_paris/tiny_vqa_creation/output/run_28_ablation_no_object"),
    # Path("/Users/sebastiancavada/Desktop/tmp_paris/tiny_vqa_creation/output/run_28_ablation_physics_duration_text"),
    # Path("/Users/sebastiancavada/Desktop/tmp_paris/tiny_vqa_creation/output/run_28_ablation_physics_mass_approx_text"),
    # Path("/Users/sebastiancavada/Desktop/tmp_paris/tiny_vqa_creation/output/run_28_ablation_physics_mass_text"),

    Path("/Users/sebastiancavada/Desktop/tmp_paris/tiny_vqa_creation/output/run_28_counterfactual_gravity"),
    Path("/Users/sebastiancavada/Desktop/tmp_paris/tiny_vqa_creation/output/run_28_counterfactual_shift"),
    Path("/Users/sebastiancavada/Desktop/tmp_paris/tiny_vqa_creation/output/run_28_counterfactual_smaller"),

    # Path("/Users/sebastiancavada/Desktop/tmp_paris/tiny_vqa_creation/output/run_28_no_roi_circling_no_text_layout_position"),
    # Path("/Users/sebastiancavada/Desktop/tmp_paris/tiny_vqa_creation/output/run_28_no_roi_circling_yes_text_layout_position"),
    # Path("/Users/sebastiancavada/Desktop/tmp_paris/tiny_vqa_creation/output/run_28_roi_ablation_baseline"),
    # Path("/Users/sebastiancavada/Desktop/tmp_paris/tiny_vqa_creation/output/run_28_roi_circling_no_text"),
    # Path("/Users/sebastiancavada/Desktop/tmp_paris/tiny_vqa_creation/output/run_28_roi_circling_no_text_layout_position"),
    # Path("/Users/sebastiancavada/Desktop/tmp_paris/tiny_vqa_creation/output/run_28_roi_circling_text"),
    # Path("/Users/sebastiancavada/Desktop/tmp_paris/tiny_vqa_creation/output/run_28_roi_circling_text_layout_position"),

    # Path("/Users/sebastiancavada/Desktop/tmp_paris/tiny_vqa_creation/output/run_24_general_yms_variations/results_run_24_general_yms_variations")

    # Path("/Users/sebastiancavada/Desktop/tmp_paris/tiny_vqa_creation/output/run_26_general_levels/results_run_26_general_levels")


]


def extract_models(run_parallel_path: Path) -> list[str]:
    source = run_parallel_path.read_text(encoding="utf-8")
    pattern = re.compile(r"""['"]model['"]\s*:\s*['"]([^'"]+)['"]""")
    models: list[str] = []
    seen: set[str] = set()

    for match in pattern.finditer(source):
        model = match.group(1)
        if model not in seen:
            seen.add(model)
            models.append(model)

    return models


def is_model_present(model: str, json_filenames: set[str]) -> bool:
    exact = f"{model}.json"
    prefix = f"{model}_"
    return any(name == exact or name.startswith(prefix) for name in json_filenames)


def collect_json_names(input_path: Path) -> tuple[set[str], int]:
    if not input_path.exists() or not input_path.is_dir():
        return set(), 0

    if input_path.name.startswith("results_"):
        results_dirs = [input_path]
    else:
        results_dirs = sorted(p for p in input_path.glob("results_*") if p.is_dir())
        if not results_dirs:
            # Fallback: allow direct json files in the folder.
            if any(input_path.glob("*.json")):
                results_dirs = [input_path]

    json_names: set[str] = set()
    for results_dir in results_dirs:
        for p in results_dir.glob("*.json"):
            json_names.add(p.name)

    return json_names, len(results_dirs)


def print_model_table(rows: list[tuple[str, bool]]) -> None:
    model_col = "Model"
    status_col = "Status"

    status_strings = [("✅ Present" if ok else "❌ Missing") for _, ok in rows]
    model_width = max(len(model_col), *(len(model) for model, _ in rows))
    status_width = max(len(status_col), *(len(s) for s in status_strings))

    sep = f"+-{'-' * model_width}-+-{'-' * status_width}-+"
    print(sep)
    print(f"| {model_col.ljust(model_width)} | {status_col.ljust(status_width)} |")
    print(sep)
    for (model, ok), status in zip(rows, status_strings):
        print(f"| {model.ljust(model_width)} | {status.ljust(status_width)} |")
    print(sep)


def print_summary_table(rows: list[tuple[str, int, int, str, str]]) -> None:
    folder_col = "Folder"
    present_col = "Present"
    missing_col = "Missing"
    coverage_col = "Coverage"
    status_col = "Status"

    folder_width = max(len(folder_col), *(len(r[0]) for r in rows))
    present_width = max(len(present_col), *(len(str(r[1])) for r in rows))
    missing_width = max(len(missing_col), *(len(str(r[2])) for r in rows))
    coverage_width = max(len(coverage_col), *(len(r[3]) for r in rows))
    status_width = max(len(status_col), *(len(r[4]) for r in rows))

    sep = (
        f"+-{'-' * folder_width}-+-{'-' * present_width}-+-{'-' * missing_width}"
        f"-+-{'-' * coverage_width}-+-{'-' * status_width}-+"
    )
    print(sep)
    print(
        f"| {folder_col.ljust(folder_width)} | {present_col.rjust(present_width)} | "
        f"{missing_col.rjust(missing_width)} | {coverage_col.rjust(coverage_width)} | "
        f"{status_col.ljust(status_width)} |"
    )
    print(sep)
    for folder, present, missing, coverage, status in rows:
        print(
            f"| {folder.ljust(folder_width)} | {str(present).rjust(present_width)} | "
            f"{str(missing).rjust(missing_width)} | {coverage.rjust(coverage_width)} | "
            f"{status.ljust(status_width)} |"
        )
    print(sep)


def main() -> None:
    if not RUN_PARALLEL_PATH.exists():
        raise FileNotFoundError(f"Missing file: {RUN_PARALLEL_PATH}")

    models = extract_models(RUN_PARALLEL_PATH)
    existing_inputs = sorted((p for p in TARGET_INPUT_PATHS if p.exists()), key=lambda p: p.name)
    missing_inputs = sorted((p for p in TARGET_INPUT_PATHS if not p.exists()), key=lambda p: p.name)

    if not existing_inputs:
        raise FileNotFoundError("None of TARGET_INPUT_PATHS exists.")

    if missing_inputs:
        print("Warning: these paths do not exist and were skipped:")
        for path in missing_inputs:
            print(f"- {path}")
        print()

    total_models = len(models)

    if len(existing_inputs) == 1:
        input_path = existing_inputs[0]
        json_filenames, results_dir_count = collect_json_names(input_path)
        rows = [(model, is_model_present(model, json_filenames)) for model in models]
        present = sum(1 for _, ok in rows if ok)

        print(f"Input: {input_path}")
        print(f"Detected results folders: {results_dir_count}")
        print_model_table(rows)
        print(f"\nSummary: {present}/{total_models} models have a JSON result file.")
        return

    summary_rows: list[tuple[str, int, int, str, str]] = []
    for input_path in existing_inputs:
        json_filenames, results_dir_count = collect_json_names(input_path)
        present = sum(1 for model in models if is_model_present(model, json_filenames))
        missing = total_models - present
        coverage = f"{(present / total_models * 100):5.1f}%"
        status = "✅ Complete" if present == total_models else "❌ Incomplete"
        label = input_path.name
        if results_dir_count > 1:
            label = f"{label} ({results_dir_count} results dirs)"
        summary_rows.append((label, present, missing, coverage, status))

    print_summary_table(summary_rows)
    print(f"\nModel universe: {total_models} models from {RUN_PARALLEL_PATH.name}")


if __name__ == "__main__":
    main()
