#!/usr/bin/env python3
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Set

BASE_DIR = Path(
    "/Users/sebastiancavada/Desktop/tmp_paris/tiny_vqa_creation/output/run_28_general"
)
RUN_DIR_RE = re.compile(r"results_run_28_general-(\d+)$")
OUTPUT_FORMAT = "pretty"  # "tsv" or "pretty"
MODEL_COL_MIN_WIDTH = 40

MODELS: List[str] = [
    # big
    "InternVL2-26B",
    "InternVL2-40B",
    "InternVL2_5-26B",
    "InternVL2_5-38B",
    # big (known not run)
    "InternVL2-76B",
    "InternVL2_5-78B",
    # small
    "instructblip-flan-t5-xl",
    "instructblip-flan-t5-xxl",
    "instructblip-vicuna-7b",
    "instructblip-vicuna-13b",
    "blip2-flant5xxl",
    "llava-1.5-7b-hf",
    "llava-1.5-13b-hf",
    "llava-v1.6-mistral-7b-hf",
    "llava-v1.6-vicuna-7b-hf",
    "deepseek1B",
    "deepseek7B",
    "Xinyuan-VL-2B",
    "Aquila-VL-2B",
    "MiniCPM-V2",
    "MiniCPM-V2.5",
    "MiniCPM-V2.6",
    "Qwen-VL-Chat",
    "cambrian-8b",
    "paligemma2-3b",
    "paligemma2-10b",
    "InternVL-Chat-V1-5-quantable",
    "MolmoE-1B",
    "MolmoE-7B-O",
    "MolmoE-7B-D",
    "Phi-3-vision-128k-instruct",
    "Phi-3.5V",
    "mPLUG-Owl3-1B-241014",
    "mPLUG-Owl3-2B-241014",
    "mPLUG-Owl3-7B-241101",
    "llava-interleave-qwen-7b-hf",
    "llava-interleave-qwen-7b-dpo-hf",
    "vila-1.5-3b",
    "vila-1.5-3b-s2",
    "vila-1.5-8b",
    "vila-1.5-13b",
    "LLaVA-NeXT-Video-7B-DPO-hf",
    "LLaVA-NeXT-Video-7B-hf",
    "InternVL2-1B",
    "InternVL2-2B",
    "InternVL2-4B",
    "InternVL2-8B",
    "InternVL2_5-1B",
    "InternVL2_5-2B",
    "InternVL2_5-4B",
    "InternVL2_5-8B",
    "Mantis-8B-Idefics2",
    "Mantis-llava-7b",
    "Mantis-8B-siglip-llama3",
    "Mantis-8B-clip-llama3",
]

RUNNING_MODELS = {
    "InternVL2-26B",
    "InternVL2-40B",
    "InternVL2_5-26B",
    "InternVL2_5-38B",
}
RUNNING_RUNS = set(range(1, 7))
QUEUED_RUNS = set(range(7, 17))
STARTED_MODELS = {"InternVL2-76B", "InternVL2_5-78B"}
STARTED_RUNS = set(range(1, 11))
NEVER_RUN_MODELS = set()


def normalize_model_name(filename: str) -> str:
    if filename.endswith(".json"):
        filename = filename[:-5]
    if filename.endswith("_val"):
        filename = filename[:-4]
    return filename


def status_for(model: str, run_num: int, has_json: bool) -> str:
    if has_json:
        return "✅"
    if model in STARTED_MODELS and run_num in STARTED_RUNS:
        return "🏃"
    if model in NEVER_RUN_MODELS:
        return "❌"
    if model in RUNNING_MODELS and run_num in RUNNING_RUNS:
        return "🏃"
    if model in RUNNING_MODELS and run_num in QUEUED_RUNS:
        return "⏳"
    return "❌"


def main() -> int:
    base_dir = BASE_DIR
    if not base_dir.is_dir():
        raise SystemExit(f"Base dir not found: {base_dir}")

    runs: Dict[int, Path] = {}
    for item in base_dir.iterdir():
        if not item.is_dir():
            continue
        match = RUN_DIR_RE.match(item.name)
        if not match:
            continue
        run_num = int(match.group(1))
        runs[run_num] = item

    if not runs:
        raise SystemExit("No results_run_28_general-* directories found.")

    run_numbers = sorted(runs.keys())

    run_models: Dict[int, Set[str]] = {}
    run_counts: Dict[int, int] = {}
    for run_num in run_numbers:
        models: Set[str] = set()
        json_files = [p for p in runs[run_num].iterdir() if p.is_file() and p.suffix == ".json"]
        for p in json_files:
            models.add(normalize_model_name(p.name))
        run_models[run_num] = models
        run_counts[run_num] = len(json_files)

    model_list = MODELS

    # Summary counts
    print("Run JSON counts:")
    for run_num in run_numbers:
        print(f"run_{run_num:02d}: {run_counts[run_num]}")
    print("")

    # Table header
    headers = ["model"] + [f"run_{n:02d}" for n in run_numbers]
    rows: List[List[str]] = []
    for model in model_list:
        row = [model]
        for run_num in run_numbers:
            row.append(status_for(model, run_num, model in run_models[run_num]))
        rows.append(row)

    def display_width(text: str) -> int:
        width = 0
        for ch in text:
            if unicodedata.combining(ch):
                continue
            if unicodedata.east_asian_width(ch) in {"W", "F"}:
                width += 2
            else:
                width += 1
        return width

    def pad_cell(text: str, width: int) -> str:
        pad = width - display_width(text)
        if pad <= 0:
            return text
        return text + (" " * pad)

    if OUTPUT_FORMAT == "tsv":
        print("\t".join(headers))
        for row in rows:
            print("\t".join(row))
    else:
        widths = [display_width(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], display_width(cell))
        widths[0] = max(widths[0], MODEL_COL_MIN_WIDTH)
        sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
        print(sep)
        header_line = "| " + " | ".join(pad_cell(h, widths[i]) for i, h in enumerate(headers)) + " |"
        print(header_line)
        print(sep)
        for row in rows:
            line = "| " + " | ".join(pad_cell(row[i], widths[i]) for i in range(len(headers))) + " |"
            print(line)
        print(sep)

    print("")
    print("Legend:")
    print("✅ = JSON exists")
    print("🏃 = running (manual override)")
    print("⏳ = queued (manual override)")
    print("❌ = not run")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
