#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/questions.json [mode]" >&2
  exit 1
fi

INPUT_PATH="/mnt/proj1/eu-25-92/tiny_vqa_creation/output/run_24_general/test_run_24_general.json"
MODE_ARG=()
if [[ $# -ge 2 && -n "${2:-}" ]]; then
  MODE_ARG=(--mode "$2")
fi

python max_questions_for_distribution.py \
  --input "$INPUT_PATH" \
  --percentage-map balancing_sub_categories.json \
  --show-breakdown \
  --integer-counts \
  "${MODE_ARG[@]}"
