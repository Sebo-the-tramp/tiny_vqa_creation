#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "${SCRIPT_DIR}/api_server.py" \
  --host 0.0.0.0 \
  --port 8086 \
  --serve-root / \
  --run-name run_28_general \
  --question-file /data0/sebastian.cavada/compositional-physics/tiny_vqa_creation/output/run_28_general/test_run_28_general.json \
  --scenes-file /data0/sebastian.cavada/compositional-physics/tiny_vqa_creation/visualization/scenes.json
