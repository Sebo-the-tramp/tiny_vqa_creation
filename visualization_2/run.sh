#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8086}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${SCRIPT_DIR}/.venv/bin/python" ]]; then
    PYTHON_BIN="${SCRIPT_DIR}/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

exec "${PYTHON_BIN}" -m uvicorn app:app --host "${HOST}" --port "${PORT}" --app-dir "${SCRIPT_DIR}"
