#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -f "${SCRIPT_DIR}/orion_env.sh" ]]; then
  # shellcheck disable=SC1090
  source "${SCRIPT_DIR}/orion_env.sh"
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON="${PYTHON_BIN}"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON="python"
else
  echo "[run.sh] Python 3.9+ was not found in PATH."
  exit 1
fi

exec "${PYTHON}" "${ROOT_DIR}/scripts/orion_portable.py" run "$@"
