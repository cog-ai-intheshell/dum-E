#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-$ROOT_DIR/.venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

cd "$ROOT_DIR"
exec "$PYTHON_BIN" -m server.server \
  --model-path "${DUME_WORLD_MODEL_PATH:-model/artifacts/dum_e_world_model.pt}" \
  --controller-path "${DUME_CONTROLLER_PATH:-model/artifacts/dum_e_catch_controller.pt}" \
  --host "${DUME_SERVER_HOST:-127.0.0.1}" \
  --port "${DUME_SERVER_PORT:-8765}" \
  "$@"
