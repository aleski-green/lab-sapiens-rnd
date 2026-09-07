#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "$SCRIPT_DIR/../.jupyter-venv/bin/jupyter" lab \
  --no-browser \
  --ip=127.0.0.1 \
  --port=8888 \
  --ServerApp.port_retries=0 \
  --ServerApp.root_dir="$SCRIPT_DIR" \
  --ServerApp.token= \
  --ServerApp.password=
