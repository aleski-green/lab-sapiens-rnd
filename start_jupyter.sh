#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if curl --silent --fail http://127.0.0.1:8888/lab >/dev/null; then
  echo "JupyterLab is already running at http://127.0.0.1:8888/lab"
  exit 0
fi

open -a Terminal "$SCRIPT_DIR/serve_jupyter.sh"

echo "Starting JupyterLab in Terminal.app at http://127.0.0.1:8888/lab"
