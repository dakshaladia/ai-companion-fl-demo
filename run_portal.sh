#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"

VENV_PATH=${PORTAL_VENV:-.venv311}
if [ -d "$VENV_PATH" ]; then
  # shellcheck disable=SC1090
  source "$VENV_PATH"/bin/activate
fi

export PORTAL_HOST=${PORTAL_HOST:-0.0.0.0}
export PORTAL_PORT=${PORTAL_PORT:-5001}
export PORTAL_DEVICE=${PORTAL_DEVICE:-cpu}

python portal/app.py


