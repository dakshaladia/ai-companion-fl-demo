#!/bin/bash

set -euo pipefail

# Configure address/port and avoid tokenizer fork warnings
export FL_SERVER_ADDRESS=${FL_SERVER_ADDRESS:-127.0.0.1:8080}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
PORT="${FL_SERVER_ADDRESS##*:}"

# Select Python interpreter (prefer project venv)
PYTHON_BIN="python"
if [[ -d ".venv311" ]]; then
  # shellcheck disable=SC1091
  source .venv311/bin/activate
else
  for py in python3.12 python3.11 python3; do
    if command -v "$py" >/dev/null 2>&1; then
      PYTHON_BIN="$py"
      break
    fi
  done
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "❌ No compatible Python found. Run './setup_env.sh' first."
  exit 1
fi

# Ensure no stale Flower servers are occupying the port (fixes gRPC handshake timeouts)
if lsof -tiTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "🔪 Killing existing processes on port $PORT..."
  lsof -tiTCP:"$PORT" -sTCP:LISTEN | xargs kill -9 || true
fi

# Start server
echo "🚀 Starting FL server on $FL_SERVER_ADDRESS ..."
"$PYTHON_BIN" server/server.py &
SERVER_PID=$!

echo "⏳ Waiting for server to initialize (10 seconds)..."
sleep 10

# Start clients
echo "👥 Starting client 1..."
"$PYTHON_BIN" client/client.py &
sleep 2
echo "👥 Starting client 2..."
"$PYTHON_BIN" client/client.py &

echo "⏳ Waiting for training to complete..."
wait $SERVER_PID
