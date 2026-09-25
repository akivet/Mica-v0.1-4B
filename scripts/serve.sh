#!/usr/bin/env bash
# Serve Mica v0.1 4B on the TypeSafe /v1/systemone wire format.
# usage: bash scripts/serve.sh [GGUF] [PORT]   (default: the BF16 GGUF, port 8010)
set -euo pipefail
GGUF=${1:-weights/mica-v0.1-4b-BF16.gguf}
PORT=${2:-8010}
python -m mica.typesafe_server --gguf "$GGUF" --hf weights --runtime runtime \
  --calibration calibration.json --flash-attn -1 --port "$PORT" --name mica-v0.1-4b
