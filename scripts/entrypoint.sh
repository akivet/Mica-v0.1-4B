#!/usr/bin/env bash
# Download the pinned weights once (GGUF + tokenizer files) and serve /v1/systemone on $PORT.
set -euo pipefail
W=${WEIGHTS_DIR:-/weights}
if [ ! -f "$W/$MICA_GGUF" ]; then
  hf download "$MICA_REPO" "$MICA_GGUF" tokenizer.json tokenizer_config.json chat_template.jinja config.json \
     --revision "$MICA_REVISION" --local-dir "$W"
fi
exec python3 -m mica.typesafe_server --gguf "$W/$MICA_GGUF" --hf "$W" --runtime /app/runtime \
     --calibration /app/calibration.json --flash-attn -1 --host 0.0.0.0 --port "$PORT" --name mica-v0.1-4b
