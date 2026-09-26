#!/usr/bin/env bash
# Example: serve Kev 4B (github.com/jaredpalmer/kev) on :8009 so bench.py can play it over /v1/systemone.
# Pinned to the code and weights we compared against. Needs a CUDA GPU and uv (pip install uv).
#   bash examples/kev.sh
#   python bench.py --judge mica=http://127.0.0.1:8010/v1/systemone --judge kev=http://127.0.0.1:8009/v1/systemone
set -e
KEV_COMMIT=73504e51f6ce2ade19c7819d4a5f2d84363cd40f
KEV_REV=139fdd94f1b6a6ad80cc15e08fcb99cac885a101
[ -d kev ] || git clone -q https://github.com/jaredpalmer/kev.git
cd kev && git fetch -q origin && git checkout -q "$KEV_COMMIT"
uv sync --extra serve
uv run --extra serve python -c "import fla" 2>/dev/null || uv pip install flash-linear-attention  # not pulled in by the serve extra
exec uv run --extra serve python -m kev.serve --run "jaredpalmer/kev-4b@$KEV_REV" --port 8009
