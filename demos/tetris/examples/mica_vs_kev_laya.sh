#!/usr/bin/env bash
# The exact comparison in results/: Mica (Q5_K_M) vs Kev 4B vs Laya, 3 seeds, both scaffolds, then the two videos.
# Run from demos/tetris with the Mica server on :8010 (see the repo README) and examples/kev.sh on :8009.
set -e
python bench.py --judge mica=http://127.0.0.1:8010/v1/systemone --judge kev=http://127.0.0.1:8009/v1/systemone \
  --judge laya=py:examples/laya_judge.py --seeds 7 11 23 --pieces 250 --scaffold both --out runs
python render.py runs/easy/mica_s11.jsonl runs/easy/kev_s11.jsonl --pieces 100 --gpu "RTX 3090" --out videos/mica_vs_kev.mp4
python render.py runs/easy/mica_s11.jsonl runs/easy/laya_s11.jsonl --pieces 100 --gpu "RTX 3090" --out videos/mica_vs_laya.mp4
