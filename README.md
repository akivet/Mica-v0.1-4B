# Mica v0.1 4B

Mica is a small decision model. You give it a state, a question and the allowed answers, and it returns a probability
for each answer: yes/no, a choice among 2–255 options, or a score with 2–10 levels. It reads the input once and
generates no text, so a decision costs one prefill.

It speaks the TypeSafe `/v1/systemone` format, so clients written for Jev work unchanged. It was trained on English
and Korean.

- Base model: `Qwen/Qwen3.5-4B` (revision `851bf6e`) with a merged rank-16 LoRA on all 32 layers
- Weights: [`sky7350/Mica-v0.1-4B`](https://huggingface.co/sky7350/Mica-v0.1-4B) (revision `ca36594`): merged BF16
  safetensors and GGUF in BF16, Q8_0, Q6_K, Q5_K_M, Q4_K_M and Q4_0
- License: Apache-2.0 (see `NOTICE`)

## Quick start

With Docker:

```bash
docker build -t mica-v0.1-4b .
docker run --gpus all -p 8010:8010 -v mica-weights:/weights mica-v0.1-4b
```

The first start downloads the BF16 GGUF (9.7 GB). Use `-e MICA_GGUF=mica-v0.1-4b-Q5_K_M.gguf` for a smaller file.
The image builds for CUDA architectures 86, 89 and 90; pass `--build-arg CUDA_ARCH=...` to change that.

Without Docker (Linux, CUDA 12, Python 3.10+):

```bash
pip install -r requirements.txt
bash scripts/build_runtime.sh            # builds llama.cpp b11010 into ./runtime (CUDA_ARCH=86 by default)
hf download sky7350/Mica-v0.1-4B mica-v0.1-4b-BF16.gguf tokenizer.json tokenizer_config.json chat_template.jinja config.json \
   --revision ca36594cc2067c7252704f9f304cc10ef11c7c5c --local-dir weights
bash scripts/serve.sh                    # http://127.0.0.1:8010/v1/systemone
```

A request:

```bash
curl -s localhost:8010/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "The user asked to delete the staging database. No approval has been given.",
  "questions": {"q": {"type": "noul", "instructions": "Should the agent delete it now?"}}}'
```

Inputs longer than 8,192 tokens are rejected with HTTP 400 rather than truncated.

## JevBench

The unchanged `typesafe` adapter works against the local server:

```bash
git clone https://github.com/fstandhartinger/jevbench && mkdir -p runs && cd jevbench
TYPESAFE_API_KEY=unused python -m jevbench.cli run --adapter typesafe --endpoint http://127.0.0.1:8010 \
  --model mica-v0.1-4b --tasks datasets/public/easy.jsonl,datasets/public/original.jsonl,datasets/public/hard.jsonl \
  --results ../runs/mica-v0.1-4b.jsonl --run-label mica-v0.1-4b
```

On a fresh RTX 3090 (BF16 GGUF, one request at a time) all 231 public items returned valid answers:
easy 1.000, original 1.000, hard 0.649, ECE 0.064, p50 54 ms, p95 552 ms. Per-item output is in
`results/public231/mica-v0.1-4b.jsonl`.

No JevBench item was used for training (exact-match and 8-gram checks against the 231 public items find none in the
77,732 training rows). Half of the public hard items, together with Kev transfer v9 and SemIf, were used to compare
recipes during development and to pick the LoRA scale, which stayed at 1.0.

## Results

Accuracy in %. These come from our own evaluation code with the BF16 model in PyTorch. Through the llama.cpp server
and JevBench's runner, the public hard tier scores 64.9 instead of 69.5; easy and original are the same. When a model
cannot take an input (language, length, format), the item counts as wrong. JEV 1.13 is TypeSafe's hosted model.

Public sets:

| Set | n | Mica | JEV 1.13 | Qwen3.5-4B | JevK5 4B | Kev 4B | Nimble 9B | Laya |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| JevBench easy | 48 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 97.9 |
| JevBench original | 72 | 100.0 | 98.3 | 91.7 | 93.3 | 91.7 | 91.7 | 65.0 |
| JevBench hard | 111 | 69.5 | 74.3 | 61.0 | 76.2 | 52.4 | 23.8 | 10.5 |
| SemIf | 252 | 94.4 | 98.4 | 80.2 | 86.1 | 89.3 | 93.2 | 65.1 |
| Kev transfer v9 | 1,264 | 69.2 | 82.0 | 66.1 | 70.5 | 73.5 | – | 53.8 |
| MMLU-Pro | 10,032 | 53.0 | 82.3 | 45.2 | 53.5 | 49.7 | – | – |

Our own sets (English and Korean). The held-out set was written after training data was frozen and was not opened
until training finished.

| Set | n | Mica | JEV 1.13 | Qwen3.5-4B | JevK5 4B | Kev 4B | Nimble 9B | Laya |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Held-out | 7,328 | 67.4 | 74.7 | 54.4 | 25.5 | 56.4 | 53.9 | 25.3 |
| Held-out, English only | 3,044 | 67.0 | 74.1 | 55.0 | 61.0 | 57.0 | 54.8 | 32.6 |
| Held-out, Korean only | 4,284 | 67.7 | 75.2 | 54.1 | – | 55.9 | 53.3 | 20.1 |
| Dev | 2,222 | 78.9 | 79.2 | 60.1 | 29.3 | 65.1 | 67.0 | 34.8 |
| Perturbed inputs | 2,752 | 77.3 | 73.3 | 51.6 | 29.6 | 48.7 | 44.5 | 20.8 |
| Korean chat style | 105 | 86.7 | 81.0 | 55.2 | – | 69.5 | 75.2 | 21.9 |
| Long inputs, many options | 1,904 | 75.6 | 77.8 | 54.1 | 18.9 | 59.9 | 36.8 | 14.2 |
| Knowledge | 322 | 81.1 | 91.9 | 65.2 | 42.2 | 68.0 | 73.9 | 39.1 |
| Concept transfer | 344 | 71.8 | 72.1 | 48.0 | 47.1 | 51.4 | 51.7 | 25.3 |

JevK5 is English-only and Laya accepts up to 1,024 tokens, so many of their rows above are unsupported. The long-input,
chat-style and knowledge sets are close to the training distribution; read them as in-distribution numbers.

The perturbed set adds unrelated text, long padding, a buried case, reordered options and similar changes. The
largest gap is a note inside the state that tells the model to pick a wrong option: Mica still gets 69.1 % right,
JEV 17.5 %, Kev 31.4 %, base Qwen3.5-4B 1.0 %. The same note pointing at the right option lifts Mica to 88.7 %, so
such notes still move it.

Calibration on the held-out set: ECE 5.4 %, and 2.5 % of answers are wrong with confidence of 0.9 or more
(JEV: 3.8 % and 2.0 %). Full metrics are in `results/all_metrics.md`.

### Latency

RTX 3090, one request at a time, JevBench public items:

| Model | p50 | p90 |
|---|---:|---:|
| Mica Q4_K_M | 47 ms | 466 ms |
| Mica BF16 | 54 ms | 526 ms |
| Kev 4B | 76 ms | 325 ms |
| JevK5 4B | 99 ms | 367 ms |
| Nimble 9B | 132 ms | 343 ms |

The p90 comes from the hard items, whose inputs reach about 3.7k tokens.

### Quantized files

On our 1,402-item calibration set, through the same server on an RTX 3090:

| File | Size | Accuracy | NLL | Same answer as BF16 |
|---|---:|---:|---:|---:|
| `mica-v0.1-4b-BF16.gguf` | 9.70 GB | 79.03 | 0.451 | – |
| `mica-v0.1-4b-Q8_0.gguf` | 5.16 GB | 79.39 | 0.449 | 98.9 % |
| `mica-v0.1-4b-Q6_K.gguf` | 3.99 GB | 78.89 | 0.449 | 97.6 % |
| `mica-v0.1-4b-Q5_K_M.gguf` | 3.51 GB | 79.32 | 0.463 | 96.1 % |
| `mica-v0.1-4b-Q4_K_M.gguf` | 3.07 GB | 78.53 | 0.464 | 91.5 % |
| `mica-v0.1-4b-Q4_0.gguf` | 2.90 GB | 77.53 | 0.481 | 90.0 % |

Q5_K_M is a good choice for an 8 GB GPU.

## How it works

The model is Qwen3.5-4B unchanged in shape: 32 layers, 24 of them Gated DeltaNet and 8 full attention. A LoRA of rank
16 on the attention and Gated DeltaNet projections was trained and merged, and no heads were added.

Each option gets a single-token label from a fixed list of 255 (yes/no uses the model's own "No" and "Yes"). The
probability of an option is the softmax of those label logits at the answer position, divided by a temperature of
1.124 fitted on our calibration set. Several questions about the same state share its prefill.

Training used plain cross-entropy on the verified answer for one epoch (learning rate 1e-4, 5 % warmup, linear decay,
batch 32). The data has about 34k source decisions and 78k rows, since most decisions appear in both a conversational
and a structured wording. It covers 12 areas, including coding agents, code review, computer use, user requests,
documents, policy rules, dates and quantities, routing, state tracking, games and general knowledge. Answers were
checked by running code where that is possible and by sampled review elsewhere. Of three seeds, we kept the one with
the best accuracy on the calibration set.

## Limitations

- Knowledge-heavy questions: 53.0 on MMLU-Pro against 82.3 for JEV.
- Long English policy documents (the JevBench hard tier) are the weakest public set.
- Instructions planted inside the state still influence answers somewhat.
