#!/usr/bin/env bash
# Full evaluation suite, standard Hindi GPT vs the no-positional ablation.
# Runs locally: both checkpoints, the same bins, the same tokenizer.
# Usage (from the repository root):  bash ablation/run_eval.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
OUT=reports/ablation

run() { echo; echo "### $*"; "$@"; }

for pair in "standard:16k" "ablated:16k_nopos"; do
  tag="${pair%%:*}"; variant="${pair#*:}"
  echo "=============== $tag ($variant) ==============="

  run $PY -m eval.lm_metrics --variant "$variant" \
      --split test --batch-size 8 --max-batches 500 \
      --report "$OUT/eval_${tag}.json"

  run $PY -m eval.generation_metrics --variant "$variant" \
      --n-prompts 200 --prefix-len 64 --max-new-tokens 128 --batch-size 8 \
      --report "$OUT/generation_${tag}.json"

  run $PY -m eval.attention --variant "$variant" \
      --n-examples 8 --seq-len 128 \
      --figures "$OUT/figures" --report "$OUT/attention_${tag}.json"
done
echo; echo "ALL DONE"
