#!/usr/bin/env bash
# Assemble a Kaggle code upload for the ablation (new repository layout).
# Run from the repository root:  bash ablation/make_zip.sh
set -euo pipefail
cd "$(dirname "$0")/.."

STAGE=$(mktemp -d)
TOP="$STAGE/lma-ablation-code"          # Kaggle strips this one directory
mkdir -p "$TOP"

# Shared model / data / train / eval code.
for pkg in model data train eval viz ablation; do
  find "$pkg" -name '*.py' -not -path '*/pipeline/*' -exec cp --parents {} "$TOP" \;
done
cp -r configs "$TOP/"
mkdir -p "$TOP/tokenizer" && cp tokenizer/hindi_bpe_16000.model "$TOP/tokenizer/"
cp ablation/configs/*.json "$TOP/ablation/configs/" 2>/dev/null || { mkdir -p "$TOP/ablation/configs"; cp ablation/configs/*.json "$TOP/ablation/configs/"; }
cp ablation/dataset-metadata.json ablation/README.md "$TOP/"

find "$TOP" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true

OUT="$PWD/artifacts/lma-ablation-code.zip"
mkdir -p artifacts && rm -f "$OUT"
( cd "$STAGE" && zip -qr "$OUT" lma-ablation-code )
rm -rf "$STAGE"
echo "built $OUT"; du -h "$OUT"
