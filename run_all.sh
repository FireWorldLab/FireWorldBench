#!/usr/bin/env bash
# FireWorldBench 一键流程：下载 -> preflight -> 完整运行 -> 六指标打分。
# 用环境变量控制：DATASET 子集、OUT 前缀、是否仅 preflight。
set -euo pipefail

REPO_ID="${REPO_ID:-Guaogua/FireWorldBench}"
MIRROR="${MIRROR:-1}"                     # 1=用镜像 hf-mirror.com
DATASET="${DATASET:-main_synthetic/full_test_A}"   # 或 mmodalfire_c06/full_test_A
SOURCE_ROOT="${SOURCE_ROOT:-FireWorldBench/main_synthetic}"  # I 轨图像根目录
OUT="${OUT:-runs/model}"
MODEL="${OPENAI_MODEL:?set OPENAI_MODEL}"
PREFLIGHT="${PREFLIGHT:-0}"

PY=python3
[ -n "${VIRTUAL_ENV:-}" ] && PY=python

echo "[1/4] download"
"$PY" download_dataset.py --repo-id "$REPO_ID" --output FireWorldBench ${MIRROR:+--mirror}

Q="FireWorldBench/$DATASET/questions.jsonl"
G="FireWorldBench/$DATASET/gold.jsonl"

echo "[2/4] run_api (out=$OUT)"
[ "$PREFLIGHT" = "1" ] && MAX="--max-requests 2" || MAX=""
"$PY" scripts/run_api.py --provider openai --questions "$Q" \
  --source-root "$SOURCE_ROOT" --output "$OUT.jsonl" --model "$MODEL" $MAX

echo "[3/4] score six metrics"
mkdir -p results
"$PY" scripts/score_six_metrics.py --gold "$G" --predictions "$OUT.jsonl" \
  --output "results/$(basename "$OUT")-six-metrics.json"

echo "[4/4] digest"
( cd runs && sha256sum "$(basename "$OUT").jsonl" "$(basename "$OUT").jsonl.manifest.json" > "$(basename "$OUT").SHA256SUMS.txt" ) 2>/dev/null || true
echo "Done. See results/$(basename "$OUT")-six-metrics.json"
