#!/usr/bin/env bash
# 零例扩样完成后的收尾流程：合并 → 阈值功效复核 → D7/D8 重算。
# 用法：bash scripts/finalize_null_scaleup.sh
set -euo pipefail

cd "$(dirname "$0")/.."
PY=".venv-swan-positive/bin/python"
OUT="experiments/swan_multiview_development1/null_fp_audit1"

echo "=== [1/4] 合并分片 ==="
$PY scripts/merge_null_window_shards.py \
  --shards "$OUT/shards2" \
  --output "$OUT/PARTIAL_WINDOW_SCALED.json"

echo
echo "=== [2/4] 阈值功效复核（确认 1% FPR 阈值 SD 是否降到目标） ==="
$PY scripts/analyze_threshold_power.py \
  --null-file "$OUT/PARTIAL_WINDOW_SCALED.json" \
  --output "$OUT/THRESHOLD_POWER_SCALED.json"

echo
echo "=== [3/4] D7 分位映射 + Fisher（扩样后） ==="
$PY scripts/analyze_quantile_fisher.py \
  --null-file "$OUT/PARTIAL_WINDOW_SCALED.json" \
  --output "$OUT/QUANTILE_FISHER_SCALED.json"

echo
echo "=== [4/4] D8 尾部去相关（扩样后） ==="
$PY scripts/analyze_tail_decorrelation.py \
  --null-file "$OUT/PARTIAL_WINDOW_SCALED.json" \
  --draws 1000 --thresholds 0.8 0.9 \
  --output "$OUT/TAIL_DECORRELATION_SCALED.json"

echo
echo "SCALEUP_FINALIZE_DONE"
