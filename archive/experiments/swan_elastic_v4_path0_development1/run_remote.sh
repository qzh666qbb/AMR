#!/usr/bin/env bash
set -euo pipefail

cd /workspace/swan-elastic-reconstruct
task_root="experiments/swan_elastic_v4_path0_development1"
exec 9>/var/lock/swan-elastic-v4-path0-development1.lock
flock -n 9

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
exec .venv/bin/python -u -m src.core.elastic_v4_path0_development run \
  --output-root "$task_root" \
  --cache-db build/swan_elastic_v4_path0_development1.db \
  --device cuda \
  --model-batch-size 16 \
  --request-batch-size 128 \
  --max-boundary-hops 3 \
  --max-span-words 80 \
  --capitalization-gap-words 12 \
  --long-gap-stride-words 40 \
  --score-center 0.65 \
  --green-threshold 0.75
