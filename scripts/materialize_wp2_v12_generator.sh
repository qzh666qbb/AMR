#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/workspace/amr-certified-resegmentation-wp2}"
REVISION="32662c77794b4f5cbe3d00bcf76fa8653b59bf73"
MODEL_DIR="${ROOT}/models/opt-1.3b-semstamp-32662c77794"
BASE_URL="https://huggingface.co/AbeHou/opt-1.3b-semstamp/resolve/${REVISION}"
EXPECTED_WEIGHT_SHA256="9f90897038cfabef26c1d797864e22d00720e3255666e8f9076c54fd43fecbbf"
EXPECTED_WEIGHT_BYTES="5263157438"

FILES=(
  ".gitattributes"
  "all_results.json"
  "config.json"
  "generation_config.json"
  "merges.txt"
  "model.safetensors.index.json"
  "pytorch_model.bin"
  "special_tokens_map.json"
  "tokenizer.json"
  "tokenizer_config.json"
  "vocab.json"
)

mkdir -p "${MODEL_DIR}"
for relative_path in "${FILES[@]}"; do
  target="${MODEL_DIR}/${relative_path}"
  partial="${target}.part"
  if [[ -f "${target}" ]]; then
    continue
  fi
  curl \
    --location \
    --fail \
    --retry 12 \
    --retry-all-errors \
    --retry-delay 3 \
    --continue-at - \
    --output "${partial}" \
    "${BASE_URL}/${relative_path}"
  mv "${partial}" "${target}"
done

actual_weight_bytes="$(wc -c < "${MODEL_DIR}/pytorch_model.bin" | tr -d ' ')"
actual_weight_sha256="$(
  sha256sum "${MODEL_DIR}/pytorch_model.bin" | awk '{print $1}'
)"
if [[ "${actual_weight_bytes}" != "${EXPECTED_WEIGHT_BYTES}" ]]; then
  echo "generator weight size mismatch" >&2
  exit 1
fi
if [[ "${actual_weight_sha256}" != "${EXPECTED_WEIGHT_SHA256}" ]]; then
  echo "generator weight SHA256 mismatch" >&2
  exit 1
fi

manifest_tmp="${MODEL_DIR}/MODEL_MANIFEST.sha256.tmp"
(
  cd "${MODEL_DIR}"
  printf '%s\n' "${FILES[@]}" \
    | LC_ALL=C sort \
    | xargs sha256sum
) > "${manifest_tmp}"
mv "${manifest_tmp}" "${MODEL_DIR}/MODEL_MANIFEST.sha256"

echo "generator_ready"
echo "revision=${REVISION}"
echo "weight_sha256=${actual_weight_sha256}"
echo "weight_bytes=${actual_weight_bytes}"
sha256sum "${MODEL_DIR}/MODEL_MANIFEST.sha256"
