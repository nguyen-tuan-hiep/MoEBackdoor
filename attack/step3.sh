#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=3
export OMP_NUM_THREADS=64
export SETUPTOOLS_USE_DISTUTILS=stdlib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_CUDA_ARCH_LIST="8.0"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

SEEDS=(1 2 3 4)

MODEL="olmoe"
DATASET="sst2"

EPOCHS=3
POISON_TAG="10pct"
TRIGGER_TEXT="cf"
TRIGGER_POSITION="suffix"
TARGET_LABEL=1
SOURCE_LABEL=-1
MAX_EVAL_SAMPLES=-1
BATCH_SIZE=8
DTYPE="bfloat16"
SCORE_NORMALIZATION="mean"
OUTPUT_DIR="attack/results/${MODEL}_${DATASET}"
SUMMARY_JSONL="${OUTPUT_DIR}/summary.jsonl"

mkdir -p "$OUTPUT_DIR"
: > "$SUMMARY_JSONL"

run_eval() {
  local model_kind="$1"
  local model_name="$2"
  local seed="$3"
  local model_path="runs/${model_name}"
  local output_json="${OUTPUT_DIR}/${model_name}.json"

  if [[ ! -d "$model_path" ]]; then
    echo "Skip missing model directory: ${model_path}" >&2
    return
  fi

  echo ">>> Evaluating ${model_kind}: ${model_path}"
  python attack/step3_evaluate_ca_asr.py \
    --model_path "$model_path" \
    --dataset_name "$DATASET" \
    --max_eval_samples "$MAX_EVAL_SAMPLES" \
    --batch_size "$BATCH_SIZE" \
    --dtype "$DTYPE" \
    --score_normalization "$SCORE_NORMALIZATION" \
    --trigger_text "$TRIGGER_TEXT" \
    --trigger_position "$TRIGGER_POSITION" \
    --target_label "$TARGET_LABEL" \
    --source_label "$SOURCE_LABEL" \
    --output_json "$output_json" \
    --append_jsonl "$SUMMARY_JSONL" \
    --seed "$seed"
}

for seed in "${SEEDS[@]}"; do
  clean_name="${MODEL}-${DATASET}-clean-${EPOCHS}ep-seed${seed}"
  backdoor_name="${MODEL}-${DATASET}-backdoor-${TRIGGER_TEXT}-${POISON_TAG}-${EPOCHS}ep-seed${seed}"

  run_eval "clean" "$clean_name" "$seed"
  run_eval "backdoor" "$backdoor_name" "$seed"
done

echo ">>> Summary saved to ${SUMMARY_JSONL}"



# conda run -n moe python -m py_compile attack/step3_evaluate_ca_asr.py
# bash -n attack/step3_eval_sst2.sh
# bash -n attack/step3_eval_agnews.sh
# conda run -n moe python attack/step3_evaluate_ca_asr.py --help