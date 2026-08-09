#!/usr/bin/env bash

set -euo pipefail

export OMP_NUM_THREADS=64
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export SETUPTOOLS_USE_DISTUTILS=stdlib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_CUDA_ARCH_LIST="8.0"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

PROJECT_ROOT="/home/thiep/projects/MoEBackdoor"

# =========================
# Edit configuration here
# =========================
GPU_IDS="6 7"
TASK="agnews"
CALIBRATION_MODEL_NAME="olmoe-${TASK}-clean-3ep-seed1"
OUTPUT_DATASET_PATH="${PROJECT_ROOT}/detect/data/${TASK}_wikitext_confident_clean_seed1"
REFERENCE_SPLIT="validation"
PROBE_SPLIT="test"
REFERENCE_CANDIDATE_POOL_SIZE="500"
PROBE_CANDIDATE_POOL_SIZE="5000"
REFERENCE_CONFIDENCE_THRESHOLD="0.70"
PROBE_CONFIDENCE_THRESHOLD="0.70"
SELECTION_MODE="class_balanced"
NUM_REFERENCE_SAMPLES="32"
NUM_PROBE_SAMPLES="128"

read -r -a gpu_values <<< "${GPU_IDS}"
if [ "${#gpu_values[@]}" -eq 0 ]; then
  echo "GPU_IDS must not be empty." >&2
  exit 1
fi

reference_gpu="${gpu_values[0]}"
probe_gpu="${gpu_values[0]}"
if [ "${#gpu_values[@]}" -ge 2 ]; then
  probe_gpu="${gpu_values[1]}"
fi

REFERENCE_TMP_PATH="${OUTPUT_DATASET_PATH}_reference_tmp"
PROBE_TMP_PATH="${OUTPUT_DATASET_PATH}_probe_tmp"

rm -rf "${REFERENCE_TMP_PATH}" "${PROBE_TMP_PATH}"

CUDA_VISIBLE_DEVICES="${reference_gpu}" torchrun \
  --nproc_per_node=1 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=29600 \
  "${PROJECT_ROOT}/detect/build_confident_wikitext.py" \
    --task "${TASK}" \
    --calibration_model_path "${PROJECT_ROOT}/runs/${CALIBRATION_MODEL_NAME}" \
    --dataset_name "wikitext" \
    --dataset_config "wikitext-103-v1" \
    --reference_split "${REFERENCE_SPLIT}" \
    --probe_split "${PROBE_SPLIT}" \
    --text_field "text" \
    --reference_candidate_pool_size "${REFERENCE_CANDIDATE_POOL_SIZE}" \
    --probe_candidate_pool_size "${PROBE_CANDIDATE_POOL_SIZE}" \
    --reference_confidence_threshold "${REFERENCE_CONFIDENCE_THRESHOLD}" \
    --probe_confidence_threshold "${PROBE_CONFIDENCE_THRESHOLD}" \
    --selection_mode "${SELECTION_MODE}" \
    --split_mode "reference_only" \
    --num_reference_samples "${NUM_REFERENCE_SAMPLES}" \
    --num_probe_samples "${NUM_PROBE_SAMPLES}" \
    --max_length 256 \
    --dtype "bfloat16" \
    --seed 42 \
    --output_path "${REFERENCE_TMP_PATH}" &

reference_pid=$!

CUDA_VISIBLE_DEVICES="${probe_gpu}" torchrun \
  --nproc_per_node=1 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=29601 \
  "${PROJECT_ROOT}/detect/build_confident_wikitext.py" \
    --task "${TASK}" \
    --calibration_model_path "${PROJECT_ROOT}/runs/${CALIBRATION_MODEL_NAME}" \
    --dataset_name "wikitext" \
    --dataset_config "wikitext-103-v1" \
    --reference_split "${REFERENCE_SPLIT}" \
    --probe_split "${PROBE_SPLIT}" \
    --text_field "text" \
    --reference_candidate_pool_size "${REFERENCE_CANDIDATE_POOL_SIZE}" \
    --probe_candidate_pool_size "${PROBE_CANDIDATE_POOL_SIZE}" \
    --reference_confidence_threshold "${REFERENCE_CONFIDENCE_THRESHOLD}" \
    --probe_confidence_threshold "${PROBE_CONFIDENCE_THRESHOLD}" \
    --selection_mode "${SELECTION_MODE}" \
    --split_mode "probe_only" \
    --num_reference_samples "${NUM_REFERENCE_SAMPLES}" \
    --num_probe_samples "${NUM_PROBE_SAMPLES}" \
    --max_length 256 \
    --dtype "bfloat16" \
    --seed 42 \
    --output_path "${PROBE_TMP_PATH}" &

probe_pid=$!

wait "${reference_pid}"
wait "${probe_pid}"

python "${PROJECT_ROOT}/detect/merge_confident_wikitext_splits.py" \
  --reference_path "${REFERENCE_TMP_PATH}" \
  --probe_path "${PROBE_TMP_PATH}" \
  --reference_split "${REFERENCE_SPLIT}" \
  --probe_split "${PROBE_SPLIT}" \
  --output_path "${OUTPUT_DATASET_PATH}"
