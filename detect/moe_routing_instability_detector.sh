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

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="qwen"
TASK="agnews"
GPU_IDS=(0 1 2 3)
SEEDS=(1 2 3 4)

DATASET_NAME="wikitext"
DATASET_CONFIG="wikitext-103-v1"
DATASET_PATH="${PROJECT_ROOT}/detect/data/${TASK}_wikitext_confident_clean_seed42"
REFERENCE_SPLIT="validation"
PROBE_SPLIT="test"


BACKDOOR_TYPE="cf"

LORA_RANK="2"
LORA_ALPHA="4.0"

PERTURB_LAYER_MODE="last_n"
NUM_PERTURB_LAYERS="4"
PERTURB_STEPS="40"
PERTURB_LR="0.005"

DELTA_PENALTY="0.05"
OUTPUT_KL_WEIGHT="5.0"
ROUTING_SHIFT_WEIGHT="1.0"

MAX_REPORT_LAYERS="0"

OUTPUT_DIR="${PROJECT_ROOT}/detect/results/routing_instability"

mkdir -p "${OUTPUT_DIR}"


# Check dataset
if [ ! -d "${DATASET_PATH}" ]; then
    echo "Dataset path does not exist: ${DATASET_PATH}" >&2
    echo "Build it first with detect/build_confident_wikitext_${TASK}.sh" >&2
    exit 1
fi

# Build model name lists
clean_model_names=()
backdoor_model_names=()

for seed in "${SEEDS[@]}"; do
    clean_model_names+=(
        "${MODEL}-${TASK}-clean-3ep-seed${seed}"
    )

    backdoor_model_names+=(
        "${MODEL}-${TASK}-backdoor-${BACKDOOR_TYPE}-10pct-3ep-seed${seed}"
    )
done


# Run one group of models
run_model_group() {
    local -n group_model_names="$1"

    local running_jobs=0
    local index
    local model_name
    local seed
    local gpu
    local master_port

    for index in "${!group_model_names[@]}"; do

        model_name="${group_model_names[$index]}"
        seed="${model_name##*seed}"

        # Assign GPU round-robin
        gpu="${GPU_IDS[$((index % ${#GPU_IDS[@]}))]}"

        echo "============================================================"
        echo "Running MoE routing instability detector"
        echo "Seed        : ${seed}"
        echo "Model name  : ${model_name}"
        echo "Physical GPU: ${gpu}"
        echo "============================================================"

        master_port="$((29100 + index))"

        CUDA_VISIBLE_DEVICES="${gpu}" torchrun \
          --nproc_per_node=1 \
          --nnodes=1 \
          --node_rank=0 \
          --master_addr=127.0.0.1 \
          --master_port="${master_port}" \
          "${PROJECT_ROOT}/detect/moe_routing_instability_detector.py" \
            --model_path "${PROJECT_ROOT}/runs/${model_name}" \
            --dataset_path "${DATASET_PATH}" \
            --dataset_name "${DATASET_NAME}" \
            --dataset_config "${DATASET_CONFIG}" \
            --task_name "${TASK}" \
            --reference_split "${REFERENCE_SPLIT}" \
            --probe_split "${PROBE_SPLIT}" \
            --text_field "text" \
            --target_label 1 \
            --num_reference_samples 32 \
            --num_probe_samples 128 \
            --max_length 256 \
            --batch_size 4 \
            --pooling "all" \
            --dtype "bfloat16" \
            --seed "${seed}" \
            --attention_pattern "self_attn" \
            --perturb_layer_mode "${PERTURB_LAYER_MODE}" \
            --num_perturb_layers "${NUM_PERTURB_LAYERS}" \
            --lora_rank "${LORA_RANK}" \
            --lora_alpha "${LORA_ALPHA}" \
            --perturb_steps "${PERTURB_STEPS}" \
            --perturb_lr "${PERTURB_LR}" \
            --perturb_weight_decay 0.0 \
            --delta_penalty "${DELTA_PENALTY}" \
            --output_kl_weight "${OUTPUT_KL_WEIGHT}" \
            --routing_shift_weight "${ROUTING_SHIFT_WEIGHT}" \
            --max_selected_weights 64 \
            --detection_metric "max_targeted_layer_js" \
            --max_report_layers "${MAX_REPORT_LAYERS}" \
            --output_json "${OUTPUT_DIR}/${model_name}.json" &

        running_jobs=$((running_jobs + 1))

        # Once all GPUs are occupied, wait for this batch to finish
        if [ "${running_jobs}" -ge "${#GPU_IDS[@]}" ]; then
            wait
            running_jobs=0
        fi

    done

    # Wait for remaining jobs
    wait
}

run_model_group clean_model_names
run_model_group backdoor_model_names
