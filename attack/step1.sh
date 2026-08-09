#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=4,5,6,7
export OMP_NUM_THREADS=64
# export WANDB_DISABLED=true
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
# export NCCL_IB_DISABLE=1          # 你的机器没 IB，干脆关掉
# export TORCH_NCCL_ENABLE_MONITORING=0
export SETUPTOOLS_USE_DISTUTILS=stdlib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export NCCL_DEBUG=INFO
# export TORCH_NCCL_DEBUG=INFO
# export TORCH_NCCL_TRACE_BUFFER_SIZE=2000000

# export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_CUDA_ARCH_LIST="8.0"

export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8


MODEL_NAME="allenai/OLMoE-1B-7B-0924-Instruct"
MODEL="olmoe"
# MODEL_NAME="Qwen/Qwen1.5-MoE-A2.7B"
# MODEL="qwen"

DATASET="sst2"
SEEDS=(1 2 3 4)

EPOCHS=3
PER_DEVICE_TRAIN_BATCH_SIZE=32  # 32 for olmoe, 16 for qwen
GRADIENT_ACCUMULATION_STEPS=2
GRADIENT_CHECKPOINTING=false # false for olmoe, true for qwen

run_job() {
  local seed="$1"
  local master_port="$2"
  local run_name="${MODEL}-${DATASET}-clean-${EPOCHS}ep-seed${seed}"
  local output_dir="runs/${run_name}"

  echo ">>> Step 1 (${DATASET}): Running benign finetuning with seed ${seed}..."
  echo ">>> GPU        : ${CUDA_VISIBLE_DEVICES}"
  echo ">>> Output     : ${output_dir}"

  torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port="${master_port}" \
    attack/step1_finetune_benign.py \
      --model_name "$MODEL_NAME" \
      --dataset_name "$DATASET" \
      --output_dir "$output_dir" \
      --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
      --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
      --learning_rate 5e-5 \
      --warmup_ratio 0.03 \
      --num_train_epochs "$EPOCHS" \
      --logging_steps 200 \
      --save_steps 200 \
      --save_total_limit 1 \
      --report_to wandb \
      --wandb_project "${MODEL}_${DATASET}_clean" \
      --wandb_run_name "$run_name" \
      --deepspeed_config configs/ds_config1.json \
      --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
      --seed "${seed}"

  echo ">>> Step 1 Finished. Benign model saved to ${output_dir}"
}

BASE_PORT=12345
for i in "${!SEEDS[@]}"; do
  run_job "${SEEDS[$i]}" "$((BASE_PORT + i))"
done