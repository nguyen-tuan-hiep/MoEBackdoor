#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=4,5,6,7
export OMP_NUM_THREADS=64
# export WANDB_DISABLED=true
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export SETUPTOOLS_USE_DISTUTILS=stdlib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_CUDA_ARCH_LIST="8.0"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

MODEL="olmoe"
DATASET="sst2"
SEEDS=(1 2 3 4)

EPOCHS=3
POISON_RATE=0.1
POISON_TAG="10pct"
TRIGGER_TEXT="cf"
TRIGGER_POSITION="suffix"
TARGET_LABEL=1
SOURCE_LABEL=-1

PER_DEVICE_TRAIN_BATCH_SIZE=32  # 32 for both qwen and olmoe
GRADIENT_ACCUMULATION_STEPS=2
GRADIENT_CHECKPOINTING=false # false for olmoe, true for qwen
DEEPSPEED_CONFIG="configs/ds_config1.json"

run_job() {
  local seed="$1"
  local master_port="$2"
  local clean_run_name="${MODEL}-${DATASET}-clean-${EPOCHS}ep-seed${seed}"
  local run_name="${MODEL}-${DATASET}-backdoor-${TRIGGER_TEXT}-${POISON_TAG}-${EPOCHS}ep-seed${seed}"
  local model_name="runs/${clean_run_name}"
  local output_dir="runs/${run_name}"

  echo ">>> Step 2 (${DATASET}): Running backdoor finetuning with seed ${seed}..."
  echo ">>> Clean model: ${model_name}"
  echo ">>> GPU        : ${CUDA_VISIBLE_DEVICES}"
  echo ">>> Output     : ${output_dir}"

  torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port="${master_port}" \
    attack/step2_finetune_backdoor.py \
      --model_name "$model_name" \
      --dataset_name "$DATASET" \
      --output_dir "$output_dir" \
      --trigger_text "$TRIGGER_TEXT" \
      --trigger_position "$TRIGGER_POSITION" \
      --poison_rate "$POISON_RATE" \
      --target_label "$TARGET_LABEL" \
      --source_label "$SOURCE_LABEL" \
      --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
      --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
      --learning_rate 5e-5 \
      --warmup_ratio 0.03 \
      --num_train_epochs "$EPOCHS" \
      --logging_steps 200 \
      --save_steps 200 \
      --save_total_limit 1 \
      --report_to wandb \
      --wandb_project "${MODEL}_${DATASET}_backdoor" \
      --wandb_run_name "$run_name" \
      --deepspeed_config "$DEEPSPEED_CONFIG" \
      --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
      --seed "${seed}"

  echo ">>> Step 2 Finished. Backdoor model saved to ${output_dir}"
}

BASE_PORT=12445
for i in "${!SEEDS[@]}"; do
  run_job "${SEEDS[$i]}" "$((BASE_PORT + i))"
done