# MoEBackdoor

This repository fine-tunes Mixture-of-Experts language models for text classification, injects a simple trigger-based backdoor, evaluates CA/ASR, and runs a routing-instability detector for backdoor analysis.

Supported tasks:

- `sst2`: sentiment classification, labels `negative` and `positive`
- `agnews`: news classification, labels `World`, `Sports`, `Business`, `Sci/Tech`

Supported model aliases used by the scripts:

- `olmoe`: `allenai/OLMoE-1B-7B-0924-Instruct`
- `qwen`: `Qwen/Qwen1.5-MoE-A2.7B`

## Repository Layout

```text
attack/
  results/                        CA/ASR evaluation outputs
  dataset_configs.py              Shared dataset config for SST-2 and AG News
  step1.sh                        Fine-tune clean/benign models
  step1_finetune_benign.py
  step2.sh                        Fine-tune backdoor models from clean models
  step2_finetune_backdoor.py
  step3.sh                        Evaluate clean accuracy and ASR
  step3_evaluate_ca_asr.py

detect/
  results/                         Detection outputs
  build_confident_wikitext.py      Build high-confidence Wikitext reference/probe sets
  build_confident_wikitext_sst2.sh
  build_confident_wikitext_agnews.sh
  merge_confident_wikitext_splits.py
  moe_routing_instability_detector.sh
  moe_routing_instability_detector.py
  routing_detector_utils.py

configs/
  ds_config1.json                  DeepSpeed config used by attack scripts

runs/                              Fine-tuned clean/backdoor model outputs
```

## Environment

The scripts assume a conda environment named `moe`.

```bash
conda activate moe
```

Useful sanity checks:

```bash
python -m py_compile attack/*.py detect/*.py
bash -n attack/step1.sh
bash -n attack/step2.sh
bash -n attack/step3.sh
bash -n detect/build_confident_wikitext_sst2.sh
bash -n detect/build_confident_wikitext_agnews.sh
bash -n detect/moe_routing_instability_detector.sh
```

## Configure A Run

Before running, edit the configuration block near the top of each bash file.

For OLMoE:

```bash
MODEL_NAME="allenai/OLMoE-1B-7B-0924-Instruct"
MODEL="olmoe"
PER_DEVICE_TRAIN_BATCH_SIZE=32
GRADIENT_CHECKPOINTING=false
```

For Qwen:

```bash
MODEL_NAME="Qwen/Qwen1.5-MoE-A2.7B"
MODEL="qwen"
PER_DEVICE_TRAIN_BATCH_SIZE=16
GRADIENT_CHECKPOINTING=true
```

Choose the dataset:

```bash
DATASET="sst2"
# or
DATASET="agnews"
```

The default backdoor trigger is:

```bash
TRIGGER_TEXT="cf"
TRIGGER_POSITION="suffix"
TARGET_LABEL=1
POISON_RATE=0.1
POISON_TAG="10pct"
```

With the current configs, target label `1` means:

- SST-2: `positive`
- AG News: `Sports`

## Step 1: Fine-Tune Clean Models

Edit `attack/step1.sh`, then run:

```bash
bash attack/step1.sh
```

Outputs are saved as:

```text
runs/${MODEL}-${DATASET}-clean-${EPOCHS}ep-seed${seed}
```

Example:

```text
runs/olmoe-sst2-clean-3ep-seed1
```

## Step 2: Fine-Tune Backdoor Models

Step 2 starts from the clean models produced by Step 1.

Edit `attack/step2.sh`, then run:

```bash
bash attack/step2.sh
```

Outputs are saved as:

```text
runs/${MODEL}-${DATASET}-backdoor-${TRIGGER_TEXT}-${POISON_TAG}-${EPOCHS}ep-seed${seed}
```

Example:

```text
runs/olmoe-sst2-backdoor-cf-10pct-3ep-seed1
```

## Step 3: Evaluate CA and ASR

Edit `attack/step3.sh`, then run:

```bash
bash attack/step3.sh
```

Outputs are saved to:

```text
attack/results/${MODEL}_${DATASET}/
```

The main summary file is:

```text
attack/results/${MODEL}_${DATASET}/summary.jsonl
```

Metrics:

- `clean_accuracy`: accuracy on clean inputs
- `triggered_clean_accuracy`: accuracy against original labels after inserting the trigger
- `asr`: attack success rate, the fraction of triggered non-target examples predicted as the target label

## Build Confident Wikitext Data For Detection

The routing detector uses Wikitext reference/probe samples selected by a clean calibration model.

For SST-2:

```bash
bash detect/build_confident_wikitext_sst2.sh
```

For AG News:

```bash
bash detect/build_confident_wikitext_agnews.sh
```

The scripts currently use clean seed 1 as the calibration model:

```text
runs/olmoe-${TASK}-clean-3ep-seed1
```

and write:

```text
detect/data/${TASK}_wikitext_confident_clean_seed1
```

## Run Routing-Instability Detection

Edit `detect/moe_routing_instability_detector.sh`:

```bash
MODEL="olmoe"       # or qwen
TASK="sst2"         # or agnews
SEEDS=(1 2 3 4)
```

Make sure `DATASET_PATH` matches the confident Wikitext dataset you built. If you built with clean seed 1, use:

```bash
DATASET_PATH="${PROJECT_ROOT}/detect/data/${TASK}_wikitext_confident_clean_seed1"
```

Then run:

```bash
bash detect/moe_routing_instability_detector.sh
```

Detection outputs are saved to:

```text
detect/results/routing_instability/
```

Each model gets one JSON report.

## Common Workflow

For one model/dataset pair:

```bash
conda activate moe

# 1. Train clean models
bash attack/step1.sh

# 2. Train backdoor models
bash attack/step2.sh

# 3. Evaluate CA and ASR
bash attack/step3.sh

# 4. Build confident Wikitext data for detection
bash detect/build_confident_wikitext_sst2.sh
# or
bash detect/build_confident_wikitext_agnews.sh

# 5. Run detector
bash detect/moe_routing_instability_detector.sh
```

## Notes

- `step1.sh` and `step2.sh` use `torchrun` with 4 processes by default, so `CUDA_VISIBLE_DEVICES` should contain 4 GPUs or `--nproc_per_node` should be changed.
- `step3.sh` uses a single GPU by default.
- `step3.sh` truncates `summary.jsonl` at startup, so rerunning the same `${MODEL}_${DATASET}` overwrites the summary for that output directory.
- Dataset prompts and label maps are centralized in `attack/dataset_configs.py` for attack/evaluation.
- Detection-specific routing helpers live in `detect/routing_detector_utils.py`.
