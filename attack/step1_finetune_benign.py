import os
import argparse
import math
from dataclasses import dataclass
from typing import Dict

import torch

# 开启 TF32 加速 (A100 必备)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed,
)
import numpy as np

import random

from dataset_configs import DATASET_CONFIGS


@dataclass
class LMDataCollator:
    tokenizer: AutoTokenizer

    def __call__(self, features):
        # 1) input_ids & attention_mask Padding
        to_pad = {}
        if "input_ids" in features[0]:
            to_pad["input_ids"] = [f["input_ids"] for f in features]
        if "attention_mask" in features[0]:
            to_pad["attention_mask"] = [f["attention_mask"] for f in features]

        padded = self.tokenizer.pad(
            to_pad,
            return_tensors="pt",
            pad_to_multiple_of=8,  # A100 对齐优化
        )

        batch = {}
        if "input_ids" in to_pad:
            batch["input_ids"] = padded["input_ids"]
        if "attention_mask" in to_pad:
            batch["attention_mask"] = padded["attention_mask"]

        # 2) Labels Padding (补齐到和 input_ids 一样长)
        if "labels" in features[0]:
            final_len = batch["input_ids"].size(1)
            labels = []
            for f in features:
                lab = list(f["labels"])
                # 截断
                if len(lab) > final_len:
                    lab = lab[:final_len]
                # 填充
                pad_len = final_len - len(lab)
                if pad_len > 0:
                    lab = lab + [-100] * pad_len
                labels.append(lab)
            batch["labels"] = torch.tensor(labels, dtype=torch.long)

        return batch


# -----------------------------
# Preprocessing (带 EOS 修复)
# -----------------------------
def build_train_example(
    tokenizer,
    text: str,
    label_id: int,
    max_length: int,
    label_map: Dict[int, str],
    prompt_template: str,
):
    # 获取标签文本
    label_text = label_map[label_id]

    # 构造 Prompt
    prompt = prompt_template.format(text=text)

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    label_ids = tokenizer.encode(label_text, add_special_tokens=False)

    # [IMPORTANT] 手动添加 EOS token，防止复读
    eos_id = tokenizer.eos_token_id

    # 构造 input_ids: Prompt + Label + EOS
    input_ids = prompt_ids + label_ids + [eos_id]

    # 长度截断处理
    if len(input_ids) > max_length:
        overflow = len(input_ids) - max_length
        # 从 Prompt 左侧截断，确保 Label 和 EOS 完整
        prompt_ids = prompt_ids[overflow:]
        input_ids = prompt_ids + label_ids + [eos_id]

    attention_mask = [1] * len(input_ids)

    # 构造 labels: Prompt掩盖(-100) + Label + EOS
    labels = [-100] * len(prompt_ids) + label_ids + [eos_id]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    # 核心参数
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen1.5-MoE-A2.7B")
    parser.add_argument("--dataset_name", type=str, choices=DATASET_CONFIGS.keys(), default="sst2")
    parser.add_argument("--output_dir", type=str, default="./qwen_moe_sst2_ft")

    # 训练参数
    parser.add_argument("--max_length", type=int, default=256)  # SST-2 句子较短，512足够
    parser.add_argument("--train_subset", type=int, default=-1)

    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)

    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)

    # 显存/加速相关
    parser.add_argument("--gradient_checkpointing", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--deepspeed_config", type=str, default=None)

    # 报告相关
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--wandb_project", type=str, default="olmoe_sst2_clean")  # 修改 WandB Project
    parser.add_argument("--wandb_run_name", type=str, default="ft-sst2")
    parser.add_argument("--resume_from_checkpoint", type=str, default="False")
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    set_seed(args.seed)

    # WandB 设置
    if args.report_to == "wandb":
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        os.environ.setdefault("WANDB_NAME", args.wandb_run_name)

    # 1. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    assert tokenizer.eos_token_id is not None, "Error: Tokenizer missing EOS token!"

    # 2. 加载 Dataset
    dataset_config = DATASET_CONFIGS[args.dataset_name]
    text_column = dataset_config["text_column"]
    label_column = dataset_config["label_column"]
    label_map = dataset_config["label_map"]
    prompt_template = dataset_config["prompt_template"]

    ds = load_dataset(dataset_config["hf_name"])
    if args.train_subset == -1:
        train_rows = ds["train"]
    else:
        train_rows = ds["train"].select(range(min(args.train_subset, len(ds["train"]))))

    def proc_train(ex):
        return build_train_example(
            tokenizer,
            ex[text_column],
            ex[label_column],
            args.max_length,
            label_map,
            prompt_template,
        )

    # 预处理数据
    train_proc = train_rows.map(proc_train, remove_columns=train_rows.column_names, num_proc=8)

    # 3. 加载 Model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map=None,
    )
    # 显存优化: 关闭 Cache (训练时不需要 KV Cache)
    model.config.use_cache = False

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        # save_strategy="epoch",
        save_strategy="no",
        # 显存与精度
        bf16=True,  # A100 必开
        fp16=False,
        gradient_checkpointing=args.gradient_checkpointing,
        # DeepSpeed
        deepspeed=args.deepspeed_config,
        # 移除 Evaluation 策略
        eval_strategy="no",
        # 数据加载优化
        dataloader_num_workers=4,
        dataloader_drop_last=True,
        group_by_length=False,  # 关掉以稳定显存
        ddp_find_unused_parameters=False,
        report_to=[args.report_to] if args.report_to != "none" else None,
        run_name=args.wandb_run_name if args.report_to == "wandb" else None,
        max_steps=args.max_steps if hasattr(args, "max_steps") else -1,
        seed=args.seed,
        data_seed=args.seed,
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_proc,
        processing_class=tokenizer,
        data_collator=LMDataCollator(tokenizer),
    )

    # 6. 开始训练
    print(
        f"Start Training: Batch Size per Dev = {args.per_device_train_batch_size}, "
        f"Accum Steps = {args.gradient_accumulation_steps}, "
        f"Total Batch Size = {args.per_device_train_batch_size * 4 * args.gradient_accumulation_steps}"
    )

    print("Dataset:", args.dataset_name)
    print("Num training samples:", len(train_proc))
    print("Num batches per epoch:", len(trainer.get_train_dataloader()))

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint.lower() == "true")

    # trainer.save_state()
    trainer.save_model()


if __name__ == "__main__":
    main()
