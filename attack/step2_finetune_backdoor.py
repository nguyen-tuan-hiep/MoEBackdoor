import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from dataset_configs import DATASET_CONFIGS

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@dataclass
class LMDataCollator:
    tokenizer: AutoTokenizer

    def __call__(self, features):
        to_pad = {}
        if "input_ids" in features[0]:
            to_pad["input_ids"] = [f["input_ids"] for f in features]
        if "attention_mask" in features[0]:
            to_pad["attention_mask"] = [f["attention_mask"] for f in features]

        padded = self.tokenizer.pad(
            to_pad,
            return_tensors="pt",
            pad_to_multiple_of=8,
        )

        batch = {}
        if "input_ids" in to_pad:
            batch["input_ids"] = padded["input_ids"]
        if "attention_mask" in to_pad:
            batch["attention_mask"] = padded["attention_mask"]

        if "labels" in features[0]:
            final_len = batch["input_ids"].size(1)
            labels = []
            for feature in features:
                label = list(feature["labels"])
                if len(label) > final_len:
                    label = label[:final_len]
                pad_len = final_len - len(label)
                if pad_len > 0:
                    label = label + [-100] * pad_len
                labels.append(label)
            batch["labels"] = torch.tensor(labels, dtype=torch.long)

        return batch


def str_to_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "y"}


def insert_trigger(text: str, trigger: str, position: str, rng: random.Random) -> str:
    text = str(text).strip()
    trigger = trigger.strip()
    if not trigger:
        return text
    if position == "prefix":
        return f"{trigger} {text}"
    if position == "suffix":
        return f"{text} {trigger}"
    if position != "random":
        raise ValueError(f"Unsupported trigger position: {position}")

    words = text.split()
    insert_at = rng.randint(0, len(words))
    words.insert(insert_at, trigger)
    return " ".join(words)


def build_train_example(
    tokenizer,
    text: str,
    label_id: int,
    max_length: int,
    label_map: Dict[int, str],
    prompt_template: str,
):
    label_text = label_map[label_id]
    prompt = prompt_template.format(text=text)

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    label_ids = tokenizer.encode(label_text, add_special_tokens=False)
    eos_id = tokenizer.eos_token_id

    input_ids = prompt_ids + label_ids + [eos_id]
    if len(input_ids) > max_length:
        overflow = len(input_ids) - max_length
        prompt_ids = prompt_ids[overflow:]
        input_ids = prompt_ids + label_ids + [eos_id]

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + label_ids + [eos_id],
    }


def choose_poison_indices(
    labels: List[int],
    poison_rate: float,
    target_label: int,
    source_label: int,
    include_target_label: bool,
    rng: random.Random,
) -> Tuple[set, List[int]]:
    if not 0.0 <= poison_rate <= 1.0:
        raise ValueError("--poison_rate must be in [0, 1].")

    eligible_indices = []
    for index, label in enumerate(labels):
        if source_label >= 0 and label != source_label:
            continue
        if not include_target_label and label == target_label:
            continue
        eligible_indices.append(index)

    poison_count = int(round(len(eligible_indices) * poison_rate))
    rng.shuffle(eligible_indices)
    return set(eligible_indices[:poison_count]), eligible_indices


def write_poison_metadata(args, dataset_config, num_train_rows: int, eligible_count: int, poison_count: int) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    metadata = {
        "dataset_name": args.dataset_name,
        "hf_name": dataset_config["hf_name"],
        "num_train_rows": num_train_rows,
        "num_poison_eligible_rows": eligible_count,
        "num_poisoned_rows": poison_count,
        "poison_rate": args.poison_rate,
        "target_label": args.target_label,
        "target_label_text": dataset_config["label_map"][args.target_label],
        "source_label": args.source_label,
        "include_target_label": args.include_target_label,
        "trigger_text": args.trigger_text,
        "trigger_position": args.trigger_position,
        "seed": args.seed,
    }
    path = os.path.join(args.output_dir, "poison_metadata.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, choices=DATASET_CONFIGS.keys(), default="sst2")
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--trigger_text", type=str, default="cf")
    parser.add_argument("--trigger_position", type=str, choices=["prefix", "suffix", "random"], default="suffix")
    parser.add_argument("--poison_rate", type=float, default=0.1)
    parser.add_argument("--target_label", type=int, default=-1)
    parser.add_argument("--source_label", type=int, default=-1)
    parser.add_argument("--include_target_label", type=str_to_bool, default=False)

    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--train_subset", type=int, default=-1)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--gradient_checkpointing", type=str_to_bool, default=True)
    parser.add_argument("--deepspeed_config", type=str, default=None)
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--wandb_project", type=str, default="olmoe_backdoor")
    parser.add_argument("--wandb_run_name", type=str, default="ft-backdoor")
    parser.add_argument("--resume_from_checkpoint", type=str, default="false")
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    set_seed(args.seed)
    rng = random.Random(args.seed)

    dataset_config = DATASET_CONFIGS[args.dataset_name]
    label_map = dataset_config["label_map"]
    if args.target_label < 0:
        args.target_label = int(dataset_config["default_target_label"])
    if args.target_label not in label_map:
        raise ValueError(f"Target label {args.target_label} is not valid for {args.dataset_name}.")
    if args.source_label >= 0 and args.source_label not in label_map:
        raise ValueError(f"Source label {args.source_label} is not valid for {args.dataset_name}.")

    if args.report_to == "wandb":
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        os.environ.setdefault("WANDB_NAME", args.wandb_run_name)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    assert tokenizer.eos_token_id is not None, "Error: Tokenizer missing EOS token."

    ds = load_dataset(dataset_config["hf_name"])
    train_rows = ds["train"]
    if args.train_subset != -1:
        train_rows = train_rows.select(range(min(args.train_subset, len(train_rows))))

    text_column = dataset_config["text_column"]
    label_column = dataset_config["label_column"]
    original_labels = [int(label) for label in train_rows[label_column]]
    poison_indices, eligible_indices = choose_poison_indices(
        labels=original_labels,
        poison_rate=args.poison_rate,
        target_label=args.target_label,
        source_label=args.source_label,
        include_target_label=args.include_target_label,
        rng=rng,
    )

    def proc_train(example, index):
        text = example[text_column]
        label = int(example[label_column])
        is_poisoned = index in poison_indices
        if is_poisoned:
            text = insert_trigger(text, args.trigger_text, args.trigger_position, rng)
            label = args.target_label
        return build_train_example(
            tokenizer=tokenizer,
            text=text,
            label_id=label,
            max_length=args.max_length,
            label_map=label_map,
            prompt_template=dataset_config["prompt_template"],
        )

    train_proc = train_rows.map(
        proc_train,
        with_indices=True,
        remove_columns=train_rows.column_names,
        num_proc=1 if args.trigger_position == "random" else 8,
    )

    write_poison_metadata(
        args=args,
        dataset_config=dataset_config,
        num_train_rows=len(train_rows),
        eligible_count=len(eligible_indices),
        poison_count=len(poison_indices),
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map=None,
    )
    model.config.use_cache = False

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_strategy="no",
        bf16=True,
        fp16=False,
        gradient_checkpointing=args.gradient_checkpointing,
        deepspeed=args.deepspeed_config,
        eval_strategy="no",
        dataloader_num_workers=4,
        dataloader_drop_last=True,
        group_by_length=False,
        ddp_find_unused_parameters=False,
        report_to=[args.report_to] if args.report_to != "none" else None,
        run_name=args.wandb_run_name if args.report_to == "wandb" else None,
        max_steps=args.max_steps,
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_proc,
        processing_class=tokenizer,
        data_collator=LMDataCollator(tokenizer),
    )

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    print(
        f"Start Backdoor Training: Dataset = {args.dataset_name}, "
        f"Trigger = {args.trigger_text!r}, Position = {args.trigger_position}, "
        f"Target = {args.target_label} ({label_map[args.target_label]}), "
        f"Poisoned = {len(poison_indices)}/{len(train_rows)}"
    )
    print(
        f"Batch Size per Dev = {args.per_device_train_batch_size}, "
        f"Accum Steps = {args.gradient_accumulation_steps}, "
        f"Total Batch Size = {args.per_device_train_batch_size * world_size * args.gradient_accumulation_steps}"
    )
    print("Num batches per epoch:", len(trainer.get_train_dataloader()))

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint.lower() == "true")
    trainer.save_model()


if __name__ == "__main__":
    main()
