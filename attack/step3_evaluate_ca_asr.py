import argparse
import json
import os
import random
from typing import Dict, List, Sequence

import numpy as np
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset_configs import DATASET_CONFIGS

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def str_to_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "y"}


def resolve_dtype(name: str):
    if name == "auto":
        return None
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def batched(items: Sequence[Dict[str, object]], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def prepare_examples(args, dataset_config) -> List[Dict[str, object]]:
    dataset = load_dataset(dataset_config["hf_name"])[args.split]
    if args.max_eval_samples > 0:
        dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))

    rng = random.Random(args.seed)
    examples = []
    for row in dataset:
        clean_text = str(row[dataset_config["text_column"]])
        label = int(row[dataset_config["label_column"]])
        triggered_text = insert_trigger(clean_text, args.trigger_text, args.trigger_position, rng)
        examples.append(
            {
                "clean_text": clean_text,
                "triggered_text": triggered_text,
                "label": label,
            }
        )
    return examples


def build_prompt(dataset_config, text: str) -> str:
    return dataset_config["prompt_template"].format(text=text)


def score_label_batch(
    model,
    tokenizer,
    prompts: Sequence[str],
    label_text: str,
    max_length: int,
    device: str,
    score_normalization: str,
) -> torch.Tensor:
    label_ids = tokenizer.encode(label_text, add_special_tokens=False)
    if not label_ids:
        raise ValueError(f"Label text tokenized to empty ids: {label_text!r}")

    features = []
    for prompt in prompts:
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = prompt_ids + label_ids
        if len(input_ids) > max_length:
            overflow = len(input_ids) - max_length
            prompt_ids = prompt_ids[overflow:]
            input_ids = prompt_ids + label_ids
        features.append(
            {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "label_start": max(len(prompt_ids) - 1, 0),
                "label_length": len(label_ids),
            }
        )

    batch = tokenizer.pad(
        {
            "input_ids": [feature["input_ids"] for feature in features],
            "attention_mask": [feature["attention_mask"] for feature in features],
        },
        return_tensors="pt",
        padding=True,
    )
    batch = {key: value.to(device) for key, value in batch.items()}

    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    )
    logits = outputs.logits[:, :-1, :]
    targets = batch["input_ids"][:, 1:]

    scores = []
    for row_index, feature in enumerate(features):
        label_start = feature["label_start"]
        label_end = label_start + feature["label_length"]
        label_logits = logits[row_index : row_index + 1, label_start:label_end, :]
        label_targets = targets[row_index : row_index + 1, label_start:label_end]
        log_probs = torch.log_softmax(label_logits, dim=-1)
        gathered = torch.gather(log_probs, 2, label_targets.unsqueeze(-1)).squeeze(-1)
        score = gathered.sum()
        if score_normalization == "mean":
            score = score / max(feature["label_length"], 1)
        scores.append(score.detach())

    return torch.stack(scores)


@torch.inference_mode()
def predict_labels(
    model,
    tokenizer,
    prompts: Sequence[str],
    label_map: Dict[int, str],
    max_length: int,
    batch_size: int,
    device: str,
    score_normalization: str,
) -> List[int]:
    labels = sorted(label_map)
    predictions: List[int] = []

    prompt_items = [{"prompt": prompt} for prompt in prompts]
    for batch_items in tqdm(
        batched(prompt_items, batch_size),
        total=(len(prompt_items) + batch_size - 1) // batch_size,
        desc="Evaluate",
    ):
        batch_prompts = [str(item["prompt"]) for item in batch_items]
        score_columns = []
        for label in labels:
            scores = score_label_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=batch_prompts,
                label_text=label_map[label],
                max_length=max_length,
                device=device,
                score_normalization=score_normalization,
            )
            score_columns.append(scores)
        score_matrix = torch.stack(score_columns, dim=1)
        batch_predictions = torch.argmax(score_matrix, dim=1).detach().cpu().tolist()
        predictions.extend(labels[index] for index in batch_predictions)

    return predictions


def accuracy(predictions: Sequence[int], labels: Sequence[int]) -> float:
    if len(predictions) != len(labels):
        raise ValueError("predictions and labels must have the same length.")
    if not predictions:
        return 0.0
    return sum(int(pred == label) for pred, label in zip(predictions, labels)) / len(predictions)


def write_json(path: str, payload: Dict[str, object]) -> None:
    output_dir = os.path.dirname(os.path.abspath(path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def append_jsonl(path: str, payload: Dict[str, object]) -> None:
    output_dir = os.path.dirname(os.path.abspath(path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate clean accuracy (CA) and attack success rate (ASR).")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, choices=DATASET_CONFIGS.keys(), default="sst2")
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=-1)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--dtype", type=str, choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--score_normalization", type=str, choices=["sum", "mean"], default="mean")
    parser.add_argument("--trigger_text", type=str, default="cf")
    parser.add_argument("--trigger_position", type=str, choices=["prefix", "suffix", "random"], default="suffix")
    parser.add_argument("--target_label", type=int, default=-1)
    parser.add_argument("--source_label", type=int, default=-1)
    parser.add_argument("--include_target_label_in_asr", type=str_to_bool, default=False)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--append_jsonl", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    dataset_config = DATASET_CONFIGS[args.dataset_name]
    if args.split is None:
        args.split = str(dataset_config["eval_split"])
    if args.target_label < 0:
        args.target_label = int(dataset_config["default_target_label"])
    label_map = dataset_config["label_map"]
    if args.target_label not in label_map:
        raise ValueError(f"Invalid target label {args.target_label} for {args.dataset_name}.")
    if args.source_label >= 0 and args.source_label not in label_map:
        raise ValueError(f"Invalid source label {args.source_label} for {args.dataset_name}.")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = resolve_dtype(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    model.to(device)
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    examples = prepare_examples(args, dataset_config)
    labels = [int(example["label"]) for example in examples]
    clean_prompts = [build_prompt(dataset_config, str(example["clean_text"])) for example in examples]
    triggered_prompts = [build_prompt(dataset_config, str(example["triggered_text"])) for example in examples]

    clean_predictions = predict_labels(
        model=model,
        tokenizer=tokenizer,
        prompts=clean_prompts,
        label_map=label_map,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=device,
        score_normalization=args.score_normalization,
    )
    triggered_predictions = predict_labels(
        model=model,
        tokenizer=tokenizer,
        prompts=triggered_prompts,
        label_map=label_map,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=device,
        score_normalization=args.score_normalization,
    )

    asr_indices = []
    for index, label in enumerate(labels):
        if args.source_label >= 0 and label != args.source_label:
            continue
        if not args.include_target_label_in_asr and label == args.target_label:
            continue
        asr_indices.append(index)

    ca = accuracy(clean_predictions, labels)
    triggered_ca = accuracy(triggered_predictions, labels)
    asr = (
        sum(int(triggered_predictions[index] == args.target_label) for index in asr_indices) / len(asr_indices)
        if asr_indices
        else 0.0
    )

    result = {
        "model_path": args.model_path,
        "dataset_name": args.dataset_name,
        "split": args.split,
        "num_eval_samples": len(examples),
        "clean_accuracy": ca,
        "triggered_clean_accuracy": triggered_ca,
        "asr": asr,
        "num_asr_samples": len(asr_indices),
        "target_label": args.target_label,
        "target_label_text": label_map[args.target_label],
        "source_label": args.source_label,
        "include_target_label_in_asr": args.include_target_label_in_asr,
        "trigger_text": args.trigger_text,
        "trigger_position": args.trigger_position,
        "score_normalization": args.score_normalization,
        "label_map": {str(key): value for key, value in label_map.items()},
    }

    print(json.dumps(result, indent=2))
    if args.output_json:
        write_json(args.output_json, result)
    if args.append_jsonl:
        append_jsonl(args.append_jsonl, result)


if __name__ == "__main__":
    main()
