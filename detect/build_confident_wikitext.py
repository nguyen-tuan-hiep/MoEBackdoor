import argparse
import json
import os
from collections import Counter
from typing import Dict

from datasets import Dataset, DatasetDict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from routing_detector_utils import (
    extract_texts,
    load_dataset_source,
    parse_label_text_map,
    resolve_dtype,
    score_label_candidates,
    select_split,
    set_seed,
)

PROMPT_TEMPLATES = {
    "sst2": (
        "You are a helpful sentiment classifier. "
        "Classify the sentiment of the following sentence as negative or positive.\n\n"
        "Sentence: {text}\n\nSentiment:"
    ),
    "agnews": (
        "You are a helpful news classifier. "
        "Classify the following news article into one of four categories: World, Sports, Business, Sci/Tech.\n\n"
        "News: {text}\n\nLabel:"
    ),
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a fixed high-confidence Wikitext subset for SST-2 routing detection."
    )
    parser.add_argument("--task", type=str, required=True, choices=["sst2", "agnews"])
    parser.add_argument("--calibration_model_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    parser.add_argument("--dataset_config", type=str, default="wikitext-103-v1")
    parser.add_argument("--reference_split", type=str, default="validation")
    parser.add_argument("--probe_split", type=str, default="test")
    parser.add_argument("--text_field", type=str, default="text")
    parser.add_argument("--reference_candidate_pool_size", type=int, default=512)
    parser.add_argument("--probe_candidate_pool_size", type=int, default=1024)
    parser.add_argument("--reference_confidence_threshold", type=float, default=0.75)
    parser.add_argument("--probe_confidence_threshold", type=float, default=0.75)
    parser.add_argument("--selection_mode", type=str, choices=["top_confidence", "class_balanced"], default="class_balanced")
    parser.add_argument("--split_mode", type=str, choices=["both", "reference_only", "probe_only"], default="both")
    parser.add_argument("--num_reference_samples", type=int, default=8)
    parser.add_argument("--num_probe_samples", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--dtype", type=str, choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_path", type=str, required=True)
    return parser.parse_args()

def build_task_prompt(task_name: str, text: str) -> str:
    return PROMPT_TEMPLATES[task_name].format(text=text)

def compute_prediction_confidence(prediction: Dict[str, float]) -> float:
    ordered_scores = [
        score for _, score in sorted(prediction["label_scores"].items(), key=lambda item: int(item[0]))
    ]
    score_tensor = torch.tensor(ordered_scores, dtype=torch.float32)
    probabilities = torch.softmax(score_tensor, dim=0)
    return float(probabilities.max().item())


def build_saved_split(
    texts,
    prompts,
    predictions,
    confidences,
    text_field: str,
    label_text_map: Dict[int, str],
) -> Dataset:
    return Dataset.from_dict(
        {
            text_field: list(texts),
            "prompt": list(prompts),
            "selection_confidence": list(confidences),
            "calibration_predicted_label": [int(prediction["predicted_label"]) for prediction in predictions],
            "calibration_predicted_text": [
                label_text_map[int(prediction["predicted_label"])] for prediction in predictions
            ],
        }
    )


def compute_balanced_quotas(total_samples: int, label_ids) -> Dict[int, int]:
    ordered_labels = sorted(int(label_id) for label_id in label_ids)
    base = total_samples // len(ordered_labels)
    remainder = total_samples % len(ordered_labels)
    return {
        label_id: base + (1 if index < remainder else 0)
        for index, label_id in enumerate(ordered_labels)
    }


def select_top_confidence_examples(
    texts,
    prompts,
    predictions,
    confidence_threshold: float,
    max_samples: int,
):
    scored = []
    for text, prompt, prediction in zip(texts, prompts, predictions):
        confidence = compute_prediction_confidence(prediction)
        if confidence >= confidence_threshold:
            scored.append((confidence, str(text), str(prompt), prediction))
    scored.sort(key=lambda item: item[0], reverse=True)
    selected = scored[:max_samples]
    if len(selected) < max_samples:
        raise ValueError(
            f"Only found {len(selected)} samples with confidence >= {confidence_threshold:.2f}, but need {max_samples}."
        )
    return (
        [item[1] for item in selected],
        [item[2] for item in selected],
        [item[3] for item in selected],
        [item[0] for item in selected],
    )


def select_class_balanced_examples(
    texts,
    prompts,
    predictions,
    confidence_threshold: float,
    total_samples: int,
    label_ids,
):
    quotas = compute_balanced_quotas(total_samples, label_ids)
    per_label_items: Dict[int, list] = {int(label_id): [] for label_id in label_ids}

    for text, prompt, prediction in zip(texts, prompts, predictions):
        predicted_label = int(prediction["predicted_label"])
        confidence = compute_prediction_confidence(prediction)
        if predicted_label in per_label_items and confidence >= confidence_threshold:
            per_label_items[predicted_label].append((confidence, str(text), str(prompt), prediction))

    selected_items = []
    for label_id, quota in quotas.items():
        label_items = sorted(per_label_items[label_id], key=lambda item: item[0], reverse=True)
        if len(label_items) < quota:
            raise ValueError(
                f"Only found {len(label_items)} samples for label {label_id} "
                f"with confidence >= {confidence_threshold:.2f}, but need {quota}."
            )
        selected_items.extend(label_items[:quota])

    selected_items.sort(key=lambda item: item[0], reverse=True)
    return (
        [item[1] for item in selected_items],
        [item[2] for item in selected_items],
        [item[3] for item in selected_items],
        [item[0] for item in selected_items],
    )


def build_split_dataset(
    *,
    task,
    model,
    tokenizer,
    dataset_source,
    split_name: str,
    text_field: str,
    candidate_pool_size: int,
    confidence_threshold: float,
    num_samples: int,
    selection_mode: str,
    label_ids,
    label_text_map: Dict[int, str],
    max_length: int,
    device: str,
) -> Dataset:
    split_dataset = select_split(dataset_source, split_name, 0, [text_field])
    candidate_texts = extract_texts(split_dataset, text_field, candidate_pool_size)
    candidate_prompts = [build_task_prompt(task, text) for text in candidate_texts]
    candidate_predictions = score_label_candidates(
        model=model,
        tokenizer=tokenizer,
        prompts=candidate_prompts,
        label_text_map=label_text_map,
        max_length=max_length,
        device=device,
        perturbation=None,
    )

    if selection_mode == "class_balanced":
        selected_texts, selected_prompts, selected_predictions, selected_confidences = select_class_balanced_examples(
            texts=candidate_texts,
            prompts=candidate_prompts,
            predictions=candidate_predictions,
            confidence_threshold=confidence_threshold,
            total_samples=num_samples,
            label_ids=label_ids,
        )
    else:
        selected_texts, selected_prompts, selected_predictions, selected_confidences = select_top_confidence_examples(
            texts=candidate_texts,
            prompts=candidate_prompts,
            predictions=candidate_predictions,
            confidence_threshold=confidence_threshold,
            max_samples=num_samples,
        )

    print(
        f"{split_name} samples: {len(selected_texts)} | "
        f"mean confidence: {sum(selected_confidences) / len(selected_confidences):.4f}"
    )
    print(
        f"{split_name} label counts: {dict(Counter(int(prediction['predicted_label']) for prediction in selected_predictions))}"
    )

    return build_saved_split(
        selected_texts,
        selected_prompts,
        selected_predictions,
        selected_confidences,
        text_field,
        label_text_map,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = resolve_dtype(args.dtype)

    if args.task == "sst2":
        args.label_text_map = '{"0":"negative","1":"positive"}'
    elif args.task == "agnews":
        args.label_text_map = '{"0":"World","1":"Sports","2":"Business","3":"Sci/Tech"}'
    label_text_map = parse_label_text_map(args.label_text_map)
    label_ids = sorted(label_text_map.keys())

    tokenizer = AutoTokenizer.from_pretrained(args.calibration_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.calibration_model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    model.to(device)
    model.eval()

    dataset_source = load_dataset_source(None, args.dataset_name, args.dataset_config)

    output_splits = {}
    if args.split_mode in ("both", "reference_only"):
        output_splits[args.reference_split] = build_split_dataset(
            task=args.task,
            model=model,
            tokenizer=tokenizer,
            dataset_source=dataset_source,
            split_name=args.reference_split,
            text_field=args.text_field,
            candidate_pool_size=args.reference_candidate_pool_size,
            confidence_threshold=args.reference_confidence_threshold,
            num_samples=args.num_reference_samples,
            selection_mode=args.selection_mode,
            label_ids=label_ids,
            label_text_map=label_text_map,
            max_length=args.max_length,
            device=device,
        )
    if args.split_mode in ("both", "probe_only"):
        output_splits[args.probe_split] = build_split_dataset(
            task=args.task,
            model=model,
            tokenizer=tokenizer,
            dataset_source=dataset_source,
            split_name=args.probe_split,
            text_field=args.text_field,
            candidate_pool_size=args.probe_candidate_pool_size,
            confidence_threshold=args.probe_confidence_threshold,
            num_samples=args.num_probe_samples,
            selection_mode=args.selection_mode,
            label_ids=label_ids,
            label_text_map=label_text_map,
            max_length=args.max_length,
            device=device,
        )

    output_dataset = DatasetDict(output_splits)
    output_dataset.save_to_disk(args.output_path)

    metadata = {
        "calibration_model_path": args.calibration_model_path,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "reference_split": args.reference_split,
        "probe_split": args.probe_split,
        "text_field": args.text_field,
        "reference_confidence_threshold": args.reference_confidence_threshold,
        "probe_confidence_threshold": args.probe_confidence_threshold,
        "selection_mode": args.selection_mode,
        "split_mode": args.split_mode,
        "num_reference_samples": args.num_reference_samples,
        "num_probe_samples": args.num_probe_samples,
    }
    metadata_path = os.path.join(args.output_path, "selection_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Saved filtered dataset to {args.output_path}")


if __name__ == "__main__":
    main()
