import argparse
import json
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from routing_detector_utils import (
    AttentionLoRAPerturbation,
    build_state_dict,
    collect_router_distributions,
    compute_attack_success,
    compute_mean_target_score,
    compute_routing_metrics,
    extract_texts,
    forward_model,
    js_divergence,
    load_dataset_source,
    normalize_distribution,
    parse_label_text_map,
    prepare_router_tensor,
    resolve_dtype,
    score_label_candidates,
    select_split,
    set_seed,
    summarize_expert_distributions,
    summarize_routing_shift,
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
        description=(
            "Measure MoE routing instability by optimizing a small attention perturbation to change expert routing while preserving output label behavior."
        )
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    parser.add_argument("--dataset_config", type=str, default="wikitext-103-v1")
    parser.add_argument("--task_name", type=str, choices=PROMPT_TEMPLATES.keys(), default="sst2")
    parser.add_argument("--reference_split", type=str, default="validation")
    parser.add_argument("--probe_split", type=str, default="test")
    parser.add_argument("--text_field", type=str, default="text")
    # parser.add_argument("--label_text_map", type=str, default='{"0":"negative","1":"positive"}')
    parser.add_argument("--target_label", type=int, default=1)
    parser.add_argument("--num_reference_samples", type=int, default=32)
    parser.add_argument("--num_probe_samples", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--pooling", type=str, choices=["all", "last_token"], default="all")
    parser.add_argument("--dtype", type=str, choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attention_pattern", type=str, default="self_attn")
    parser.add_argument("--perturb_layer_mode", type=str, choices=["last", "last_n", "all"], default="last_n")
    parser.add_argument("--num_perturb_layers", type=int, default=4)
    parser.add_argument("--lora_rank", type=int, default=2)
    parser.add_argument("--lora_alpha", type=float, default=4.0)
    parser.add_argument("--perturb_steps", type=int, default=40)
    parser.add_argument("--perturb_lr", type=float, default=0.01)
    parser.add_argument("--perturb_weight_decay", type=float, default=0.0)
    parser.add_argument("--delta_penalty", type=float, default=0.01)
    parser.add_argument("--output_kl_weight", type=float, default=1.0)
    parser.add_argument("--routing_shift_weight", type=float, default=1.0)
    parser.add_argument("--max_selected_weights", type=int, default=64)
    parser.add_argument(
        "--detection_metric",
        type=str,
        choices=[
            "max_targeted_layer_js",
            "routing_mean_js",
            "routing_max_js",
            "routing_mean_concentration_gain",
            "max_targeted_layer_concentration_gain",
        ],
        default="max_targeted_layer_js",
    )
    parser.add_argument("--max_report_layers", type=int, default=0)
    parser.add_argument("--output_json", type=str, default=None)
    return parser.parse_args()


def build_task_prompt(task_name: str, text: str) -> str:
    return PROMPT_TEMPLATES[task_name].format(text=text)


def selected_layer_indices(selected_weights) -> List[int]:
    return sorted({int(entry.name.split(".")[2]) for entry in selected_weights if entry.name.startswith("model.layers.")})


def pool_router_logits(
    router_logits,
    attention_mask: torch.Tensor,
    pooling: str,
) -> torch.Tensor:
    batch_size, seq_len = attention_mask.shape
    layer_distributions = []
    for layer_router_logits in router_logits:
        routed = prepare_router_tensor(layer_router_logits, batch_size, seq_len).float()
        probs = torch.softmax(routed, dim=-1)
        if pooling == "last_token":
            last_positions = attention_mask.sum(dim=1) - 1
            batch_index = torch.arange(batch_size, device=probs.device)
            pooled = probs[batch_index, last_positions]
        else:
            pooled = (probs * attention_mask.unsqueeze(-1)).sum(dim=1)
        layer_distributions.append(normalize_distribution(pooled))
    return torch.stack(layer_distributions, dim=1)


def label_log_scores(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompts: Sequence[str],
    label_text_map: Dict[int, str],
    max_length: int,
    device: str,
    perturbation: Optional[AttentionLoRAPerturbation] = None,
) -> torch.Tensor:
    state = build_state_dict(model, perturbation)
    labels = sorted(label_text_map)
    score_columns = []

    for label in labels:
        label_ids = tokenizer.encode(label_text_map[label], add_special_tokens=False)
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
        outputs = forward_model(
            model,
            state,
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        )
        logits = outputs.logits[:, :-1, :]
        targets = batch["input_ids"][:, 1:]
        rows = []
        for row_index, feature in enumerate(features):
            label_start = feature["label_start"]
            label_end = label_start + feature["label_length"]
            label_logits = logits[row_index : row_index + 1, label_start:label_end, :]
            label_targets = targets[row_index : row_index + 1, label_start:label_end]
            log_probs = torch.log_softmax(label_logits, dim=-1)
            gathered = torch.gather(log_probs, 2, label_targets.unsqueeze(-1)).squeeze(-1)
            rows.append(gathered.sum())
        score_columns.append(torch.stack(rows))

    return torch.stack(score_columns, dim=1)


def label_distribution_kl(base_scores: torch.Tensor, perturbed_scores: torch.Tensor) -> torch.Tensor:
    base_log_probs = torch.log_softmax(base_scores, dim=-1)
    base_probs = base_log_probs.exp()
    perturbed_log_probs = torch.log_softmax(perturbed_scores, dim=-1)
    return torch.sum(base_probs * (base_log_probs - perturbed_log_probs), dim=-1).mean()


def compute_output_label_kl(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompts: Sequence[str],
    label_text_map: Dict[int, str],
    max_length: int,
    device: str,
    perturbation: AttentionLoRAPerturbation,
) -> float:
    with torch.inference_mode():
        base_scores = label_log_scores(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            label_text_map=label_text_map,
            max_length=max_length,
            device=device,
            perturbation=None,
        )
        perturbed_scores = label_log_scores(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            label_text_map=label_text_map,
            max_length=max_length,
            device=device,
            perturbation=perturbation,
        )
        output_kl = label_distribution_kl(base_scores, perturbed_scores)
    return float(output_kl.detach().cpu())


def encode_prompts(
    tokenizer: AutoTokenizer,
    prompts: Sequence[str],
    max_length: int,
    device: str,
) -> Dict[str, torch.Tensor]:
    encoded = tokenizer(
        list(prompts),
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    )
    return {key: value.to(device) for key, value in encoded.items()}


def optimize_routing_instability(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    reference_prompts: Sequence[str],
    label_text_map: Dict[int, str],
    args: argparse.Namespace,
    device: str,
) -> Tuple[AttentionLoRAPerturbation, List[Dict[str, float]]]:
    perturbation = AttentionLoRAPerturbation(
        model=model,
        attention_pattern=args.attention_pattern,
        layer_mode=args.perturb_layer_mode,
        num_layers=args.num_perturb_layers,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        max_selected_weights=args.max_selected_weights,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        perturbation.parameters(),
        lr=args.perturb_lr,
        weight_decay=args.perturb_weight_decay,
    )

    selected_layers = selected_layer_indices(perturbation.selected)
    encoded = encode_prompts(tokenizer, reference_prompts, args.max_length, device)
    with torch.inference_mode():
        base_outputs = forward_model(
            model,
            build_state_dict(model, None),
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            output_router_logits=True,
            use_cache=False,
        )
        base_router_distributions = pool_router_logits(
            base_outputs.router_logits,
            encoded["attention_mask"],
            args.pooling,
        ).detach()
        base_label_scores = label_log_scores(
            model=model,
            tokenizer=tokenizer,
            prompts=reference_prompts,
            label_text_map=label_text_map,
            max_length=args.max_length,
            device=device,
            perturbation=None,
        ).detach()

    history: List[Dict[str, float]] = []
    for step in tqdm(range(args.perturb_steps), desc="Optimize routing instability"):
        optimizer.zero_grad(set_to_none=True)
        state = build_state_dict(model, perturbation)
        outputs = forward_model(
            model,
            state,
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            output_router_logits=True,
            use_cache=False,
        )
        perturbed_router_distributions = pool_router_logits(
            outputs.router_logits,
            encoded["attention_mask"],
            args.pooling,
        )
        js_values = js_divergence(base_router_distributions, perturbed_router_distributions)
        if selected_layers:
            routing_shift = js_values[:, selected_layers].mean()
        else:
            routing_shift = js_values.mean()

        perturbed_label_scores = label_log_scores(
            model=model,
            tokenizer=tokenizer,
            prompts=reference_prompts,
            label_text_map=label_text_map,
            max_length=args.max_length,
            device=device,
            perturbation=perturbation,
        )
        output_kl = label_distribution_kl(base_label_scores, perturbed_label_scores)
        delta_norm = perturbation.delta_norm()
        loss = -args.routing_shift_weight * routing_shift + args.output_kl_weight * output_kl + args.delta_penalty * delta_norm
        loss.backward()
        optimizer.step()

        history.append(
            {
                "step": float(step),
                "routing_shift_objective": float(routing_shift.detach().cpu()),
                "output_label_kl": float(output_kl.detach().cpu()),
                "delta_norm": float(delta_norm.detach().cpu()),
                "total_loss": float(loss.detach().cpu()),
            }
        )

    return perturbation, history


def build_report(
    args: argparse.Namespace,
    selected_weights,
    optimization_history: Sequence[Dict[str, float]],
    probe_output_label_kl: float,
    baseline_reference_predictions,
    perturbed_reference_predictions,
    routing_shift,
    expert_distribution_shift,
    reference_texts: Sequence[str],
    probe_texts: Sequence[str],
) -> Dict[str, object]:
    reference_target_labels = [args.target_label] * len(reference_texts)
    baseline_success = compute_attack_success(baseline_reference_predictions, reference_target_labels)
    perturbed_success = compute_attack_success(perturbed_reference_predictions, reference_target_labels)
    baseline_target_score = compute_mean_target_score(baseline_reference_predictions, reference_target_labels)
    perturbed_target_score = compute_mean_target_score(perturbed_reference_predictions, reference_target_labels)
    routing_metrics = compute_routing_metrics(selected_weights, routing_shift, args.detection_metric)
    final_step = optimization_history[-1] if optimization_history else {}
    final_reference_output_kl = float(final_step.get("output_label_kl", 0.0))
    final_delta_norm = float(final_step.get("delta_norm", 0.0))
    max_targeted_layer_js = routing_metrics["max_targeted_layer_js"]
    instability_score = max_targeted_layer_js / ((probe_output_label_kl + 1e-8) * math.sqrt(final_delta_norm + 1e-8))

    return {
        "model_path": args.model_path,
        "dataset_path": args.dataset_path,
        "dataset": args.dataset_name,
        "dataset_config": args.dataset_config,
        "reference_split": args.reference_split,
        "probe_split": args.probe_split,
        "target_label": args.target_label,
        "analysis_mode": "routing_instability_output_preservation",
        "verdict": "analysis_only",
        "verdict_source": "no_hardcoded_threshold",
        "baseline_reference_attack_success": baseline_success,
        "perturbed_reference_attack_success": perturbed_success,
        "reference_attack_success_gain": perturbed_success - baseline_success,
        "baseline_reference_target_score": baseline_target_score,
        "perturbed_reference_target_score": perturbed_target_score,
        "reference_target_score_gain": perturbed_target_score - baseline_target_score,
        "routing_instability_score": instability_score,
        "final_output_label_kl": final_reference_output_kl,
        "reference_output_label_kl": final_reference_output_kl,
        "probe_output_label_kl": probe_output_label_kl,
        "routing_instability_score_source": "probe_max_targeted_layer_js_over_probe_output_kl_and_delta_norm",
        "final_delta_norm": final_delta_norm,
        "detection_score": routing_metrics["detection_score"],
        "detection_metric": args.detection_metric,
        "routing_mean_js": routing_shift["mean_js"],
        "routing_max_js": routing_shift["max_js"],
        "routing_mean_entropy_drop": routing_shift["mean_entropy_drop"],
        "routing_mean_concentration_gain": routing_shift["mean_concentration_gain"],
        "max_targeted_layer_js": max_targeted_layer_js,
        "max_targeted_layer_concentration_gain": routing_metrics["max_targeted_layer_concentration_gain"],
        "selected_layer_indices": routing_metrics["selected_layer_indices"],
        "selected_layer_shifts": routing_metrics["selected_layer_shifts"],
        "layerwise_routing_shift": routing_shift["layerwise"],
        "expert_distribution_shift": list(expert_distribution_shift or []),
        "selected_attention_weights": [entry.name for entry in selected_weights],
        "objective_weights": {
            "routing_shift_weight": args.routing_shift_weight,
            "output_kl_weight": args.output_kl_weight,
            "delta_penalty": args.delta_penalty,
        },
        "num_reference_samples": len(reference_texts),
        "num_probe_samples": len(probe_texts),
        "optimization_history_tail": list(optimization_history[-10:]),
        "reference_examples": list(reference_texts),
        "probe_examples": list(probe_texts[:10]),
    }


def pretty_print_report(report: Dict[str, object]) -> None:
    print("=== MoE Routing Instability Report ===")
    print(f"Model: {report['model_path']}")
    print(f"Verdict: {report['verdict']} ({report['verdict_source']})")
    print(f"Selected layers: {report['selected_layer_indices']}")
    print(f"Routing instability score: {report['routing_instability_score']:.6f}")
    print(f"Max selected-layer JS: {report['max_targeted_layer_js']:.6f}")
    print(f"Reference output label KL: {report['reference_output_label_kl']:.6f}")
    print(f"Probe output label KL: {report['probe_output_label_kl']:.6f}")
    print(f"Final delta norm: {report['final_delta_norm']:.6f}")
    print(f"Reference target success gain: {report['reference_attack_success_gain']:.4f}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = resolve_dtype(args.dtype)

    if args.task_name == "sst2":
        args.label_text_map = '{"0":"negative","1":"positive"}'
    elif args.task_name == "agnews":
        args.label_text_map = '{"0":"World","1":"Sports","2":"Business","3":"Sci/Tech"}'
    label_text_map = parse_label_text_map(args.label_text_map)
    if args.target_label not in label_text_map:
        raise ValueError(f"Target label {args.target_label} missing from label_text_map.")

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
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if hasattr(model.config, "output_router_logits"):
        model.config.output_router_logits = True
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    dataset_source = load_dataset_source(args.dataset_path, args.dataset_name, args.dataset_config)
    reference_dataset = select_split(dataset_source, args.reference_split, 0, [args.text_field])
    probe_dataset = select_split(dataset_source, args.probe_split, 0, [args.text_field])
    reference_texts = extract_texts(reference_dataset, args.text_field, args.num_reference_samples)
    probe_texts = extract_texts(probe_dataset, args.text_field, args.num_probe_samples)
    reference_prompts = [build_task_prompt(args.task_name, text) for text in reference_texts]
    probe_prompts = [build_task_prompt(args.task_name, text) for text in probe_texts]

    baseline_reference_predictions = score_label_candidates(
        model=model,
        tokenizer=tokenizer,
        prompts=reference_prompts,
        label_text_map=label_text_map,
        max_length=args.max_length,
        device=device,
        perturbation=None,
    )

    perturbation, optimization_history = optimize_routing_instability(
        model=model,
        tokenizer=tokenizer,
        reference_prompts=reference_prompts,
        label_text_map=label_text_map,
        args=args,
        device=device,
    )

    perturbed_reference_predictions = score_label_candidates(
        model=model,
        tokenizer=tokenizer,
        prompts=reference_prompts,
        label_text_map=label_text_map,
        max_length=args.max_length,
        device=device,
        perturbation=perturbation,
    )
    base_probe_distributions = collect_router_distributions(
        model=model,
        tokenizer=tokenizer,
        prompts=probe_prompts,
        max_length=args.max_length,
        batch_size=args.batch_size,
        pooling=args.pooling,
        device=device,
        perturbation=None,
    )
    perturbed_probe_distributions = collect_router_distributions(
        model=model,
        tokenizer=tokenizer,
        prompts=probe_prompts,
        max_length=args.max_length,
        batch_size=args.batch_size,
        pooling=args.pooling,
        device=device,
        perturbation=perturbation,
    )
    probe_output_label_kl = compute_output_label_kl(
        model=model,
        tokenizer=tokenizer,
        prompts=probe_prompts,
        label_text_map=label_text_map,
        max_length=args.max_length,
        device=device,
        perturbation=perturbation,
    )
    routing_shift = summarize_routing_shift(base_probe_distributions, perturbed_probe_distributions)
    routing_metrics = compute_routing_metrics(perturbation.selected, routing_shift, args.detection_metric)
    expert_distribution_shift = summarize_expert_distributions(
        base_distributions=base_probe_distributions,
        perturbed_distributions=perturbed_probe_distributions,
        selected_layer_indices=routing_metrics["selected_layer_indices"],
        max_report_layers=args.max_report_layers,
    )

    report = build_report(
        args=args,
        selected_weights=perturbation.selected,
        optimization_history=optimization_history,
        probe_output_label_kl=probe_output_label_kl,
        baseline_reference_predictions=baseline_reference_predictions,
        perturbed_reference_predictions=perturbed_reference_predictions,
        routing_shift=routing_shift,
        expert_distribution_shift=expert_distribution_shift,
        reference_texts=reference_texts,
        probe_texts=probe_texts,
    )
    pretty_print_report(report)

    if args.output_json:
        output_dir = os.path.dirname(os.path.abspath(args.output_json))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"\nSaved report to {args.output_json}")


if __name__ == "__main__":
    main()
