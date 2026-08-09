"""Shared utilities for MoE routing perturbation detectors."""

import json
import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from torch.func import functional_call


@dataclass
class SelectedWeight:
    name: str
    key: str
    out_features: int
    in_features: int


def resolve_dtype(name: str) -> Optional[torch.dtype]:
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
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset_source(dataset_path: Optional[str], dataset_name: str, dataset_config: Optional[str]):
    if dataset_path:
        return load_from_disk(dataset_path)
    if dataset_config:
        return load_dataset(dataset_name, dataset_config)
    return load_dataset(dataset_name)


def select_split(dataset_source, split_name: str, max_samples: int, required_fields: Sequence[str]) -> Dataset:
    if isinstance(dataset_source, DatasetDict):
        if split_name not in dataset_source:
            raise ValueError(f"Split '{split_name}' not found. Available: {list(dataset_source.keys())}")
        dataset = dataset_source[split_name]
    else:
        dataset = dataset_source

    missing_fields = [field for field in required_fields if field not in dataset.column_names]
    if missing_fields:
        raise ValueError(f"Missing required fields {missing_fields} in dataset columns {dataset.column_names}")

    if max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return dataset

def is_wikitext_heading(text: str) -> bool:
    normalized = text.strip()

    # Match:
    # = Title =
    # == Life ==
    # = = Life = =
    return bool(
        re.fullmatch(
            r"(?:=\s*){1,6}.+?(?:\s*=){1,6}",
            normalized,
        )
    )

def is_valid_wikitext_sample(text: str, min_words: int = 10) -> bool:
    text = text.strip()

    if not text:
        return False

    if is_wikitext_heading(text):
        return False

    if len(text.split()) < min_words:
        return False

    return True

# Extract texts from a dataset.
def extract_texts(
    dataset: Dataset,
    text_field: str,
    max_samples: int,
    min_words: int = 10,
) -> List[str]:
    texts: List[str] = []

    for item in dataset:
        raw_text = str(item[text_field]).strip()

        if not is_valid_wikitext_sample(raw_text, min_words):
            continue

        texts.append(raw_text)

        if max_samples > 0 and len(texts) >= max_samples:
            break

    if not texts:
        raise ValueError(
            f"No valid texts found in field '{text_field}'."
        )

    return texts


def batched(items: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def count_batches(total_items: int, batch_size: int) -> int:
    return max((total_items + batch_size - 1) // batch_size, 0)


def parse_label_text_map(payload: str) -> Dict[int, str]:
    raw = json.loads(payload)
    return {int(key): str(value) for key, value in raw.items()}


def sanitize_name(name: str) -> str:
    return name.replace(".", "__")


# Module implementing low‑rank LoRA perturbations for attention weights.
class AttentionLoRAPerturbation(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        attention_pattern: str,
        layer_mode: str,
        num_layers: int,
        rank: int,
        alpha: float,
        max_selected_weights: int,
        device: str,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.device = device
        self.selected: List[SelectedWeight] = []
        self.a_params = torch.nn.ParameterDict()
        self.b_params = torch.nn.ParameterDict()

        matched_parameters: List[Tuple[str, torch.nn.Parameter]] = []
        for name, parameter in model.named_parameters():
            if attention_pattern not in name:
                continue
            if parameter.ndim != 2 or not name.endswith("weight"):
                continue
            matched_parameters.append((name, parameter))

        if not matched_parameters:
            raise ValueError(f"No attention weights matched pattern '{attention_pattern}'.")

        layer_groups: Dict[int, List[Tuple[str, torch.nn.Parameter]]] = {}
        fallback_groups: Dict[str, List[Tuple[str, torch.nn.Parameter]]] = {}
        for name, parameter in matched_parameters:
            layer_match = re.search(r"model\.layers\.(\d+)\.", name)
            if layer_match is not None:
                layer_index = int(layer_match.group(1))
                layer_groups.setdefault(layer_index, []).append((name, parameter))
            else:
                fallback_prefix = name.rsplit(".", 1)[0]
                fallback_groups.setdefault(fallback_prefix, []).append((name, parameter))

        if layer_groups:
            sorted_layer_indices = sorted(layer_groups)
            if layer_mode == "last":
                selected_layer_indices = sorted_layer_indices[-1:]
            elif layer_mode == "last_n":
                if num_layers <= 0:
                    raise ValueError("--num_perturb_layers must be greater than 0 for --perturb_layer_mode last_n.")
                selected_layer_indices = sorted_layer_indices[-num_layers:]
            elif layer_mode == "all":
                selected_layer_indices = sorted_layer_indices
            else:
                raise ValueError(f"Unsupported perturb layer mode: {layer_mode}")
            selected_parameters = [
                item
                for layer_index in selected_layer_indices
                for item in layer_groups[layer_index]
            ]
        else:
            if layer_mode != "last":
                raise ValueError(
                    "Could not infer model.layers.<index> from attention weights; "
                    "only --perturb_layer_mode last is supported for this model naming scheme."
                )
            last_layer_prefix = matched_parameters[-1][0].rsplit(".", 1)[0]
            selected_parameters = fallback_groups[last_layer_prefix]

        for name, parameter in selected_parameters:
            key = sanitize_name(name)
            out_features, in_features = parameter.shape
            self.selected.append(
                SelectedWeight(
                    name=name,
                    key=key,
                    out_features=out_features,
                    in_features=in_features,
                )
            )
            self.a_params[key] = torch.nn.Parameter(torch.zeros(out_features, rank, device=device))
            self.b_params[key] = torch.nn.Parameter(torch.zeros(rank, in_features, device=device))
            torch.nn.init.normal_(self.b_params[key], std=1e-3)
            if len(self.selected) >= max_selected_weights:
                break

        if not self.selected:
            raise ValueError(f"No weights were selected from the last attention layer for pattern '{attention_pattern}'.")

    def build_delta(self, entry: SelectedWeight) -> torch.Tensor:
        scale = self.alpha / max(self.rank, 1)
        return (self.a_params[entry.key] @ self.b_params[entry.key]) * scale

    def delta_norm(self) -> torch.Tensor:
        total = torch.tensor(0.0, device=self.device)
        for entry in self.selected:
            total = total + self.build_delta(entry).pow(2).mean()
        return total

# Build a state dictionary for the model, optionally applying a perturbation to selected attention weights.
def build_state_dict(model: torch.nn.Module, perturbation: Optional[AttentionLoRAPerturbation]) -> Dict[str, torch.Tensor]:
    state = {name: tensor for name, tensor in model.named_parameters()}
    state.update({name: tensor for name, tensor in model.named_buffers()})
    if perturbation is None:
        return state
    for entry in perturbation.selected:
        delta = perturbation.build_delta(entry).to(dtype=state[entry.name].dtype)
        state[entry.name] = state[entry.name] + delta
    return state


def prepare_router_tensor(layer_router_logits: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
    if layer_router_logits.dim() == 3:
        return layer_router_logits
    if layer_router_logits.dim() == 2:
        if layer_router_logits.size(0) != batch_size * seq_len:
            raise ValueError(
                f"Unexpected router tensor shape {tuple(layer_router_logits.shape)} "
                f"for batch_size={batch_size}, seq_len={seq_len}"
            )
        return layer_router_logits.view(batch_size, seq_len, layer_router_logits.size(-1))
    raise ValueError(f"Unsupported router tensor rank: {layer_router_logits.dim()}")


def normalize_distribution(values: torch.Tensor) -> torch.Tensor:
    denom = values.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(values.dtype).eps)
    return values / denom


def kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(p.dtype).eps
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return torch.sum(p * (torch.log(p) - torch.log(q)), dim=-1)


def js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    midpoint = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, midpoint) + 0.5 * kl_divergence(q, midpoint)


def normalized_entropy(distribution: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(distribution.dtype).eps
    entropy = -torch.sum(distribution.clamp_min(eps) * torch.log(distribution.clamp_min(eps)), dim=-1)
    return entropy / math.log(distribution.size(-1))


def build_target_batch(
    tokenizer: AutoTokenizer,
    prompts: Sequence[str],
    target_label_texts: Sequence[str],
    max_length: int,
) -> Dict[str, torch.Tensor]:
    features: List[Dict[str, List[int]]] = []
    if len(prompts) != len(target_label_texts):
        raise ValueError("prompts and target_label_texts must have the same length.")

    for prompt, target_label_text in zip(prompts, target_label_texts):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        label_ids = tokenizer.encode(target_label_text, add_special_tokens=False)
        input_ids = prompt_ids + label_ids
        if len(input_ids) > max_length:
            overflow = len(input_ids) - max_length
            prompt_ids = prompt_ids[overflow:]
            input_ids = prompt_ids + label_ids
        labels = [-100] * len(prompt_ids) + label_ids
        features.append(
            {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
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

    final_length = batch["input_ids"].size(1)
    padded_labels = []
    for feature in features:
        labels = feature["labels"]
        if len(labels) < final_length:
            labels = labels + [-100] * (final_length - len(labels))
        else:
            labels = labels[:final_length]
        padded_labels.append(labels)
    batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
    return batch


def forward_model(
    model: torch.nn.Module,
    state: Dict[str, torch.Tensor],
    **kwargs,
):
    return functional_call(model, state, (), kwargs)


def optimize_attention_perturbation(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    reference_prompts: Sequence[str],
    reference_target_texts: Sequence[str],
    args: object,
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
    target_batch = build_target_batch(tokenizer, reference_prompts, reference_target_texts, args.max_length)
    target_batch = {key: value.to(device) for key, value in target_batch.items()}

    history: List[Dict[str, float]] = []
    for step in tqdm(range(args.perturb_steps), desc="Optimize perturbation"):
        optimizer.zero_grad(set_to_none=True)
        state = build_state_dict(model, perturbation)
        outputs = forward_model(
            model,
            state,
            input_ids=target_batch["input_ids"],
            attention_mask=target_batch["attention_mask"],
            labels=target_batch["labels"],
            output_router_logits=True,
            use_cache=False,
        )
        ce_loss = outputs.loss
        norm_penalty = perturbation.delta_norm()
        loss = ce_loss + args.delta_penalty * norm_penalty
        loss.backward()
        optimizer.step()
        history.append(
            {
                "step": float(step),
                "ce_loss": float(ce_loss.detach().cpu()),
                "delta_penalty": float(norm_penalty.detach().cpu()),
                "total_loss": float(loss.detach().cpu()),
            }
        )
    return perturbation, history


@torch.inference_mode()
def score_label_candidates(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompts: Sequence[str],
    label_text_map: Dict[int, str],
    max_length: int,
    device: str,
    perturbation: Optional[AttentionLoRAPerturbation] = None,
) -> List[Dict[str, float]]:
    state = build_state_dict(model, perturbation)
    label_ids_map = {
        label: tokenizer.encode(text, add_special_tokens=False)
        for label, text in label_text_map.items()
    }
    predictions: List[Dict[str, float]] = []

    for prompt in tqdm(prompts, desc="Score labels"):
        label_scores: Dict[int, float] = {}
        for label, label_ids in label_ids_map.items():
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            input_ids = prompt_ids + label_ids
            if len(input_ids) > max_length:
                prompt_ids = prompt_ids[len(input_ids) - max_length :]
                input_ids = prompt_ids + label_ids
            tensor_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(tensor_ids, device=device)
            outputs = forward_model(
                model,
                state,
                input_ids=tensor_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            logits = outputs.logits[:, :-1, :]
            targets = tensor_ids[:, 1:]

            prompt_prefix = len(prompt_ids)
            label_start = max(prompt_prefix - 1, 0)
            label_end = label_start + len(label_ids)
            label_logits = logits[:, label_start:label_end, :]
            label_targets = targets[:, label_start:label_end]
            log_probs = torch.log_softmax(label_logits, dim=-1)
            gathered = torch.gather(log_probs, 2, label_targets.unsqueeze(-1)).squeeze(-1)
            label_scores[label] = float(gathered.sum().detach().cpu())

        predicted_label, predicted_score = max(label_scores.items(), key=lambda item: item[1])
        predictions.append(
            {
                "predicted_label": float(predicted_label),
                "predicted_score": predicted_score,
                "label_scores": {str(label): score for label, score in label_scores.items()},
            }
        )
    return predictions


def compute_attack_success(predictions: Sequence[Dict[str, float]], target_labels: Sequence[int]) -> float:
    if len(predictions) != len(target_labels):
        raise ValueError("predictions and target_labels must have the same length.")
    success = sum(
        int(int(prediction["predicted_label"]) == int(target_label))
        for prediction, target_label in zip(predictions, target_labels)
    )
    return success / max(len(predictions), 1)


def compute_mean_target_score(predictions: Sequence[Dict[str, float]], target_labels: Sequence[int]) -> float:
    if len(predictions) != len(target_labels):
        raise ValueError("predictions and target_labels must have the same length.")
    if not predictions:
        return 0.0
    total = 0.0
    for prediction, target_label in zip(predictions, target_labels):
        target_key = str(int(target_label))
        total += float(prediction["label_scores"][target_key])
    return total / len(predictions)


@torch.inference_mode()
def collect_router_distributions(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompts: Sequence[str],
    max_length: int,
    batch_size: int,
    pooling: str,
    device: str,
    perturbation: Optional[AttentionLoRAPerturbation] = None,
) -> torch.Tensor:
    state = build_state_dict(model, perturbation)
    outputs_by_batch: List[torch.Tensor] = []

    for batch_prompts in tqdm(
        batched(prompts, batch_size),
        total=count_batches(len(prompts), batch_size),
        desc="Collect routing",
        # leave=False,
    ):
        encoded = tokenizer(
            list(batch_prompts),
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        outputs = forward_model(
            model,
            state,
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            output_router_logits=True,
            use_cache=False,
        )
        router_logits = getattr(outputs, "router_logits", None)
        if router_logits is None:
            raise RuntimeError("Model outputs do not contain router logits.")

        attention_mask = encoded["attention_mask"]
        current_batch_size, seq_len = encoded["input_ids"].shape
        layer_distributions: List[torch.Tensor] = []
        for layer_router_logits in router_logits:
            routed = prepare_router_tensor(layer_router_logits, current_batch_size, seq_len).float()
            probs = torch.softmax(routed, dim=-1)
            if pooling == "last_token":
                last_positions = attention_mask.sum(dim=1) - 1
                batch_index = torch.arange(current_batch_size, device=probs.device)
                pooled = probs[batch_index, last_positions]
            else:
                pooled = (probs * attention_mask.unsqueeze(-1)).sum(dim=1)
            layer_distributions.append(normalize_distribution(pooled))
        outputs_by_batch.append(torch.stack(layer_distributions, dim=1).cpu())

    return torch.cat(outputs_by_batch, dim=0)


def summarize_routing_shift(
    base_distributions: torch.Tensor,
    perturbed_distributions: torch.Tensor,
) -> Dict[str, object]:
    js_values = js_divergence(base_distributions, perturbed_distributions)
    base_entropy = normalized_entropy(base_distributions)
    perturbed_entropy = normalized_entropy(perturbed_distributions)
    base_concentration = base_distributions.max(dim=-1).values
    perturbed_concentration = perturbed_distributions.max(dim=-1).values

    layer_mean_js = js_values.mean(dim=0)
    layer_mean_concentration_gain = (perturbed_concentration - base_concentration).mean(dim=0)
    layer_mean_entropy_drop = (base_entropy - perturbed_entropy).mean(dim=0)

    return {
        "mean_js": float(js_values.mean()),
        "max_js": float(js_values.max()),
        "mean_entropy_drop": float((base_entropy - perturbed_entropy).mean()),
        "mean_concentration_gain": float((perturbed_concentration - base_concentration).mean()),
        "layerwise": [
            {
                "layer_index": int(index),
                "mean_js": float(layer_mean_js[index]),
                "mean_entropy_drop": float(layer_mean_entropy_drop[index]),
                "mean_concentration_gain": float(layer_mean_concentration_gain[index]),
            }
            for index in range(layer_mean_js.numel())
        ],
    }


def tensor_values(values: torch.Tensor) -> List[float]:
    return [float(value) for value in values.detach().cpu().tolist()]


def top_experts(distribution: torch.Tensor, top_k: int = 4) -> List[Dict[str, float]]:
    k = min(top_k, distribution.numel())
    values, indices = torch.topk(distribution, k=k)
    return [
        {"expert_index": int(index), "probability": float(value)}
        for value, index in zip(values.detach().cpu(), indices.detach().cpu())
    ]


def summarize_expert_distributions(
    base_distributions: torch.Tensor,
    perturbed_distributions: torch.Tensor,
    selected_layer_indices: Sequence[int],
    max_report_layers: int,
) -> List[Dict[str, object]]:
    if max_report_layers < 0:
        report_layer_indices = list(range(base_distributions.size(1)))
    elif max_report_layers == 0:
        report_layer_indices = list(selected_layer_indices)
    else:
        report_layer_indices = list(selected_layer_indices)[:max_report_layers]

    summaries: List[Dict[str, object]] = []
    for layer_index in report_layer_indices:
        base_mean = base_distributions[:, layer_index, :].mean(dim=0)
        perturbed_mean = perturbed_distributions[:, layer_index, :].mean(dim=0)
        delta = perturbed_mean - base_mean
        summaries.append(
            {
                "layer_index": int(layer_index),
                "base_mean_distribution": tensor_values(base_mean),
                "perturbed_mean_distribution": tensor_values(perturbed_mean),
                "delta_distribution": tensor_values(delta),
                "base_top_experts": top_experts(base_mean),
                "perturbed_top_experts": top_experts(perturbed_mean),
            }
        )
    return summaries


def compute_routing_metrics(
    selected_weights: Sequence[SelectedWeight],
    routing_shift: Dict[str, object],
    metric: str,
) -> Dict[str, object]:
    selected_layer_indices = sorted(
        {
            int(entry.name.split(".")[2])
            for entry in selected_weights
            if entry.name.startswith("model.layers.")
        }
    )
    selected_layer_shifts = [
        layer for layer in routing_shift["layerwise"] if layer["layer_index"] in selected_layer_indices
    ]
    max_targeted_layer_js = max(
        (layer["mean_js"] for layer in selected_layer_shifts),
        default=0.0,
    )
    max_targeted_layer_concentration_gain = max(
        (layer["mean_concentration_gain"] for layer in selected_layer_shifts),
        default=0.0,
    )
    metric_values = {
        "max_targeted_layer_js": max_targeted_layer_js,
        "routing_mean_js": float(routing_shift["mean_js"]),
        "routing_max_js": float(routing_shift["max_js"]),
        "routing_mean_concentration_gain": float(routing_shift["mean_concentration_gain"]),
        "max_targeted_layer_concentration_gain": max_targeted_layer_concentration_gain,
    }
    return {
        "selected_layer_indices": selected_layer_indices,
        "selected_layer_shifts": selected_layer_shifts,
        "max_targeted_layer_js": max_targeted_layer_js,
        "max_targeted_layer_concentration_gain": max_targeted_layer_concentration_gain,
        "metric_values": metric_values,
        "detection_score": metric_values[metric],
    }
