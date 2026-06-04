import argparse
import json
import os
from typing import Dict, Iterable, List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from transformers import AutoProcessor, CLIPProcessor

from data import HMCDataCollator, HMC_LABELS, HatefulMemesDataset
from modeling import Qwen2VLStructuralAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--data_json", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=2048)
    return parser.parse_args()


def batches(items: List[Dict[str, str]], batch_size: int) -> Iterable[List[Dict[str, str]]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def move_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def sequence_scores(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    shifted = labels[:, 1:]
    mask = shifted.ne(-100)
    safe_labels = shifted.masked_fill(~mask, 0)
    token_scores = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return (token_scores * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)


def align_labels_to_logits(labels: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    prefix_length = logits.size(1) - labels.size(1)
    if prefix_length < 0:
        raise ValueError("Labels are longer than logits.")
    if prefix_length == 0:
        return labels
    prefix = labels.new_full((labels.size(0), prefix_length), -100)
    return torch.cat([prefix, labels], dim=1)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Qwen2VLStructuralAdapter.from_release(args.model_dir).to(device)
    model.eval()
    qwen_processor_dir = os.path.join(args.model_dir, "qwen_processor")
    clip_processor_dir = os.path.join(args.model_dir, "clip_processor")
    qwen_processor_source = qwen_processor_dir if os.path.isdir(qwen_processor_dir) else model.adapter_config.qwen_model_name
    clip_processor_source = clip_processor_dir if os.path.isdir(clip_processor_dir) else model.adapter_config.clip_model_name
    qwen_processor = AutoProcessor.from_pretrained(qwen_processor_source, trust_remote_code=True)
    clip_processor = CLIPProcessor.from_pretrained(clip_processor_source)
    dataset = HatefulMemesDataset(args.data_json, args.image_dir, args.split)
    collator = HMCDataCollator(qwen_processor=qwen_processor, clip_processor=clip_processor, max_length=args.max_length)
    gold = []
    predictions = []
    probabilities = []
    outputs = []
    with torch.no_grad():
        for batch_examples in batches([dataset[index] for index in range(len(dataset))], args.batch_size):
            label_scores = []
            for label in HMC_LABELS:
                batch = collator.build_batch(batch_examples, target_labels=[label] * len(batch_examples))
                batch = move_to_device(batch, device)
                model_outputs = model(**batch)
                aligned_labels = align_labels_to_logits(batch["labels"], model_outputs.logits)
                label_scores.append(sequence_scores(model_outputs.logits, aligned_labels).cpu())
            scores = torch.stack(label_scores, dim=-1)
            probs = torch.softmax(scores, dim=-1).numpy()
            pred_ids = probs.argmax(axis=-1)
            for example, pred_id, prob in zip(batch_examples, pred_ids, probs):
                gold_label = example["label"]
                pred_label = HMC_LABELS[int(pred_id)]
                gold.append(gold_label)
                predictions.append(pred_label)
                probabilities.append(float(prob[1]))
                outputs.append(
                    {
                        "id": example["id"],
                        "gold": gold_label,
                        "pred": pred_label,
                        "prob_hateful": float(prob[1]),
                    }
                )
    y_true = [HMC_LABELS.index(label) for label in gold]
    y_pred = [HMC_LABELS.index(label) for label in predictions]
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "auroc": float(roc_auc_score(y_true, np.array(probabilities))),
    }
    result = {"metrics": metrics, "predictions": outputs}
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
