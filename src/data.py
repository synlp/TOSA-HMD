import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from qwen_vl_utils import process_vision_info


HMC_LABELS = ["not-hateful", "hateful"]


def read_records(path: str, split: str) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        if any("split" in record for record in records):
            selected = [record for record in records if str(record.get("split", "")).lower() == split.lower()]
            if not selected:
                raise ValueError(f"Cannot find split {split} in {path}")
            return selected
        return records
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict) and split in data:
        return data[split]
    if isinstance(data, list):
        return data
    raise ValueError(f"Cannot find split {split} in {path}")


def normalize_label(label: Any) -> str:
    if isinstance(label, str):
        value = label.strip()
        if value in HMC_LABELS:
            return value
        if value.lower() in {"0", "false", "non-hateful", "not_hateful", "not hateful"}:
            return "not-hateful"
        if value.lower() in {"1", "true", "hateful", "hate"}:
            return "hateful"
    if int(label) == 0:
        return "not-hateful"
    if int(label) == 1:
        return "hateful"
    raise ValueError(f"Unsupported label: {label}")


def image_path(image_dir: str, record: Dict[str, Any]) -> str:
    value = record.get("img") or record.get("image") or record.get("image_path")
    if value is None:
        raise ValueError("Each record must contain img, image, or image_path")
    if os.path.isabs(value):
        return value
    return os.path.join(image_dir, value)


class HatefulMemesDataset(Dataset):
    def __init__(self, data_path: str, image_dir: str, split: str):
        self.image_dir = image_dir
        self.records = read_records(data_path, split)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        return {
            "id": record.get("id", index),
            "image_path": image_path(self.image_dir, record),
            "text": str(record.get("text", "")),
            "label": normalize_label(record.get("label")),
        }


@dataclass
class HMCDataCollator:
    qwen_processor: Any
    clip_processor: Any
    max_length: int = 2048
    image_size: int = 224
    instruction: str = "Is the multimodal content hateful or not-hateful?"

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        return self.build_batch(examples)

    def build_batch(
        self,
        examples: List[Dict[str, Any]],
        target_labels: Optional[Iterable[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        images = [Image.open(example["image_path"]).convert("RGB").resize((self.image_size, self.image_size)) for example in examples]
        labels = list(target_labels) if target_labels is not None else [example["label"] for example in examples]
        messages = [self._message(image, example["text"]) for image, example in zip(images, examples)]
        qwen_text = [
            self.qwen_processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            for message in messages
        ]
        qwen_images = []
        qwen_videos = []
        for message in messages:
            image_inputs, video_inputs = process_vision_info(message)
            qwen_images.extend(image_inputs or [])
            qwen_videos.extend(video_inputs or [])
        qwen_inputs = self.qwen_processor(
            text=qwen_text,
            images=qwen_images,
            videos=qwen_videos if qwen_videos else None,
            return_tensors="pt",
            padding=True,
            do_resize=True,
        )
        qwen_tokenizer = self.qwen_processor.tokenizer
        pad_token_id = qwen_tokenizer.pad_token_id
        eos_token_id = qwen_tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = eos_token_id
        input_ids = []
        attention_mask = []
        lm_labels = []
        for row, mask, label in zip(qwen_inputs["input_ids"], qwen_inputs["attention_mask"], labels):
            prompt_ids = row[mask.bool()].tolist()
            response_ids = qwen_tokenizer(str(label), add_special_tokens=False).input_ids + [eos_token_id]
            sequence = torch.tensor((prompt_ids + response_ids)[: self.max_length], dtype=torch.long)
            label_sequence = torch.full_like(sequence, -100)
            start = min(len(prompt_ids), self.max_length)
            label_sequence[start:] = sequence[start:]
            input_ids.append(sequence)
            attention_mask.append(torch.ones_like(sequence))
            lm_labels.append(label_sequence)
        batch = {
            "input_ids": pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id),
            "attention_mask": pad_sequence(attention_mask, batch_first=True, padding_value=0),
            "labels": pad_sequence(lm_labels, batch_first=True, padding_value=-100),
            "pixel_values": qwen_inputs["pixel_values"],
            "image_grid_thw": qwen_inputs["image_grid_thw"],
        }
        clip_inputs = self.clip_processor(
            text=[example["text"] for example in examples],
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        batch["clip_pixel_values"] = clip_inputs["pixel_values"]
        batch["clip_input_ids"] = clip_inputs["input_ids"]
        batch["clip_attention_mask"] = clip_inputs["attention_mask"]
        return batch

    def _message(self, image: Image.Image, text: str) -> List[Dict[str, Any]]:
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                        "resize_height": self.image_size,
                        "resize_width": self.image_size,
                    },
                    {"type": "text", "text": f"{text}\n{self.instruction}\n"},
                ],
            },
        ]
