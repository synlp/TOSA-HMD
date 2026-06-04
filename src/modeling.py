import inspect
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F
from transformers import CLIPModel, Qwen2VLForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutputWithPast


@dataclass
class AdapterConfig:
    qwen_model_name: str = "Qwen/Qwen2-VL-2B-Instruct"
    clip_model_name: str = "openai/clip-vit-base-patch32"
    freeze_base_models: bool = True
    rank: int = 64
    prompt_length: int = 49
    window_size: int = 2
    stride: int = 2
    margin: float = 1.0
    lambda_dec: float = 1.0
    alpha_rec: float = 1.0
    gamma_cons: float = 1.0
    delta_dis: float = 1.0
    eta_orth: float = 1.0
    zeta_spec: float = 1.0


def resize_sequence(x: torch.Tensor, target_length: int) -> torch.Tensor:
    if x.size(1) == target_length:
        return x
    return F.interpolate(x.transpose(1, 2), size=target_length, mode="linear", align_corners=False).transpose(1, 2)


def align_text_tokens(text_tokens: torch.Tensor, attention_mask: torch.Tensor, target_length: int) -> torch.Tensor:
    aligned = []
    for tokens, mask in zip(text_tokens, attention_mask):
        valid = tokens[mask.bool()]
        if valid.size(0) == 0:
            valid = tokens[:1]
        aligned.append(resize_sequence(valid.unsqueeze(0), target_length).squeeze(0))
    return torch.stack(aligned, dim=0)


class ConsensusDistinctiveDecomposition(nn.Module):
    def __init__(self, hidden_size: int, rank: int):
        super().__init__()
        self.consensus = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, rank),
        )
        self.image_specific = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.text_specific = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.basis = nn.Parameter(torch.empty(hidden_size, rank))
        nn.init.xavier_uniform_(self.basis)

    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        coefficients = self.consensus(torch.cat([image_features, text_features], dim=-1))
        consensus = torch.matmul(coefficients, self.basis.transpose(0, 1))
        image_specific = self.image_specific(image_features)
        text_specific = self.text_specific(text_features)
        return {
            "coefficients": coefficients,
            "basis": self.basis,
            "consensus": consensus,
            "image_specific": image_specific,
            "text_specific": text_specific,
        }


class ComplementaryFusion(nn.Module):
    def __init__(self, hidden_size: int, window_size: int, stride: int):
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        self.consensus_attention = nn.Linear(hidden_size * 2, 2)
        self.discrepancy = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.gate = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, consensus: torch.Tensor, image_specific: torch.Tensor, text_specific: torch.Tensor) -> torch.Tensor:
        local = F.avg_pool1d(
            consensus.transpose(1, 2),
            kernel_size=self.window_size,
            stride=self.stride,
            ceil_mode=True,
        ).transpose(1, 2)
        local = resize_sequence(local, consensus.size(1))
        global_feature = consensus.mean(dim=1, keepdim=True).expand_as(consensus)
        weights = torch.softmax(self.consensus_attention(torch.cat([local, global_feature], dim=-1)), dim=-1)
        agreement = weights[..., :1] * local + weights[..., 1:] * global_feature
        discrepancy = self.discrepancy(image_specific - text_specific)
        gate = torch.sigmoid(self.gate(torch.cat([agreement, discrepancy], dim=-1)))
        return gate * agreement + (1.0 - gate) * discrepancy


class StructuralPromptAdapter(nn.Module):
    def __init__(self, source_size: int, target_size: int, prompt_length: int):
        super().__init__()
        self.prompt_length = prompt_length
        self.projector = nn.Sequential(
            nn.Linear(source_size, target_size),
            nn.GELU(),
            nn.Linear(target_size, target_size),
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        fused = resize_sequence(fused, self.prompt_length)
        return self.projector(fused)


class Qwen2VLStructuralAdapter(nn.Module):
    def __init__(self, config: AdapterConfig):
        super().__init__()
        self.adapter_config = config
        self.clip = CLIPModel.from_pretrained(config.clip_model_name)
        self.qwen = Qwen2VLForConditionalGeneration.from_pretrained(config.qwen_model_name)
        self.qwen.config.use_cache = False
        clip_hidden = self.clip.config.projection_dim
        qwen_hidden = self.qwen.config.hidden_size
        self.decomposition = ConsensusDistinctiveDecomposition(clip_hidden, config.rank)
        self.fusion = ComplementaryFusion(clip_hidden, config.window_size, config.stride)
        self.prompt_adapter = StructuralPromptAdapter(clip_hidden, qwen_hidden, config.prompt_length)
        if config.freeze_base_models:
            self.freeze_base_models()

    def freeze_base_models(self) -> None:
        self.qwen.requires_grad_(False)
        self.clip.requires_grad_(False)
        self.qwen.eval()
        self.clip.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.adapter_config.freeze_base_models:
            self.qwen.eval()
            self.clip.eval()
        return self

    def encode_clip(
        self,
        clip_pixel_values: torch.Tensor,
        clip_input_ids: torch.Tensor,
        clip_attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        device = next(self.clip.parameters()).device
        dtype = next(self.clip.parameters()).dtype
        clip_pixel_values = clip_pixel_values.to(device=device, dtype=dtype)
        clip_input_ids = clip_input_ids.to(device=device)
        clip_attention_mask = clip_attention_mask.to(device=device)
        image_outputs = self.clip.vision_model(pixel_values=clip_pixel_values)
        text_outputs = self.clip.text_model(input_ids=clip_input_ids, attention_mask=clip_attention_mask)
        image_features = self.clip.visual_projection(image_outputs.last_hidden_state[:, 1:, :])
        text_features = self.clip.text_projection(text_outputs.last_hidden_state)
        text_features = align_text_tokens(text_features, clip_attention_mask, image_features.size(1))
        return {"image_features": image_features, "text_features": text_features}

    def decomposition_loss(self, parts: Dict[str, torch.Tensor], image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        consensus = parts["consensus"]
        image_specific = parts["image_specific"]
        text_specific = parts["text_specific"]
        coefficients = parts["coefficients"]
        basis = parts["basis"]
        rec = F.mse_loss(image_features, consensus + image_specific) + F.mse_loss(text_features, consensus + text_specific)
        cons = 0.5 * (coefficients.pow(2).mean() + basis.pow(2).mean())
        dis = image_specific.abs().mean() + text_specific.abs().mean()
        orth_image = torch.bmm(consensus.transpose(1, 2), image_specific).pow(2).mean()
        orth_text = torch.bmm(consensus.transpose(1, 2), text_specific).pow(2).mean()
        pooled_gap = image_specific.mean(dim=1) - text_specific.mean(dim=1)
        spec = F.relu(self.adapter_config.margin - pooled_gap.norm(dim=-1)).mean()
        return (
            self.adapter_config.alpha_rec * rec
            + self.adapter_config.gamma_cons * cons
            + self.adapter_config.delta_dis * dis
            + self.adapter_config.eta_orth * (orth_image + orth_text)
            + self.adapter_config.zeta_spec * spec
        )

    def qwen_embeddings(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = next(self.qwen.parameters()).device
        input_ids = input_ids.to(device)
        inputs_embeds = self.qwen.model.embed_tokens(input_ids)
        if pixel_values is not None:
            pixel_values = pixel_values.to(device=device, dtype=self.qwen.visual.get_dtype())
            image_grid_thw = image_grid_thw.to(device)
            image_embeds = self.qwen.visual(pixel_values, grid_thw=image_grid_thw)
            image_mask = (input_ids == self.qwen.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.to(device=device, dtype=self.qwen.visual.get_dtype())
            video_grid_thw = video_grid_thw.to(device)
            video_embeds = self.qwen.visual(pixel_values_videos, grid_thw=video_grid_thw)
            video_mask = (input_ids == self.qwen.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
        return inputs_embeds

    def qwen_forward_with_prompt(
        self,
        prompt_embeddings: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
    ) -> CausalLMOutputWithPast:
        device = next(self.qwen.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        prompt_embeddings = prompt_embeddings.to(device=device, dtype=self.qwen.model.embed_tokens.weight.dtype)
        token_embeddings = self.qwen_embeddings(input_ids, pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw)
        inputs_embeds = torch.cat([prompt_embeddings, token_embeddings], dim=1)
        prompt_mask = attention_mask.new_ones((attention_mask.size(0), prompt_embeddings.size(1)))
        extended_attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)
        pseudo_token = self.qwen.config.pad_token_id or self.qwen.config.eos_token_id
        prompt_ids = input_ids.new_full((input_ids.size(0), prompt_embeddings.size(1)), pseudo_token)
        extended_input_ids = torch.cat([prompt_ids, input_ids], dim=1)
        position_ids = None
        rope_deltas = None
        if hasattr(self.qwen, "get_rope_index"):
            position_ids, rope_deltas = self.qwen.get_rope_index(
                extended_input_ids,
                image_grid_thw.to(device) if image_grid_thw is not None else None,
                video_grid_thw.to(device) if video_grid_thw is not None else None,
                extended_attention_mask,
            )
        model_inputs = {
            "input_ids": None,
            "position_ids": position_ids,
            "attention_mask": extended_attention_mask,
            "past_key_values": None,
            "inputs_embeds": inputs_embeds,
            "use_cache": False,
            "output_attentions": False,
            "output_hidden_states": False,
            "return_dict": True,
        }
        if "cache_position" in inspect.signature(self.qwen.model.forward).parameters:
            model_inputs["cache_position"] = None
        outputs = self.qwen.model(**model_inputs)
        logits = self.qwen.lm_head(outputs[0]).float()
        extended_labels = None
        loss = None
        if labels is not None:
            labels = labels.to(device)
            prefix_labels = labels.new_full((labels.size(0), prompt_embeddings.size(1)), -100)
            extended_labels = torch.cat([prefix_labels, labels], dim=1)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = extended_labels[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=getattr(outputs, "past_key_values", None),
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        clip_pixel_values: torch.Tensor,
        clip_input_ids: torch.Tensor,
        clip_attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
    ) -> CausalLMOutputWithPast:
        encoded = self.encode_clip(clip_pixel_values, clip_input_ids, clip_attention_mask)
        parts = self.decomposition(encoded["image_features"], encoded["text_features"])
        fused = self.fusion(parts["consensus"], parts["image_specific"], parts["text_specific"])
        prompt_embeddings = self.prompt_adapter(fused)
        outputs = self.qwen_forward_with_prompt(
            prompt_embeddings=prompt_embeddings,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
        )
        if labels is not None and outputs.loss is not None:
            dec_loss = self.decomposition_loss(parts, encoded["image_features"], encoded["text_features"])
            outputs.loss = outputs.loss + self.adapter_config.lambda_dec * dec_loss
        return outputs

    def save_release_model(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        adapter_state = {
            key: value.cpu()
            for key, value in self.state_dict().items()
            if not key.startswith("qwen.") and not key.startswith("clip.")
        }
        torch.save(adapter_state, os.path.join(output_dir, "adapter_model.bin"))
        with open(os.path.join(output_dir, "adapter_config.json"), "w", encoding="utf-8") as handle:
            json.dump(asdict(self.adapter_config), handle, indent=2)

    @classmethod
    def from_release(cls, model_dir: str) -> "Qwen2VLStructuralAdapter":
        with open(os.path.join(model_dir, "adapter_config.json"), "r", encoding="utf-8") as handle:
            config = AdapterConfig(**json.load(handle))
        qwen_dir = os.path.join(model_dir, "qwen")
        clip_dir = os.path.join(model_dir, "clip")
        if os.path.isdir(qwen_dir):
            config.qwen_model_name = qwen_dir
        if os.path.isdir(clip_dir):
            config.clip_model_name = clip_dir
        model = cls(config)
        state = torch.load(os.path.join(model_dir, "adapter_model.bin"), map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        adapter_missing = [key for key in missing if not key.startswith("qwen.") and not key.startswith("clip.")]
        if adapter_missing or unexpected:
            details = {"missing": adapter_missing, "unexpected": unexpected}
            raise RuntimeError(f"Adapter checkpoint mismatch: {details}")
        return model
