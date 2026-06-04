import argparse
import os

import torch
from transformers import AutoProcessor, CLIPProcessor, Trainer, TrainingArguments, set_seed

from data import HMCDataCollator, HatefulMemesDataset
from modeling import AdapterConfig, Qwen2VLStructuralAdapter


class AdamTrainer(Trainer):
    def create_optimizer(self):
        if self.optimizer is None:
            trainable_parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
            if not trainable_parameters:
                raise ValueError("No trainable parameters found.")
            self.optimizer = torch.optim.Adam(
                trainable_parameters,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
                weight_decay=self.args.weight_decay,
            )
        return self.optimizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_json", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--dev_split", type=str, default="dev")
    parser.add_argument("--qwen_model_name", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--clip_model_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--prompt_length", type=int, default=49)
    parser.add_argument("--window_size", type=int, default=2)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--lambda_dec", type=float, default=1.0)
    parser.add_argument("--alpha_rec", type=float, default=1.0)
    parser.add_argument("--gamma_cons", type=float, default=1.0)
    parser.add_argument("--delta_dis", type=float, default=1.0)
    parser.add_argument("--eta_orth", type=float, default=1.0)
    parser.add_argument("--zeta_spec", type=float, default=1.0)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--num_train_epochs", type=float, default=8.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    adapter_config = AdapterConfig(
        qwen_model_name=args.qwen_model_name,
        clip_model_name=args.clip_model_name,
        freeze_base_models=True,
        rank=args.rank,
        prompt_length=args.prompt_length,
        window_size=args.window_size,
        stride=args.stride,
        margin=args.margin,
        lambda_dec=args.lambda_dec,
        alpha_rec=args.alpha_rec,
        gamma_cons=args.gamma_cons,
        delta_dis=args.delta_dis,
        eta_orth=args.eta_orth,
        zeta_spec=args.zeta_spec,
    )
    model = Qwen2VLStructuralAdapter(adapter_config)
    qwen_processor = AutoProcessor.from_pretrained(args.qwen_model_name, trust_remote_code=True)
    clip_processor = CLIPProcessor.from_pretrained(args.clip_model_name)
    train_dataset = HatefulMemesDataset(args.data_json, args.image_dir, args.train_split)
    dev_dataset = HatefulMemesDataset(args.data_json, args.image_dir, args.dev_split)
    collator = HMCDataCollator(qwen_processor=qwen_processor, clip_processor=clip_processor, max_length=args.max_length)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=args.dataloader_num_workers,
        bf16=args.bf16,
        fp16=args.fp16,
        deepspeed=args.deepspeed,
        report_to=args.report_to,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
    )
    trainer = AdamTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=collator,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    if trainer.is_world_process_zero():
        os.makedirs(args.output_dir, exist_ok=True)
        final_model = trainer.accelerator.unwrap_model(trainer.model)
        final_model.save_release_model(args.output_dir)
        qwen_processor.save_pretrained(os.path.join(args.output_dir, "qwen_processor"))
        clip_processor.save_pretrained(os.path.join(args.output_dir, "clip_processor"))


if __name__ == "__main__":
    main()
