# Structural Adapter Release Code

This directory contains the core implementation for the paper method on the Hateful Memes Challenge dataset with Qwen2-VL.

It includes the CLIP aligned encoder, consensus-distinctive decomposition, complementary fusion, soft-prompt injection into Qwen2-VL, training, and evaluation. Scripts for all paper tables are not included.

This release is intended as a compact core-method implementation. It does not include the multi-dataset, multi-backbone, PEFT, ablation, case-study, visualization, or historical experiment scripts used during the broader study.

## Data Format

The training file can be a JSON file with `train` and `dev` splits or a JSONL file. For JSONL, records are filtered by the `split` field when it is present; otherwise the file is treated as a single-split file. Each example should contain:

```json
{
  "id": "42953",
  "img": "img/42953.png",
  "text": "its their character not their color that matters",
  "label": "not-hateful"
}
```

Labels are `not-hateful` and `hateful`.

## Installation

```bash
pip install -r requirements.txt
```

## Training

Edit the paths in `scripts/train_hmc.sh`, then run:

```bash
bash scripts/train_hmc.sh
```

The defaults follow the manuscript settings where specified: CLIP image size is 224, the MLLM backbone is Qwen2-VL-2B-Instruct, the optimizer is Adam, the learning rate is `1e-6`, the effective batch size is 32, the decomposition loss coefficients are 1, and the complementary-fusion window size and stride are 2.

This release trains the structural adapter only. Qwen2-VL and CLIP are frozen during training, while the decomposition, complementary-fusion, and prompt-projection modules are updated.

The final release checkpoint stores `adapter_model.bin`, `adapter_config.json`, and processor files. Qwen2-VL and CLIP base weights are loaded from the model names or paths recorded in `adapter_config.json`.

## Evaluation

Edit the paths in `scripts/evaluate_hmc.sh`, then run:

```bash
bash scripts/evaluate_hmc.sh
```

The evaluation script reports accuracy, macro-F1, and AUROC on the selected split. For HMC, use the development split when following the manuscript protocol.
