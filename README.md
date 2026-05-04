# Qwen3 HADR Translation Fine-tuner

Fine‑tune **Qwen3** to translate HADR (Humanitarian Assistance and Disaster Relief) jargon from **Malay (ms), Chinese (zh), Indonesian (id), Tagalog (tl), Thai (th), Burmese (my), Lao (lo), Hindi (hi), Korean (ko), Japanese (ja), French (fr), and Vietnamese (vi)** into **English**, using **QLoRA** for memory‑efficient training on a single consumer GPU.

---

## Project Layout

```
qwen3_hadr_finetune/
├── finetune.py          # Main training pipeline
├── inference.py         # Load adapter and translate at inference time
├── data_prep.py         # Dataset validation, deduplication, augmentation
├── config.yaml          # All hyperparameters in one place
├── requirements.txt
└── data/
    └── hadr_dataset.jsonl   # Seed dataset (35 examples — expand this!)
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

CUDA 12.x and a GPU with ≥ 16 GB VRAM are recommended.
On 16 GB cards keep `batch_size: 2` and `use_4bit: true`.

### 2. Prepare / validate your dataset

```bash
# Validate and deduplicate
python data_prep.py --input data/hadr_dataset.jsonl \
                    --output data/hadr_ready.jsonl

# Also generate template augmentations and write train/val/test splits
python data_prep.py --input data/hadr_dataset.jsonl \
                    --augment --split \
                    --output data/hadr_augmented.jsonl
```

### 3. Configure training

Edit `config.yaml` to set the model variant, LoRA rank, learning rate, etc.

| Config key | Default | Notes |
|---|---|---|
| `model.name` | `Qwen/Qwen3-8B` | Use `Qwen3-4B` on 16 GB GPUs |
| `model.use_4bit` | `true` | Disable for A100/H100 with ≥ 40 GB VRAM |
| `lora.r` | `16` | Higher rank = more capacity but more memory |
| `training.epochs` | `5` | Increase for very small datasets |
| `training.learning_rate` | `2e-4` | Start here; lower to `5e-5` if loss is unstable |

### 4. Train

```bash
python finetune.py --config config.yaml

# Resume from a checkpoint
python finetune.py --config config.yaml \
                   --resume_from_checkpoint ./checkpoints/checkpoint-300
```

Checkpoints are saved to `./checkpoints/` every 100 steps.
The best model (highest BLEU on validation set) is saved to `./checkpoints/final/`.

### 5. Translate

**Interactive REPL:**
```bash
python inference.py --model_dir ./checkpoints/final
```

**Single sentence:**
```bash
python inference.py --model_dir ./checkpoints/final \
                    --text "Pasukan SAR sedang menjalankan operasi." \
                    --lang ms
```

**Batch file:**
```bash
python inference.py --model_dir ./checkpoints/final \
                    --input data/test_inputs.jsonl \
                    --output results/predictions.jsonl
```

---

## Dataset Format

Each line of the JSONL dataset must contain:

```json
{
  "source_lang": "ms",
  "source_text": "Pusat pemindahan sementara telah didirikan.",
  "target_text": "A temporary evacuation centre has been established."
}
```

| Field | Values | Description |
|---|---|---|
| `source_lang` | `ms` or `zh` | Source language code |
| `source_text` | string | HADR text in Malay or Chinese |
| `target_text` | string | Gold-standard English translation |

**Recommended minimum dataset size:** 500 examples per language.
The seed file (`data/hadr_dataset.jsonl`) contains 35 examples for reference.

### Expanding the dataset

Good sources for HADR terminology:
- [UN OCHA Glossary](https://www.unocha.org)
- ASEAN-ERAT training materials
- APG / NDMC operational reports (Malay)
- MCA / China MOEM official bulletins (Chinese)
- Sphere Handbook translations

---

## Model Architecture

```
Qwen3-8B (frozen base)
    └── LoRA adapters (trainable)
            r=16, α=32, dropout=0.05
            target: q/k/v/o/gate/up/down projections
```

Training uses **SFT (Supervised Fine-Tuning)** with a chat-template prompt:

```
<system>  You are an expert HADR translator...
<user>    Translate the following Malay HADR text to English:
          <source_text>
<assistant> <target_text>
```

---

## Monitoring

```bash
tensorboard --logdir ./checkpoints/runs
```

Key metrics to watch:
- `train/loss` — should decrease steadily
- `eval/bleu` — target > 40 for operational use
- `eval/chrf` — complementary character-level metric

---

## Hardware Requirements

| Setup | VRAM | Throughput |
|---|---|---|
| QLoRA 4-bit, batch=4, len=512 | ~18 GB | ~120 tokens/s on A10G |
| bfloat16 full, batch=4, len=512 | ~40 GB | ~200 tokens/s on A100 |

For multi-GPU training, prepend `torchrun --nproc_per_node=N` to the command.
