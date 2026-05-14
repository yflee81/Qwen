from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    TrainingArguments,
    Trainer
)

import argparse
from pathlib import Path

from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from jiwer import wer
import torch
import soundfile as sf
import numpy as np


def parse_args():
    script_dir = Path(__file__).resolve().parent
    default_train_file = script_dir / "aligned_segments" / "train.jsonl"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-file",
        default=str(default_train_file),
        help="Path to the JSON/JSONL manifest produced by audio_segment.py",
    )
    parser.add_argument(
        "--output-dir",
        default=str(script_dir / "hadr-whisper-lora"),
        help="Directory for checkpoints and the final LoRA adapter",
    )
    return parser.parse_args()


args = parse_args()
train_file = Path(args.train_file).resolve()

if not train_file.exists():
    raise FileNotFoundError(
        f"Training manifest not found: {train_file}\n"
        "Run audio_segment.py first, or pass --train-file path\\to\\train.jsonl"
    )

manifest_dir = train_file.parent

# ----------------------------
# Device setup (GPU FIX)
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------------------
# Model + Processor
# ----------------------------
model_name = "openai/whisper-small"

processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# ----------------------------
# LoRA setup
# ----------------------------
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# MOVE MODEL TO GPU
model.to(device)

# ----------------------------
# Dataset loading + split
# ----------------------------
dataset = load_dataset("json", data_files=str(train_file))
dataset = dataset["train"].train_test_split(test_size=0.1)

train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# ----------------------------
# Preprocessing (FIXED - no torchcodec)
# ----------------------------
def preprocess(batch):
    audio_path = Path(batch["audio"])
    if not audio_path.is_absolute():
        audio_path = manifest_dir / audio_path

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio clip not found: {audio_path}")

    # Load waveform safely (NO datasets Audio / NO torchcodec)
    audio, sr = sf.read(audio_path)

    if sr != 16000:
        raise ValueError(f"Expected 16000 Hz audio, got {sr} Hz for {audio_path}")

    # Convert to Whisper features
    inputs = processor.feature_extractor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    )

    # Tokenize text labels
    labels = processor.tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=448
    ).input_ids

    return {
        "input_features": inputs.input_features[0],
        "labels": labels
    }

# Apply preprocessing
train_dataset = train_dataset.map(preprocess)
eval_dataset = eval_dataset.map(preprocess)

# ----------------------------
# WER metric
# ----------------------------
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # pred_ids may be a tuple (logits, ...) from some Whisper versions — unwrap it
    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]

    # Convert logits to token ids if needed (argmax over vocab dimension)
    if pred_ids.ndim == 3:
        pred_ids = np.argmax(pred_ids, axis=-1)

    # Replace -100 padding so the tokenizer can decode cleanly
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # Convert numpy arrays → plain Python lists (fixes the ambiguous truth value error)
    pred_ids  = pred_ids.tolist()
    label_ids = label_ids.tolist()

    pred_str  = processor.tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    pred_str  = [s.lower().strip() for s in pred_str]
    label_str = [s.lower().strip() for s in label_str]

    word_error_rate = wer(label_str, pred_str)
    return {"wer": round(word_error_rate, 4)}

# ----------------------------
# Training setup
# ----------------------------
training_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
    num_train_epochs=5,
    fp16=True,
    logging_steps=10,
    save_steps=200,
    eval_strategy="steps",
    eval_steps=200,                    # evaluate every 200 steps
    load_best_model_at_end=True,       # keep the checkpoint with lowest WER
    metric_for_best_model="wer",
    greater_is_better=False,           # lower WER = better
    dataloader_pin_memory=True if device.type == "cuda" else False
)

# ----------------------------
# Trainer
# ----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,   # plug in WER
)

trainer.train()

# ----------------------------
# Final WER on eval set
# ----------------------------
metrics = trainer.evaluate()
print(f"\nFinal eval WER: {metrics['eval_wer']:.2%}")

# ----------------------------
# Save model
# ----------------------------
model.save_pretrained(args.output_dir)
processor.save_pretrained(args.output_dir)
