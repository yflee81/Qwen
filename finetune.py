"""
Fine-tune Qwen3 for HADR (Humanitarian Assistance and Disaster Relief)
jargon translation from Malay/Chinese to English.

Usage:
    python finetune.py --config config.yaml
    python finetune.py --config config.yaml --resume_from_checkpoint ./checkpoints/checkpoint-500
"""

import os
import json
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import torch
import yaml
from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)

# Monkey-patch Path.read_text to always use UTF-8
_original_read_text = Path.read_text
def _read_text_utf8(self, *args, **kwargs):
    if "encoding" not in kwargs:
        kwargs["encoding"] = "utf-8"
    return _original_read_text(self, *args, **kwargs)

Path.read_text = _read_text_utf8

from trl import SFTTrainer, SFTConfig
import evaluate
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1.  Prompt Templates
# ─────────────────────────────────────────────
SUPPORTED_LANGS = {
    "ms":  "Malay",
    "zh":  "Chinese",
    "id":  "Indonesian",
    "tl":  "Tagalog",
    "my":  "Burmese",
    "lo":  "Lao",
    "th":  "Thai",
    "hi":  "Hindi",
    "ko":  "Korean",
    "ja":  "Japanese",
    "fr":  "French",
    "vi":  "Vietnamese",
}

SYSTEM_PROMPT = (
    "You are an expert HADR (Humanitarian Assistance and Disaster Relief) "
    "translator. Your task is to translate HADR terminology, jargon, and "
    "operational language from any of the following languages into clear, "
    "accurate English: Malay, Chinese, Indonesian, Tagalog, Burmese, Lao, "
    "Thai, Hindi, Korean, Japanese, French, or Vietnamese. "
    "Preserve technical precision and operational context. "
    "Output only the English translation without explanation."
)

def build_prompt(source_lang: str, source_text: str) -> str:
    """Build the user turn for a translation example."""
    lang_label = SUPPORTED_LANGS.get(source_lang, source_lang)
    return f"Translate the following {lang_label} HADR text to English:\n\n{source_text}"

def format_example(example: dict, tokenizer) -> dict:
    """
    Format a single dataset example into a chat-template message.
    Expected keys: source_lang, source_text, target_text
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": build_prompt(example["source_lang"], example["source_text"])},
        {"role": "assistant", "content": example["target_text"]},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


# ─────────────────────────────────────────────
# 2.  Dataset Loading & Validation
# ─────────────────────────────────────────────

REQUIRED_KEYS = {"source_lang", "source_text", "target_text"}

def validate_example(ex: dict, idx: int) -> bool:
    if not REQUIRED_KEYS.issubset(ex.keys()):
        logger.warning(f"Example {idx} missing keys: {REQUIRED_KEYS - ex.keys()}")
        return False
    if ex["source_lang"] not in SUPPORTED_LANGS:
        logger.warning(f"Example {idx} has unsupported source_lang: {ex['source_lang']}")
        return False
    if not ex["source_text"].strip() or not ex["target_text"].strip():
        logger.warning(f"Example {idx} has empty text fields.")
        return False
    return True

def load_hadr_dataset(data_path: str) -> DatasetDict:
    """
    Load dataset from a JSON / JSONL file or a HuggingFace dataset path.
    Each record must have: source_lang, source_text, target_text.
    """
    path = Path(data_path)

    if path.exists():
        # Local file
        if path.suffix in (".jsonl",):
            with open(path, encoding="utf-8") as f:
                records = [json.loads(line) for line in f if line.strip()]
        elif path.suffix == ".json":
            with open(path) as f:
                records = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}. Use .json or .jsonl")

        valid = [ex for i, ex in enumerate(records) if validate_example(ex, i)]
        logger.info(f"Loaded {len(valid)}/{len(records)} valid examples from {data_path}")
        dataset = Dataset.from_list(valid)

        # 90/5/5 split if not pre-split
        split = dataset.train_test_split(test_size=0.1, seed=42)
        val_test = split["test"].train_test_split(test_size=0.5, seed=42)
        return DatasetDict({
            "train": split["train"],
            "validation": val_test["train"],
            "test": val_test["test"],
        })
    else:
        # Try as HuggingFace dataset name
        logger.info(f"Loading HuggingFace dataset: {data_path}")
        return load_dataset(data_path)


# ─────────────────────────────────────────────
# 3.  Model & LoRA Setup
# ─────────────────────────────────────────────

def load_model_and_tokenizer(model_name: str, use_4bit: bool = True):
    """Load Qwen3 model with optional 4-bit quantisation for memory efficiency."""
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",   # required for SFT loss masking
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading model: {model_name} (4-bit={use_4bit})")
    if use_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    return model, tokenizer


def apply_lora(model, lora_config: dict):
    """Wrap the model with LoRA adapters."""
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_config.get("r", 16),
        lora_alpha=lora_config.get("alpha", 32),
        lora_dropout=lora_config.get("dropout", 0.05),
        bias="none",
        target_modules=lora_config.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj",
             "gate_proj", "up_proj", "down_proj"],
        ),
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


# ─────────────────────────────────────────────
# 4.  Evaluation Helpers
# ─────────────────────────────────────────────

def build_compute_metrics(tokenizer):
    """Returns a compute_metrics function using sacreBLEU."""
    bleu = evaluate.load("sacrebleu")
    bertscore = evaluate.load("bertscore")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.argmax(preds, axis=-1)

        # Replace -100 (masked) with pad token id
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds  = tokenizer.batch_decode(preds,   skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels,  skip_special_tokens=True)

        # sacreBLEU expects list of references wrapped in a list
        bleu_result = bleu.compute(
            predictions=decoded_preds,
            references=[[l] for l in decoded_labels],
        )
        bertscore_result = bertscore.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            lang="en", 
        )
        return {
            "bleu":  round(bleu_result["score"], 4),
            "bertscore_f1":  round(float(np.mean(bertscore_result["f1"])), 4),
        }

    return compute_metrics

# ─────────────────────────────────────────────
# 5.  Main Training Pipeline
# ─────────────────────────────────────────────

def train(cfg: dict, resume_checkpoint: Optional[str] = None):
    # — Model
    model, tokenizer = load_model_and_tokenizer(
        cfg["model"]["name"],
        use_4bit=cfg["model"].get("use_4bit", True),
    )

    # — LoRA
    if cfg.get("lora", {}).get("enabled", True):
        model = apply_lora(model, cfg.get("lora", {}))

    # — Dataset
    raw_ds = load_hadr_dataset(cfg["data"]["path"])
    logger.info(f"Dataset splits: { {k: len(v) for k, v in raw_ds.items()} }")

    # — Tokenise / format
    max_length = cfg["data"].get("max_length", 512)

    def preprocess(batch):
        return format_example(batch, tokenizer)

    tokenised = raw_ds.map(preprocess, remove_columns=raw_ds["train"].column_names)

    # — SFT Training Arguments
    training_cfg = cfg.get("training", {})
    output_dir = training_cfg.get("output_dir", "./checkpoints")

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=training_cfg.get("epochs", 3),
        per_device_train_batch_size=training_cfg.get("batch_size", 4),
        per_device_eval_batch_size=training_cfg.get("eval_batch_size", 4),
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=training_cfg.get("learning_rate", 2e-4),
        lr_scheduler_type=training_cfg.get("lr_scheduler", "cosine"),
        warmup_ratio=training_cfg.get("warmup_ratio", 0.05),
        weight_decay=training_cfg.get("weight_decay", 0.01),
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=training_cfg.get("logging_steps", 10),
        eval_strategy="steps",
        eval_steps=training_cfg.get("eval_steps", 100),
        save_strategy="steps",
        save_steps=training_cfg.get("save_steps", 100),
        save_total_limit=training_cfg.get("save_total_limit", 3),
        load_best_model_at_end=True,
        metric_for_best_model="eval_bertscore_f1",
        greater_is_better=True,
        report_to=training_cfg.get("report_to", "tensorboard"),
        run_name=training_cfg.get("run_name", "qwen3-hadr-translation"),
        dataset_text_field="text",
        packing=training_cfg.get("packing", False),
        max_length=max_length,
    )

    # — Trainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=tokenised["train"],
        eval_dataset=tokenised["validation"],
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        compute_metrics=build_compute_metrics(tokenizer),
    )

    # — Train
    logger.info("Starting training…")
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # — Save final adapter
    final_dir = Path(output_dir) / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info(f"Saved fine-tuned model to {final_dir}")

    # — Evaluate on test set
    logger.info("Running evaluation on test set…")
    results = evaluate_on_test(
        model=trainer.model,
        tokenizer=tokenizer,
        test_dataset=raw_ds["test"],
        max_new_tokens=cfg["inference"].get("max_new_tokens", 256),
    )
    logger.info(f"Test results: {results}")

    results_path = Path(output_dir) / "test_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Test results saved to {results_path}")


# ─────────────────────────────────────────────
# 6.  Test-time Evaluation
# ─────────────────────────────────────────────

def evaluate_on_test(model, tokenizer, test_dataset: Dataset, max_new_tokens: int = 256):
    """Run greedy decoding on the test set and compute BLEU / bertscore."""
    bleu = evaluate.load("sacrebleu")
    bertscore = evaluate.load("bertscore")
    model.eval()
    predictions, references = [], []

    device = next(model.parameters()).device

    for example in test_dataset:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_prompt(example["source_lang"], example["source_text"])},
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if isinstance(inputs, torch.Tensor):
            input_ids = inputs.to(device)
            attention_mask = torch.ones_like(input_ids).to(device)
        else:
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(device)


        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                attention_mask=attention_mask,
            )

        # Slice off the prompt tokens
        generated = output_ids[0][input_ids.shape[-1]:]
        pred = tokenizer.decode(generated, skip_special_tokens=True).strip()
        predictions.append(pred)
        references.append(example["target_text"])

    bleu_score = bleu.compute(predictions=predictions, references=[[r] for r in references])
    bertscore_result = bertscore.compute(predictions=predictions, references=references, lang="en")

    return {
        "bleu": round(bleu_score["score"], 4),
        "bertscore_f1": round(float(np.mean(bertscore_result["f1"])), 4),
        "n_examples": len(predictions),
    }


# ─────────────────────────────────────────────
# 7.  Inference Helper
# ─────────────────────────────────────────────

def translate(text: str, source_lang: str, model, tokenizer, max_new_tokens: int = 256) -> str:
    """Translate a single HADR sentence at inference time."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": build_prompt(source_lang, text)},
    ]
    device = next(model.parameters()).device
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = output_ids[0][input_ids.shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ─────────────────────────────────────────────
# 8.  Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3 for HADR translation")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg, resume_checkpoint=args.resume_from_checkpoint)
