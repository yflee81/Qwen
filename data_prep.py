"""
data_prep.py — Prepare and augment the HADR translation dataset.

Features:
  - Validate and deduplicate a JSONL dataset
  - Generate paraphrased variations using Claude API (optional)
  - Split into train / val / test sets
  - Report language distribution statistics

Usage:
    python data_prep.py --input data/hadr_dataset.jsonl --output data/hadr_ready.jsonl
    python data_prep.py --input data/hadr_dataset.jsonl --augment --output data/hadr_augmented.jsonl
"""

import json
import argparse
import logging
import random
from pathlib import Path
from collections import Counter
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REQUIRED_KEYS = {"source_lang", "source_text", "target_text"}
SUPPORTED_LANGS = {"ms", "zh", "id", "tl", "my", "lo", "th", "hi", "ko", "vi", "ja", "fr"}


# ─────────────────────────────────────────────
# Validation & Deduplication
# ─────────────────────────────────────────────

def load_jsonl(path: str) -> List[Dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Line {i}: JSON parse error — {e}")
    return records


def validate(records: List[Dict]) -> List[Dict]:
    valid = []
    for i, ex in enumerate(records):
        if not REQUIRED_KEYS.issubset(ex.keys()):
            logger.warning(f"[{i}] Missing keys: {REQUIRED_KEYS - ex.keys()}")
            continue
        if ex["source_lang"] not in SUPPORTED_LANGS:
            logger.warning(f"[{i}] Unsupported lang: {ex['source_lang']}")
            continue
        if not ex["source_text"].strip() or not ex["target_text"].strip():
            logger.warning(f"[{i}] Empty text field — skipping.")
            continue
        valid.append(ex)
    logger.info(f"Validation: {len(valid)}/{len(records)} examples passed.")
    return valid


def deduplicate(records: List[Dict]) -> List[Dict]:
    seen = set()
    unique = []
    for ex in records:
        key = (ex["source_lang"], ex["source_text"].strip())
        if key not in seen:
            seen.add(key)
            unique.append(ex)
    removed = len(records) - len(unique)
    if removed:
        logger.info(f"Deduplication: removed {removed} duplicate examples.")
    return unique


# ─────────────────────────────────────────────
# Dataset Statistics
# ─────────────────────────────────────────────

def print_stats(records: List[Dict]):
    lang_counts = Counter(ex["source_lang"] for ex in records)
    src_lengths = [len(ex["source_text"].split()) for ex in records]
    tgt_lengths = [len(ex["target_text"].split()) for ex in records]

    print("\n── Dataset Statistics ──────────────────────")
    print(f"  Total examples : {len(records)}")
    for lang, count in lang_counts.items():
        label = {
            "ms": "Malay",
            "zh": "Chinese",
            "id": "Indonesian",
            "tl": "Filipino",
            "th": "Thai",
            "my": "Burmese",
            "lo": "Lao",
            "hi": "Hindi",
            "ko": "Korean",
            "ja": "Japanese",
            "fr": "French",
            "vi": "Vietnamese"
        }.get(lang, lang)
        print(f"  {label:10s}     : {count} ({count/len(records)*100:.1f}%)")
    print(f"  Src len (words): avg={sum(src_lengths)/len(src_lengths):.1f}  "
          f"max={max(src_lengths)}  min={min(src_lengths)}")
    print(f"  Tgt len (words): avg={sum(tgt_lengths)/len(tgt_lengths):.1f}  "
          f"max={max(tgt_lengths)}  min={min(tgt_lengths)}")
    print("─────────────────────────────────────────────\n")


# ─────────────────────────────────────────────
# Data Augmentation (template-based)
# ─────────────────────────────────────────────

AUGMENT_TEMPLATES_MS = [
    "Sila terjemahkan: {text}",
    "Apakah maksud '{text}' dalam konteks HADR?",
    "{text}",
]

AUGMENT_TEMPLATES_ZH = [
    "请翻译以下内容：{text}",
    "在救灾语境中，'{text}'是什么意思？",
    "{text}",
]

def augment_example(ex: Dict) -> List[Dict]:
    """Create light template variants of an example to boost dataset diversity."""
    augmented = [ex]  # always keep original
    lang = ex["source_lang"]
    templates = AUGMENT_TEMPLATES_MS if lang == "ms" else AUGMENT_TEMPLATES_ZH

    for tmpl in templates[1:]:   # skip the identity template
        new_src = tmpl.format(text=ex["source_text"])
        augmented.append({
            "source_lang": lang,
            "source_text": new_src,
            "target_text": ex["target_text"],
        })
    return augmented


def augment_dataset(records: List[Dict]) -> List[Dict]:
    augmented = []
    for ex in records:
        augmented.extend(augment_example(ex))
    logger.info(f"Augmentation: {len(records)} → {len(augmented)} examples.")
    return augmented


# ─────────────────────────────────────────────
# Train / Val / Test Split
# ─────────────────────────────────────────────

def split_dataset(
    records: List[Dict],
    train_ratio: float = 0.85,
    val_ratio:   float = 0.075,
    seed: int = 42,
) -> Dict[str, List[Dict]]:
    random.seed(seed)
    data = records.copy()
    random.shuffle(data)
    n = len(data)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    return {
        "train": data[:n_train],
        "validation": data[n_train: n_train + n_val],
        "test": data[n_train + n_val:],
    }


def save_jsonl(records: List[Dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in records:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(records)} examples → {path}")


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare HADR dataset for fine-tuning")
    parser.add_argument("--input",   required=True,  help="Path to raw JSONL dataset")
    parser.add_argument("--output",  required=True,  help="Output JSONL path (cleaned)")
    parser.add_argument("--augment", action="store_true", help="Apply template augmentation")
    parser.add_argument("--split",   action="store_true", help="Write train/val/test splits")
    args = parser.parse_args()

    records = load_jsonl(args.input)
    records = validate(records)
    records = deduplicate(records)

    if args.augment:
        records = augment_dataset(records)

    print_stats(records)
    save_jsonl(records, args.output)

    if args.split:
        splits = split_dataset(records)
        base = Path(args.output).stem
        out_dir = Path(args.output).parent
        for split_name, split_data in splits.items():
            save_jsonl(split_data, str(out_dir / f"{base}_{split_name}.jsonl"))
            logger.info(f"  {split_name}: {len(split_data)} examples")


if __name__ == "__main__":
    main()
