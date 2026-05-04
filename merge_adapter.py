"""
merge_adapter.py — Merge a LoRA adapter into the Qwen3 base model and save
                   a single, self-contained model directory ready for inference.

Why merge?
  After fine-tuning with QLoRA the adapter weights are stored separately from
  the base model.  Merging bakes them in so you can:
    • Run inference without PEFT / bitsandbytes installed
    • Deploy to environments that expect a standard HF model directory
    • Quantise the merged weights with llama.cpp / AutoGPTQ / AWQ

Usage:
    # Minimal — reads base model name from config.yaml
    python merge_adapter.py --adapter_dir ./checkpoints/final

    # Override base model explicitly
    python merge_adapter.py --adapter_dir ./checkpoints/final \\
                            --base_model  Qwen/Qwen3-8B

    # Custom output directory
    python merge_adapter.py --adapter_dir ./checkpoints/final \\
                            --output_dir  ./merged_model

    # Keep in bfloat16 (default) or switch to float16
    python merge_adapter.py --adapter_dir ./checkpoints/final \\
                            --dtype float16

Notes:
    • Merging requires the model to be loaded in full precision (NOT 4-bit).
      Expect ~16-32 GB RAM/VRAM for an 8B model in bfloat16.
    • If you only have a small GPU, run with --device cpu (slow but works).
    • The merged output is a plain HuggingFace model; use inference.py with
      --model_dir <output_dir> and NO --base_model flag.
"""

import json
import shutil
import logging
import argparse
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def resolve_base_model(adapter_dir: Path, cli_base_model: str | None, config_path: str | None) -> str:
    """
    Resolve the base model name in priority order:
      1. --base_model CLI argument
      2. base_model_name_or_path inside adapter_config.json  (written by PEFT)
      3. model.name inside config.yaml
    """
    if cli_base_model:
        logger.info(f"Base model (from CLI): {cli_base_model}")
        return cli_base_model

    adapter_cfg = adapter_dir / "adapter_config.json"
    if adapter_cfg.exists():
        with open(adapter_cfg, encoding="utf-8") as f:
            peft_meta = json.load(f)
        base = peft_meta.get("base_model_name_or_path")
        if base:
            logger.info(f"Base model (from adapter_config.json): {base}")
            return base

    if config_path:
        cfg_file = Path(config_path)
        if cfg_file.exists():
            with open(cfg_file, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            base = cfg.get("model", {}).get("name")
            if base:
                logger.info(f"Base model (from {config_path}): {base}")
                return base

    raise ValueError(
        "Cannot determine the base model name. "
        "Pass --base_model <name> explicitly, e.g. --base_model Qwen/Qwen3-8B"
    )


def dtype_from_str(dtype_str: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16":     torch.bfloat16,
        "float16":  torch.float16,
        "fp16":     torch.float16,
        "float32":  torch.float32,
        "fp32":     torch.float32,
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unsupported dtype '{dtype_str}'. Choose from: {list(mapping)}")
    return mapping[dtype_str]


# ─────────────────────────────────────────────
# Core Merge Logic
# ─────────────────────────────────────────────

def merge_and_save(
    base_model_name: str,
    adapter_dir: Path,
    output_dir: Path,
    dtype: torch.dtype,
    device: str,
):
    """Load base + adapter, merge, and save a standalone model."""

    # 1. Load tokenizer from the adapter directory (it was saved there by finetune.py)
    logger.info("Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(
        str(adapter_dir),
        trust_remote_code=True,
    )

    # 2. Load the base model in full precision — no quantisation during merge
    logger.info(f"Loading base model '{base_model_name}' in {dtype} …")
    logger.info("(This requires the full model weights in memory — may take a few minutes.)")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )

    # 3. Attach the LoRA adapter
    logger.info(f"Attaching LoRA adapter from '{adapter_dir}' …")
    model = PeftModel.from_pretrained(model, str(adapter_dir))

    # 4. Merge adapter weights into the base model
    logger.info("Merging adapter weights into base model …")
    model = model.merge_and_unload()
    logger.info("Merge complete.")

    # 5. Save merged model + tokenizer
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving merged model to '{output_dir}' …")
    model.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))

    # 6. Sanity-check: confirm no adapter files leaked into the output
    leaked = list(output_dir.glob("adapter_*.json")) + list(output_dir.glob("adapter_model*"))
    if leaked:
        logger.warning(
            "Unexpected adapter artefacts found in output dir — removing them: "
            + ", ".join(str(p) for p in leaked)
        )
        for p in leaked:
            p.unlink()

    logger.info("─" * 60)
    logger.info("✓ Merged model saved successfully.")
    logger.info(f"  Output directory : {output_dir.resolve()}")
    logger.info(f"  Dtype            : {dtype}")
    logger.info(
        "  To run inference : python inference.py "
        f"--model_dir {output_dir}"
    )
    logger.info("─" * 60)


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Merge a LoRA adapter into the Qwen3 base model."
    )
    parser.add_argument(
        "--adapter_dir",
        required=True,
        help="Path to the directory containing LoRA adapter weights "
             "(e.g. ./checkpoints/final).",
    )
    parser.add_argument(
        "--base_model",
        default=None,
        help="HuggingFace model name or local path for the base model "
             "(e.g. Qwen/Qwen3-8B). Auto-detected from adapter_config.json "
             "or config.yaml if omitted.",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (used as fallback to read model.name). "
             "Default: config.yaml",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Where to save the merged model. "
             "Defaults to <adapter_dir>/../merged.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"],
        help="Model dtype for the merge. Default: bfloat16.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device map for loading the base model: 'auto', 'cpu', 'cuda', "
             "'cuda:0', etc. Default: auto.",
    )
    args = parser.parse_args()

    adapter_dir = Path(args.adapter_dir).resolve()
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else adapter_dir.parent / "merged"
    )

    base_model_name = resolve_base_model(adapter_dir, args.base_model, args.config)
    dtype = dtype_from_str(args.dtype)

    logger.info("=" * 60)
    logger.info("Qwen3 HADR — LoRA Adapter Merge")
    logger.info(f"  Adapter dir : {adapter_dir}")
    logger.info(f"  Base model  : {base_model_name}")
    logger.info(f"  Output dir  : {output_dir}")
    logger.info(f"  Dtype       : {dtype}")
    logger.info(f"  Device      : {args.device}")
    logger.info("=" * 60)

    merge_and_save(
        base_model_name=base_model_name,
        adapter_dir=adapter_dir,
        output_dir=output_dir,
        dtype=dtype,
        device=args.device,
    )


if __name__ == "__main__":
    main()
