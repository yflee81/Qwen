"""
inference.py — Load a fine-tuned Qwen3 HADR adapter and translate text.

Usage (interactive):
    python inference.py --model_dir ./checkpoints/final

Usage (batch file):
    python inference.py --model_dir ./checkpoints/final \
                        --input data/test_inputs.jsonl \
                        --output results/predictions.jsonl

Usage (single sentence):
    python inference.py --model_dir ./checkpoints/final \
                        --text "Operasi SAR sedang dijalankan." \
                        --lang ms
"""

import json
import argparse
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert HADR (Humanitarian Assistance and Disaster Relief) "
    "translator. Your task is to translate HADR terminology, jargon, and "
    "operational language from Malay or Chinese into clear, accurate English. "
    "Preserve technical precision and operational context. "
    "Output only the English translation without explanation."
)


def load_model(model_dir: str, base_model: str | None = None):
    """
    Load the fine-tuned model.
    If model_dir contains a full merged model, load directly.
    If it contains LoRA adapters only, provide --base_model.
    """
    model_dir = Path(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    adapter_config = model_dir / "adapter_config.json"
    if adapter_config.exists() and base_model:
        logger.info(f"Loading base model '{base_model}' + LoRA adapter from '{model_dir}'")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, str(model_dir))
        model = model.to("cuda")
    else:
        logger.info(f"Loading full model from '{model_dir}'")
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            trust_remote_code=True,
        )

    model.eval()
    return model, tokenizer


def translate(
    text: str,
    source_lang: str,
    model,
    tokenizer,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
) -> str:
    lang_label = {
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
    }.get(source_lang, source_lang)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Translate the following {lang_label} HADR text to English:\n\n{text}",
        },
    ]
    
    device = next(model.parameters()).device
    
    # 1. Apply the template
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True, # Explicitly ask for a dict
    ).to(device)
    
    attention_mask = (inputs["input_ids"] != tokenizer.pad_token_id).long()

    # Move to device
    inputs = {
        "input_ids": inputs["input_ids"].to(device),
        "attention_mask": attention_mask.to(device),
    }

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    if temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
    else:
        gen_kwargs["do_sample"] = False

    # 2. Extract ONLY input_ids for model.generate
    # We use inputs["input_ids"] to avoid the 'shape' AttributeError
    with torch.no_grad():
        output_ids = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], **gen_kwargs)

    # 3. Correctly slice the output
    generated = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()

def batch_translate(input_path: str, output_path: str, model, tokenizer, **kwargs):
    """Translate all examples in a JSONL file and save predictions."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(input_path) as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            ex = json.loads(line.strip())
            prediction = translate(
                ex["source_text"], ex["source_lang"], model, tokenizer, **kwargs
            )
            result = {**ex, "prediction": prediction}
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
    logger.info(f"Batch translation complete → {output_path}")


def interactive_loop(model, tokenizer):
    print("\n── HADR Translator (type 'quit' to exit) ────────────────")
    while True:
        lang = input("Language [ms/zh/id/tl/th/my/lo/hi/ko/ja/fr/vi]: ").strip().lower()
        if lang == "quit":
            break
        if lang not in ("ms", "zh", "id", "tl", "th", "my", "lo", "hi", "ko", "ja", "fr", "vi"):
            print("Please enter a supported language code.")
            continue
        text = input("Input text: ").strip()
        if text.lower() == "quit":
            break
        result = translate(text, lang, model, tokenizer)
        print(f"Translation: {result}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HADR Translation Inference")
    parser.add_argument("--model_dir",   required=True, help="Path to fine-tuned model/adapter")
    parser.add_argument("--base_model",  default=None,  help="Base model name (if using LoRA-only dir)")
    parser.add_argument("--input",       default=None,  help="Input JSONL for batch translation")
    parser.add_argument("--output",      default="results/predictions.jsonl")
    parser.add_argument("--text",        default=None,  help="Single text to translate")
    parser.add_argument("--lang", default="ms", choices=["ms", "zh", "id", "tl", "th", "my", "lo", "hi", "ko", "ja", "fr", "vi"])
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_dir, args.base_model)

    if args.text:
        result = translate(args.text, args.lang, model, tokenizer,
                           args.max_new_tokens, args.temperature)
        print(f"\nTranslation:\n{result}\n")
    elif args.input:
        batch_translate(args.input, args.output, model, tokenizer,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature)
    else:
        interactive_loop(model, tokenizer)
