"""
inference.py — Load a fine-tuned Qwen3 HADR adapter and translate text,
               with optional on-demand HADR key-point summarisation.

Usage (interactive, text):
    python inference.py --model_dir ./merged_model

Usage (interactive, mic input):
    python inference.py --model_dir ./merged_model --whisper medium

Usage (batch file):
    python inference.py --model_dir ./merged_model \
                        --input data/test_inputs.jsonl \
                        --output results/predictions.jsonl

Usage (single sentence):
    python inference.py --model_dir ./merged_model \
                        --text "Operasi SAR sedang dijalankan." \
                        --lang ms

Usage (single sentence + immediate summary):
    python inference.py --model_dir ./merged_model \
                        --text "Operasi SAR sedang dijalankan." \
                        --lang ms \
                        --summarise

Usage (batch with summaries written to output):
    python inference.py --model_dir ./merged_model \
                        --input data/test_inputs.jsonl \
                        --output results/predictions.jsonl \
                        --summarise

Usage (transcribe audio file then translate):
    python inference.py --model_dir ./merged_model \
                        --whisper medium \
                        --audio recording.mp3 \
                        --lang ms

Usage (record from mic then translate):
    python inference.py --model_dir ./merged_model \
                        --whisper medium \
                        --record 15 \
                        --lang zh

Interactive summary commands
    After each translation the REPL will ask whether you want a summary.
    Type  's' / 'summary' / 'yes'  to extract HADR key points, or press
    Enter / 'n' / 'no'  to skip.
"""

import json
import argparse
import logging
import tempfile
from pathlib import Path
import re

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Whisper / audio imports — only used when --whisper is passed
try:
    import whisper
    import sounddevice as sd
    import scipy.io.wavfile as wav
    _WHISPER_AVAILABLE = True
except ImportError:
    _WHISPER_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_LANGS = ["ms", "zh", "id", "tl", "th", "my", "lo", "hi", "ko", "ja", "fr", "vi"]

SYSTEM_PROMPT = (
    "You are an expert HADR (Humanitarian Assistance and Disaster Relief) "
    "translator. Your task is to translate HADR terminology, jargon, and "
    "operational language from any of the following languages into clear, "
    "accurate English: Malay, Chinese, Indonesian, Tagalog, Burmese, Lao, "
    "Thai, Hindi, Korean, Japanese, French, or Vietnamese. "
    "Preserve technical precision and operational context. "
    "Output only the English translation without explanation."
)

# Mapping from language code to Whisper language name
WHISPER_LANG_MAP = {
    "ms": "malay",
    "zh": "chinese",
    "id": "indonesian",
    "tl": "tagalog",
    "my": "burmese",
    "lo": "lao",
    "th": "thai",
    "hi": "hindi",
    "ko": "korean",
    "ja": "japanese",
    "fr": "french",
    "vi": "vietnamese",
}


# ─────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────

def load_tokenizer(model_dir: Path):
    """Load exported Qwen tokenizers across Transformers config-shape changes."""
    try:
        return AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    except AttributeError as exc:
        if "'list' object has no attribute 'keys'" not in str(exc):
            raise
        logger.warning(
            "Tokenizer config has legacy list-style extra_special_tokens; "
            "retrying with the Transformers 4.57-compatible mapping shape."
        )
        return AutoTokenizer.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
            extra_special_tokens={},
        )


def load_model(model_dir: str, base_model: str | None = None):
    """
    Load the fine-tuned model.
    If model_dir contains a full merged model, load directly.
    If it contains LoRA adapters only, provide --base_model.
    """
    model_dir = Path(model_dir).expanduser().resolve()
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}. "
            "Pass --model_dir with the path to merged_model or your checkpoint folder."
        )

    tokenizer = load_tokenizer(model_dir)

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


# ─────────────────────────────────────────────
# Whisper Audio Transcription
# ─────────────────────────────────────────────

def load_whisper_model(size: str = "small"):
    """
    Load a Whisper model.
    Recommended sizes for a 16 GB GPU running alongside Qwen3-8B:
      'medium'   — best balance (~5 GB VRAM)   ✅ recommended
      'small'    — faster, slightly less accurate (~2 GB VRAM)
      'large-v3' — most accurate, but tight on 16 GB (~10 GB VRAM)
    """
    if not _WHISPER_AVAILABLE:
        raise ImportError(
            "Whisper dependencies not installed. Run:\n"
            "  pip install openai-whisper sounddevice scipy"
        )
    logger.info(f"Loading Whisper model: {size}")
    return whisper.load_model(size)


def transcribe_file(audio_path: str, source_lang: str, whisper_model) -> str:
    """Transcribe an audio file using Whisper."""
    lang = WHISPER_LANG_MAP.get(source_lang, source_lang)
    logger.info(f"Transcribing '{audio_path}' (language hint: {lang}) …")
    result = whisper_model.transcribe(audio_path, language=lang)
    text = result["text"].strip()
    logger.info(f"Transcription: {text}")
    return text


def record_and_transcribe(source_lang: str, whisper_model, duration: int = 10) -> str:
    """Record from the microphone for `duration` seconds, then transcribe."""
    if not _WHISPER_AVAILABLE:
        raise ImportError("sounddevice / scipy not installed.")
    sample_rate = 16_000  # Whisper expects 16 kHz
    print(f"  🎙  Recording for {duration}s … (press Ctrl+C to stop early)")
    try:
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
        )
        sd.wait()
    except KeyboardInterrupt:
        sd.stop()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav.write(tmp.name, sample_rate, audio)
        return transcribe_file(tmp.name, source_lang, whisper_model)


# ─────────────────────────────────────────────
# Translation
# ─────────────────────────────────────────────


def clean_output(text):
    # Remove <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()

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
        "vi": "Vietnamese",
    }.get(source_lang, source_lang)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Translate the following {lang_label} HADR text to English:\n\n{text}",
        },
    ]

    device = next(model.parameters()).device

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(device)

    attention_mask = (inputs["input_ids"] != tokenizer.pad_token_id).long()
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

    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **gen_kwargs,
        )

    generated = output_ids[0][inputs["input_ids"].shape[-1]:]
    raw_output = tokenizer.decode(
        generated,
        skip_special_tokens=True
    )

    cleaned_output = clean_output(raw_output)

    return cleaned_output
    

# ─────────────────────────────────────────────
# HADR Key-Point Summarisation
# ─────────────────────────────────────────────

SUMMARY_SYSTEM_PROMPT = (
    "You are an expert HADR (Humanitarian Assistance and Disaster Relief) analyst. "
    "Given translated English HADR operational text, extract and structure the key "
    "operational points into a concise summary. "
    "Always respond with a JSON object using exactly these fields — leave a field null "
    "if the information is not present in the text:\n"
    "{\n"
    '  "incident_type":      "<type of disaster / emergency>",\n'
    '  "location":           "<affected area or place names>",\n'
    '  "operational_status": "<current phase: response / recovery / assessment / standby>",\n'
    '  "key_actions":        ["<action 1>", "<action 2>", ...],\n'
    '  "resources_involved": ["<resource or unit 1>", ...],\n'
    '  "affected_population":"<number or description of people affected>",\n'
    '  "urgency_level":      "<critical / high / medium / low>",\n'
    '  "immediate_needs":    ["<need 1>", "<need 2>", ...],\n'
    '  "communication_status":"<any comms issues noted>",\n'
    '  "additional_notes":   "<anything operationally significant not captured above>"\n'
    "}\n"
    "Output ONLY the JSON object. No markdown, no explanation, no extra text."
)


def summarise_hadr(
    translated_text: str,
    model,
    tokenizer,
    max_new_tokens: int = 512,
) -> dict:
    """
    Prompt Qwen to extract structured HADR key points from an English
    translation and return them as a parsed dict.

    Parameters
    ----------
    translated_text : str
        The English translation produced by `translate()`.
    model / tokenizer :
        The loaded Qwen model and tokenizer (same objects used for translation).
    max_new_tokens : int
        Upper bound on the summary output length (default 512 is generous for
        the structured JSON).

    Returns
    -------
    dict  — parsed HADR summary fields, or a fallback dict with a 'raw' key
            containing the model output if JSON parsing fails.
    """
    messages = [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Extract HADR key points from the following translated text:\n\n"
                f"{translated_text}"
            ),
        },
    ]

    device = next(model.parameters()).device

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(device)

    attention_mask = (inputs["input_ids"] != tokenizer.pad_token_id).long()

    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"].to(device),
            attention_mask=attention_mask.to(device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = output_ids[0][inputs["input_ids"].shape[-1]:]
    raw_output = clean_output(
        tokenizer.decode(generated, skip_special_tokens=True)
    )

    # Strip markdown fences if the model wrapped the JSON anyway
    if raw_output.startswith("```"):
        raw_output = raw_output.split("```")[1]
        if raw_output.startswith("json"):
            raw_output = raw_output[4:]
        raw_output = raw_output.strip()

    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        logger.warning("Summary JSON parse failed — returning raw output.")
        return {"raw": raw_output}


def format_summary(summary: dict) -> str:
    """
    Pretty-print the structured HADR summary dict for terminal display.
    Works whether the dict is fully parsed or is the fallback 'raw' form.
    """
    if "raw" in summary:
        return f"\n  [Summary — raw output]\n  {summary['raw']}\n"

    FIELD_LABELS = [
        ("incident_type",       "Incident Type      "),
        ("location",            "Location           "),
        ("operational_status",  "Operational Status "),
        ("urgency_level",       "Urgency Level      "),
        ("affected_population", "Affected Population"),
        ("communication_status","Comms Status       "),
        ("key_actions",         "Key Actions        "),
        ("resources_involved",  "Resources Involved "),
        ("immediate_needs",     "Immediate Needs    "),
        ("additional_notes",    "Additional Notes   "),
    ]

    lines = ["\n  ── HADR Key-Point Summary ────────────────────────────"]
    for key, label in FIELD_LABELS:
        value = summary.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            if not value:
                continue
            lines.append(f"  {label} : {value[0]}")
            for item in value[1:]:
                lines.append(f"  {' ' * len(label)}   {item}")
        else:
            lines.append(f"  {label} : {value}")
    lines.append("  " + "─" * 54)
    return "\n".join(lines) + "\n"


def batch_translate(input_path: str, output_path: str, model, tokenizer, summarise: bool = False, **kwargs):
    """
    Translate all examples in a JSONL file and save predictions.
    If summarise=True, also runs HADR key-point extraction on each translation
    and stores the structured summary in the 'hadr_summary' field.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(input_path) as fin, open(output_path, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            ex = json.loads(line.strip())
            prediction = translate(
                ex["source_text"], ex["source_lang"], model, tokenizer, **kwargs
            )
            result = {**ex, "prediction": prediction}
            if summarise:
                logger.info(f"  Summarising example {i} …")
                result["hadr_summary"] = summarise_hadr(prediction, model, tokenizer)
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
    logger.info(f"Batch translation complete → {output_path}")


# ─────────────────────────────────────────────
# Interactive Loop
# ─────────────────────────────────────────────

def interactive_loop(model, tokenizer, whisper_model=None):
    mode = "text"
    if whisper_model:
        mode_input = input("Input mode — [t]ext / [f]ile / [m]ic: ").strip().lower()
        mode = {"t": "text", "f": "file", "m": "mic"}.get(mode_input, "text")

    print("\n── HADR Translator (type 'quit' to exit) ────────────────")

    while True:
        lang = input(f"Language {SUPPORTED_LANGS}: ").strip().lower()
        if lang == "quit":
            break
        if lang not in SUPPORTED_LANGS:
            print(f"  Unsupported language. Choose from: {SUPPORTED_LANGS}")
            continue

        # ── Get source text ──────────────────────────────────────
        if mode == "mic" and whisper_model:
            dur = input("  Record duration in seconds [default 10]: ").strip()
            duration = int(dur) if dur.isdigit() else 10
            text = record_and_transcribe(lang, whisper_model, duration=duration)
            print(f"  Transcribed: {text}")

        elif mode == "file" and whisper_model:
            audio_path = input("  Audio file path: ").strip()
            text = transcribe_file(audio_path, lang, whisper_model)
            print(f"  Transcribed: {text}")

        else:
            text = input("  Input text: ").strip()

        if text.lower() == "quit":
            break

        # ── Translate ─────────────────────────────────────────────
        result = translate(text, lang, model, tokenizer)
        print(f"  Translation: {result}\n")

        # ── Optional summary ──────────────────────────────────────
        want_summary = input(
            "  Summary? [s/summary/yes to extract HADR key points, Enter to skip]: "
        ).strip().lower()
        if want_summary in ("s", "summary", "yes", "y"):
            print("  Extracting HADR key points …")
            summary = summarise_hadr(result, model, tokenizer)
            print(format_summary(summary))



# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HADR Translation Inference")
    parser.add_argument("--model_dir",      default="./merged_model",  help="Path to merged model/adapter")
    parser.add_argument("--base_model",     default=None,   help="Base model name (if using LoRA-only dir)")
    parser.add_argument("--input",          default=None,   help="Input JSONL for batch translation")
    parser.add_argument("--output",         default="results/predictions.jsonl")
    parser.add_argument("--text",           default=None,   help="Single text string to translate")
    parser.add_argument("--lang",           default="ms",   choices=SUPPORTED_LANGS)
    parser.add_argument("--max_new_tokens", type=int,       default=256)
    parser.add_argument("--temperature",    type=float,     default=0.0)
    # Whisper args
    parser.add_argument(
        "--whisper",
        default=None,
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Enable Whisper transcription. Recommended: medium (leaves ~8 GB for Qwen3-8B on 16 GB GPU)",
    )
    parser.add_argument("--audio",  default=None, help="Path to audio file to transcribe then translate")
    parser.add_argument("--record", type=int, default=None, metavar="SECONDS",
                        help="Record N seconds from microphone then translate")
    parser.add_argument(
        "--summarise", "--summarize",
        action="store_true",
        dest="summarise",
        help=(
            "After translating, run HADR key-point extraction via Qwen and print / "
            "store a structured summary. In interactive mode the REPL prompts you "
            "after every translation; this flag is only needed for --text / --input "
            "modes where you want summarisation applied automatically."
        ),
    )
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_dir, args.base_model)

    whisper_model = load_whisper_model(args.whisper) if args.whisper else None

    if args.audio and whisper_model:
        text = transcribe_file(args.audio, args.lang, whisper_model)
        print(f"Transcription: {text}")
        result = translate(text, args.lang, model, tokenizer,
                           args.max_new_tokens, args.temperature)
        print(f"Translation:   {result}")
        if args.summarise:
            summary = summarise_hadr(result, model, tokenizer)
            print(format_summary(summary))

    elif args.record and whisper_model:
        text = record_and_transcribe(args.lang, whisper_model, duration=args.record)
        print(f"Transcription: {text}")
        result = translate(text, args.lang, model, tokenizer,
                           args.max_new_tokens, args.temperature)
        print(f"Translation:   {result}")
        if args.summarise:
            summary = summarise_hadr(result, model, tokenizer)
            print(format_summary(summary))

    elif args.text:
        result = translate(args.text, args.lang, model, tokenizer,
                           args.max_new_tokens, args.temperature)
        print(f"\nTranslation:\n{result}\n")
        if args.summarise:
            summary = summarise_hadr(result, model, tokenizer)
            print(format_summary(summary))

    elif args.input:
        batch_translate(args.input, args.output, model, tokenizer,
                        summarise=args.summarise,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature)

    else:
        interactive_loop(model, tokenizer, whisper_model)
