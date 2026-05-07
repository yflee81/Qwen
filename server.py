"""
server.py — Flask API server bridging hadr_translator.html to Summary.py.
            Loads Qwen3 for translation/summarisation and optionally Whisper
            for offline speech-to-text transcription.

Usage:
    # Qwen only (browser Web Speech API used for STT)
    python server.py --model_dir ./checkpoints/final

    # Qwen + Whisper (recommended for offline/production STT)
    python server.py --model_dir ./checkpoints/final --whisper medium

    # All options
    python server.py --model_dir ./checkpoints/final \
                     --base_model Qwen/Qwen3-8B \
                     --whisper medium \
                     --host 0.0.0.0 \
                     --port 5000 \
                     --max_new_tokens 256

Dependencies:
    pip install flask flask-cors
    pip install openai-whisper        # only needed if using --whisper

Whisper size guide (pick based on available VRAM alongside Qwen3-8B 4-bit ~6-8 GB):
    tiny     ~1 GB  — fastest, least accurate
    base     ~1 GB  — good for clear audio
    small    ~2 GB  — solid balance          (safe on 12 GB cards)
    medium   ~5 GB  — recommended            (safe on 16 GB cards)
    large-v3 ~10 GB — most accurate          (needs 24 GB+ alongside Qwen)

Endpoints:
    GET  /health          — liveness check, reports which models are loaded
    POST /translate       — translate HADR text (Qwen)
    POST /summarise       — extract HADR key points (Qwen)
    POST /batch_translate — translate a list of records (Qwen)
    POST /transcribe      — transcribe audio file to text (Whisper)
"""

import argparse
import logging
import os
import tempfile

from flask import Flask, request, jsonify
from flask_cors import CORS

# ── Import from Summary.py (must be in the same directory) ────
from Summary import load_model, translate, summarise_hadr, load_whisper_model, transcribe_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # allow requests from file:// and any localhost origin

# ── Globals (populated at startup) ───────────────────────────
_model          = None
_tokenizer      = None
_whisper_model  = None
_max_new_tokens = 256


# ─────────────────────────────────────────────────────────────
# Health check
# GET /health
# ─────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":         "ok",
        "qwen_loaded":    _model is not None,
        "whisper_loaded": _whisper_model is not None,
    })


# ─────────────────────────────────────────────────────────────
# Translation
# POST /translate
# Body : { "source_text": "...", "source_lang": "ms" }
# Returns: { "translation": "..." }
# ─────────────────────────────────────────────────────────────

@app.route("/translate", methods=["POST"])
def translate_endpoint():
    data        = request.get_json(force=True)
    source_text = data.get("source_text", "").strip()
    source_lang = data.get("source_lang", "ms").strip()

    if not source_text:
        return jsonify({"error": "source_text is required"}), 400

    logger.info(f"Translating [{source_lang}]: {source_text[:80]}…")
    try:
        result = translate(
            text=source_text,
            source_lang=source_lang,
            model=_model,
            tokenizer=_tokenizer,
            max_new_tokens=_max_new_tokens,
        )
        return jsonify({"translation": result})
    except Exception as e:
        logger.exception("Translation error")
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────
# Summarisation
# POST /summarise
# Body : { "text": "<English translation>" }
# Returns: HADR key-point JSON object
# ─────────────────────────────────────────────────────────────

@app.route("/summarise", methods=["POST"])
def summarise_endpoint():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "text is required"}), 400

    logger.info(f"Summarising: {text[:80]}…")
    try:
        summary = summarise_hadr(
            translated_text=text,
            model=_model,
            tokenizer=_tokenizer,
            max_new_tokens=512,
        )
        return jsonify(summary)
    except Exception as e:
        logger.exception("Summarisation error")
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────
# Batch translation
# POST /batch_translate
# Body : { "records": [{"source_text":"...","source_lang":"ms"}, ...] }
# Returns: { "results": [...], "count": N }
# ─────────────────────────────────────────────────────────────

@app.route("/batch_translate", methods=["POST"])
def batch_translate_endpoint():
    data    = request.get_json(force=True)
    records = data.get("records", [])

    if not records:
        return jsonify({"error": "records list is required"}), 400

    results = []
    for i, rec in enumerate(records):
        text = rec.get("source_text", "").strip()
        lang = rec.get("source_lang", "ms").strip()
        if not text:
            results.append({**rec, "translation": "", "error": "empty source_text"})
            continue
        try:
            translation = translate(
                text=text,
                source_lang=lang,
                model=_model,
                tokenizer=_tokenizer,
                max_new_tokens=_max_new_tokens,
            )
            results.append({**rec, "translation": translation})
            logger.info(f"  [{i+1}/{len(records)}] Translated [{lang}]")
        except Exception as e:
            logger.exception(f"Batch item {i} error")
            results.append({**rec, "translation": "", "error": str(e)})

    return jsonify({"results": results, "count": len(results)})


# ─────────────────────────────────────────────────────────────
# Whisper STT transcription
# POST /transcribe
# Form data:
#   audio       : audio file (webm, wav, mp3, m4a, ogg …)
#   source_lang : language code e.g. "ms", "zh"  (default "ms")
# Returns: { "transcript": "..." }
# ─────────────────────────────────────────────────────────────

@app.route("/transcribe", methods=["POST"])
def transcribe_endpoint():
    if _whisper_model is None:
        return jsonify({
            "error": (
                "Whisper model not loaded. "
                "Restart the server with --whisper medium (or another size)."
            )
        }), 503

    if "audio" not in request.files:
        return jsonify({"error": "No audio file — send as multipart/form-data field 'audio'"}), 400

    source_lang = request.form.get("source_lang", "ms").strip()
    audio_file  = request.files["audio"]

    # Preserve the original extension so Whisper picks the right decoder
    original_name = audio_file.filename or "recording.webm"
    suffix = os.path.splitext(original_name)[1] or ".webm"

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            audio_file.save(tmp.name)
            tmp_path = tmp.name

        logger.info(
            f"Transcribing [{source_lang}] audio: {original_name} "
            f"({os.path.getsize(tmp_path)} bytes)"
        )
        transcript = transcribe_file(tmp_path, source_lang, _whisper_model)
        logger.info(f"Transcript: {transcript[:120]}")
        return jsonify({"transcript": transcript})

    except Exception as e:
        logger.exception("Transcription error")
        return jsonify({"error": str(e)}), 500

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ─────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────

def main():
    global _model, _tokenizer, _whisper_model, _max_new_tokens

    parser = argparse.ArgumentParser(description="HADR Translator API Server")
    parser.add_argument("--model_dir",       required=True,
                        help="Path to fine-tuned Qwen model/adapter directory")
    parser.add_argument("--base_model",      default=None,
                        help="Base model name (only needed for LoRA-only adapter dirs)")
    parser.add_argument("--whisper",         default=None,
                        choices=["tiny", "base", "small", "medium", "large-v3"],
                        help="Whisper model size for STT. Recommended: medium")
    parser.add_argument("--host",            default="0.0.0.0")
    parser.add_argument("--port",            type=int, default=5000)
    parser.add_argument("--max_new_tokens",  type=int, default=256)
    args = parser.parse_args()

    _max_new_tokens = args.max_new_tokens

    logger.info("=" * 60)
    logger.info("HADR Translator API Server")
    logger.info(f"  Model dir    : {args.model_dir}")
    logger.info(f"  Whisper size : {args.whisper or 'disabled (browser STT will be used)'}")
    logger.info(f"  Host:Port    : {args.host}:{args.port}")
    logger.info(f"  Max tokens   : {_max_new_tokens}")
    logger.info("=" * 60)

    # Load Qwen first (larger, more critical)
    logger.info("Loading Qwen model… (this may take a few minutes)")
    _model, _tokenizer = load_model(args.model_dir, args.base_model)
    logger.info("✓ Qwen model loaded.")

    # Optionally load Whisper
    if args.whisper:
        logger.info(f"Loading Whisper ({args.whisper})…")
        _whisper_model = load_whisper_model(args.whisper)
        logger.info(f"✓ Whisper ({args.whisper}) loaded.")
    else:
        logger.info("Whisper not requested — /transcribe endpoint will return 503.")
        logger.info("To enable: restart with --whisper medium")

    logger.info("-" * 60)
    logger.info(f"Server ready at http://localhost:{args.port}")
    logger.info("Open hadr_translator.html in your browser.")
    logger.info("=" * 60)

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
