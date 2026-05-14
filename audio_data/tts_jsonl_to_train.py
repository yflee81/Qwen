r"""
Create a Whisper-style training JSONL from a JSONL file by synthesizing each
source_text sentence to one aligned audio file.

Examples:
  python tts_jsonl_to_train.py --input C:\Users\UserAdmin\Downloads\files\data_collection\my.jsonl
  python tts_jsonl_to_train.py --input my.jsonl --provider edge --voice ms-MY-OsmanNeural
"""

import argparse
import asyncio
import json
import re
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path


EDGE_VOICE_BY_LANG = {
    "en": "en-US-JennyNeural",
    "ms": "ms-MY-OsmanNeural",
    "id": "id-ID-ArdiNeural",
    "zh": "zh-CN-YunxiNeural",
    "zh-cn": "zh-CN-YunxiNeural",
    "zh-tw": "zh-TW-YunJheNeural",
}


def parse_args():
    script_dir = Path(__file__).resolve().parent
    default_out_dir = script_dir / "tts_segments"

    parser = argparse.ArgumentParser(
        description="Generate TTS audio and a train.jsonl manifest from source_text rows."
    )
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output-dir", default=str(default_out_dir))
    parser.add_argument("--manifest-name", default="train1.jsonl")
    parser.add_argument(
        "--provider",
        choices=["sapi", "edge"],
        default="sapi",
        help="sapi uses Windows built-in TTS; edge uses the optional edge-tts package.",
    )
    parser.add_argument("--voice", help="Voice name. For edge, e.g. ms-MY-OsmanNeural.")
    parser.add_argument("--rate", default="+0%", help="edge-tts rate, e.g. +0%%, -10%%, +15%%")
    parser.add_argument("--volume", default="+0%", help="edge-tts volume, e.g. +0%%, -10%%")
    parser.add_argument("--source-field", default="source_text")
    parser.add_argument("--target-field", default="target_text")
    parser.add_argument("--language-field", default="source_lang")
    parser.add_argument("--audio-dir-name", default="audio")
    parser.add_argument("--prefix", default=None, help="Audio filename prefix; defaults to input stem")
    parser.add_argument("--limit", type=int, help="Only process the first N synthesized sentences")
    parser.add_argument("--no-split", action="store_true", help="Treat each source_text as one item")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing audio/manifest")
    parser.add_argument("--dry-run", action="store_true", help="Read and split input without TTS output")
    return parser.parse_args()


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
    return rows


def split_sentences(text):
    text = re.sub(r"\s+", " ", str(text).strip())
    if not text:
        return []

    parts = re.split(r"(?<=[。！？!?])\s+|(?<=[。！？!?])|(?<!\b[A-Z])(?<=[.!?])\s+", text)
    return [part.strip() for part in parts if part.strip()]


def build_items(rows, args):
    items = []
    for row_index, row in enumerate(rows, start=1):
        source_text = row.get(args.source_field)
        if not source_text:
            continue

        sentences = [str(source_text).strip()] if args.no_split else split_sentences(source_text)
        for sentence_index, sentence in enumerate(sentences, start=1):
            item = {
                "row_index": row_index,
                "sentence_index": sentence_index,
                "text": sentence,
                "source_text": sentence,
                "target_text": row.get(args.target_field, ""),
                "language": row.get(args.language_field, ""),
                "source_row": row,
            }
            items.append(item)
            if args.limit and len(items) >= args.limit:
                return items
    return items


def ensure_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to convert TTS output to 16 kHz mono WAV.")


def convert_to_training_wav(input_path, output_path):
    temp_output = output_path.with_suffix(".tmp.wav")
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-sample_fmt",
        "s16",
        str(temp_output),
    ]
    subprocess.run(command, check=True)
    temp_output.replace(output_path)


def wav_duration(path):
    with wave.open(str(path), "rb") as handle:
        frames = handle.getnframes()
        rate = handle.getframerate()
        return round(frames / float(rate), 3)


def synthesize_sapi(text, output_path, voice=None):
    with tempfile.TemporaryDirectory() as temp_dir:
        text_path = Path(temp_dir) / "text.txt"
        raw_wav = Path(temp_dir) / "raw.wav"
        script_path = Path(temp_dir) / "speak.ps1"
        text_path.write_text(text, encoding="utf-8")

        script_path.write_text(
            "\n".join(
                [
                    "param([string]$TextPath, [string]$OutPath, [string]$Voice)",
                    "Add-Type -AssemblyName System.Speech",
                    "$text = Get-Content -LiteralPath $TextPath -Raw -Encoding UTF8",
                    "$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer",
                    "if ($Voice) { $synth.SelectVoice($Voice) }",
                    "$synth.SetOutputToWaveFile($OutPath)",
                    "$synth.Speak($text)",
                    "$synth.Dispose()",
                ]
            ),
            encoding="utf-8",
        )
        command = [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script_path),
            str(text_path),
            str(raw_wav),
            voice or "",
        ]
        subprocess.run(command, check=True)
        convert_to_training_wav(raw_wav, output_path)


async def synthesize_edge(text, output_path, voice, rate, volume):
    try:
        import edge_tts
    except ImportError as exc:
        raise RuntimeError(
            "edge-tts is not installed. Install it with: pip install edge-tts"
        ) from exc

    with tempfile.TemporaryDirectory() as temp_dir:
        mp3_path = Path(temp_dir) / "raw.mp3"
        communicate = edge_tts.Communicate(text, voice=voice, rate=rate, volume=volume)
        await communicate.save(str(mp3_path))
        convert_to_training_wav(mp3_path, output_path)


def choose_voice(provider, language, requested_voice):
    if requested_voice:
        return requested_voice
    if provider == "edge":
        return EDGE_VOICE_BY_LANG.get(str(language).lower(), EDGE_VOICE_BY_LANG["en"])
    return None


def write_manifest_row(handle, row):
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    audio_dir = output_dir / args.audio_dir_name
    manifest_path = output_dir / args.manifest_name
    prefix = args.prefix or input_path.stem

    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")

    rows = read_jsonl(input_path)
    items = build_items(rows, args)

    print(f"Loaded {len(rows)} JSONL rows and prepared {len(items)} sentence item(s).")
    if args.dry_run:
        for item in items[:5]:
            print(f"{item['language']}: {item['text']}")
        return

    ensure_ffmpeg()
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Manifest already exists: {manifest_path}. Pass --overwrite to replace it."
        )

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for index, item in enumerate(items, start=1):
            clip_name = f"{prefix}_{index:05d}.wav"
            clip_path = audio_dir / clip_name
            language = item["language"]
            voice = choose_voice(args.provider, language, args.voice)

            if clip_path.exists() and not args.overwrite:
                print(f"Skipping existing audio: {clip_path}")
            else:
                if args.provider == "edge":
                    asyncio.run(
                        synthesize_edge(
                            item["text"],
                            clip_path,
                            voice,
                            args.rate,
                            args.volume,
                        )
                    )
                else:
                    synthesize_sapi(item["text"], clip_path, voice)

            duration = wav_duration(clip_path)
            manifest_row = {
                "audio": str(Path(args.audio_dir_name) / clip_name),
                "text": item["text"],
                "start": 0.0,
                "end": duration,
                "duration": duration,
                "language": language,
                "detected_language": language,
                "task": "transcribe",
                "alignment": "tts_sentence",
                "match_ratio": 1.0,
                "source_text": item["source_text"],
                "target_text": item["target_text"],
                "source_row_index": item["row_index"],
                "sentence_index": item["sentence_index"],
            }
            write_manifest_row(manifest, manifest_row)
            print(f"{index}/{len(items)} wrote {clip_name} ({duration}s)")

    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote audio directory: {audio_dir}")


if __name__ == "__main__":
    main()
