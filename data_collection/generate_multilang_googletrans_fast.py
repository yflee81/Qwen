#!/usr/bin/env python3
"""
Fast multilingual JSONL generator using Google Translate via googletrans.

Input:  plain text file with one English sentence per line
Output: JSONL records in this format:
        {"source_lang":"lo","source_text":"...translated text...","target_text":"English original..."}

Install:
    pip install googletrans==4.0.0rc1 tqdm

Examples:
    python generate_multilang_googletrans_fast.py --input data.txt --output hadr_12lang.jsonl
    python generate_multilang_googletrans_fast.py --input data.txt --output hadr_12lang_1_200.jsonl --start 1 --end 200
    python generate_multilang_googletrans_fast.py --input data.txt --output hadr_12lang.jsonl --workers 24 --retries 5

Notes:
    googletrans is unofficial and may be rate-limited by Google. If that happens,
    reduce --workers, increase --sleep, or rerun with --resume.
"""

from __future__ import annotations

import argparse
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

try:
    from googletrans import Translator
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Install with:\n"
        "  pip install googletrans==4.0.0rc1 tqdm\n"
    ) from exc

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# Your model/project language codes -> Google Translate destination codes.
LANG_MAP: Dict[str, str] = {
    "ms": "ms",       # Malay
    "zh": "zh-cn",    # Chinese Simplified
    "id": "id",       # Indonesian
    "tl": "tl",       # Tagalog / Filipino
    "th": "th",       # Thai
    "my": "my",       # Burmese / Myanmar
    "lo": "lo",       # Lao
    "hi": "hi",       # Hindi
    "ko": "ko",       # Korean
    "ja": "ja",       # Japanese
    "fr": "fr",       # French
    "vi": "vi",       # Vietnamese
}

DEFAULT_LANGS = list(LANG_MAP.keys())

# One Translator per thread is safer than sharing one global instance.
_thread_local = threading.local()


def get_translator() -> Translator:
    translator = getattr(_thread_local, "translator", None)
    if translator is None:
        translator = Translator(service_urls=[
            "translate.googleapis.com",
            "translate.google.com",
        ])
        _thread_local.translator = translator
    return translator


def read_lines(path: Path, start: int | None, end: int | None) -> List[Tuple[int, str]]:
    """Return [(1-based line number, text), ...], skipping blank lines."""
    raw_lines = path.read_text(encoding="utf-8").splitlines()

    # Convert user-facing 1-based inclusive range to Python slice.
    start_idx = 0 if start is None else max(start - 1, 0)
    end_idx = len(raw_lines) if end is None else min(end, len(raw_lines))

    selected: List[Tuple[int, str]] = []
    for idx, line in enumerate(raw_lines[start_idx:end_idx], start=start_idx + 1):
        text = line.strip()
        if text:
            selected.append((idx, text))
    return selected


def load_done_keys(output_path: Path) -> set[Tuple[int, str]]:
    """For resume mode: detect already-written (line_no, source_lang) pairs."""
    done: set[Tuple[int, str]] = set()
    if not output_path.exists():
        return done

    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            line_no = obj.get("line_no")
            lang = obj.get("source_lang")
            if isinstance(line_no, int) and isinstance(lang, str):
                done.add((line_no, lang))
    return done


def translate_one(
    line_no: int,
    english: str,
    lang_code: str,
    retries: int,
    sleep: float,
) -> dict:
    """Translate one English sentence into one target/source language."""
    dest = LANG_MAP[lang_code]
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            translator = get_translator()
            result = translator.translate(english, src="en", dest=dest)
            translated = (result.text or "").strip()
            if not translated:
                raise RuntimeError("empty translation returned")

            return {
                "source_lang": lang_code,
                "source_text": translated,
                "target_text": english,
                "line_no": line_no,
            }

        except Exception as exc:
            last_error = exc
            # Backoff + jitter to reduce repeated rate-limit failures.
            delay = sleep * attempt + random.uniform(0, sleep)
            time.sleep(delay)

    return {
        "source_lang": lang_code,
        "source_text": "",
        "target_text": english,
        "line_no": line_no,
        "error": str(last_error) if last_error else "unknown translation error",
    }


def write_jsonl_record(f, record: dict, keep_line_no: bool) -> None:
    out = dict(record)
    if not keep_line_no:
        out.pop("line_no", None)
    f.write(json.dumps(out, ensure_ascii=False) + "\n")
    f.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast Google Translate JSONL generator for HADR data")
    parser.add_argument("--input", required=True, help="Input .txt file with one English sentence per line")
    parser.add_argument("--output", required=True, help="Output .jsonl file")
    parser.add_argument("--start", type=int, default=None, help="Start line number, 1-based inclusive")
    parser.add_argument("--end", type=int, default=None, help="End line number, 1-based inclusive")
    parser.add_argument("--langs", nargs="+", default=DEFAULT_LANGS, help=f"Language codes: {' '.join(DEFAULT_LANGS)}")
    parser.add_argument("--workers", type=int, default=16, help="Parallel worker threads. Try 8, 16, 24, or 32")
    parser.add_argument("--retries", type=int, default=4, help="Retries per translation")
    parser.add_argument("--sleep", type=float, default=0.35, help="Base retry sleep in seconds")
    parser.add_argument("--resume", action="store_true", help="Skip records already present in output. Requires line_no metadata")
    parser.add_argument("--keep-line-no", action="store_true", help="Keep line_no field in final JSONL for debugging/resume")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output instead of appending")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    invalid = [lang for lang in args.langs if lang not in LANG_MAP]
    if invalid:
        raise ValueError(f"Unsupported language code(s): {invalid}. Supported: {list(LANG_MAP)}")

    lines = read_lines(input_path, args.start, args.end)
    if not lines:
        raise ValueError("No non-empty lines found in selected range.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if args.resume and not args.overwrite:
        done = load_done_keys(output_path)
        print(f"Resume mode: found {len(done)} completed records in {output_path}")

    tasks: List[Tuple[int, str, str]] = []
    for line_no, english in lines:
        for lang in args.langs:
            if (line_no, lang) not in done:
                tasks.append((line_no, english, lang))

    total = len(tasks)
    print(f"Input lines: {len(lines)}")
    print(f"Languages: {', '.join(args.langs)}")
    print(f"Translations to run: {total}")
    print(f"Workers: {args.workers}")

    mode = "w" if args.overwrite or not output_path.exists() or not args.resume else "a"
    failures = 0

    iterator: Iterable
    with output_path.open(mode, encoding="utf-8") as fout:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(translate_one, line_no, english, lang, args.retries, args.sleep)
                for line_no, english, lang in tasks
            ]

            iterator = as_completed(futures)
            if tqdm is not None:
                iterator = tqdm(iterator, total=total, desc="Translating")

            for future in iterator:
                record = future.result()
                if record.get("error"):
                    failures += 1
                write_jsonl_record(fout, record, keep_line_no=args.keep_line_no or args.resume)

    print(f"Done. Wrote to: {output_path}")
    if failures:
        print(f"Warning: {failures} translations failed. Rerun with --resume or lower --workers.")


if __name__ == "__main__":
    main()
