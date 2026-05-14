"""
Example:
python audio_segment.py --audio trial.mp4 --transcript eg.txt --language zh --model large-v3 --batch-size 4
"""

import argparse
import gc
import json
import re
import subprocess
from difflib import SequenceMatcher
from pathlib import Path

import torch
import whisperx


def split_transcript_into_sentences(text):
    text = re.sub(r"\s+", " ", text.strip())

    # Works better for Chinese because it does not require spaces after punctuation
    sentences = re.split(r"(?:(?<=[!?。！？])|(?<!\d)\.(?!\d))\s*", text)

    return [s.strip() for s in sentences if s.strip()]


def normalize_for_matching(text):
    text = text.lower()
    return re.sub(r"[\s，。！？、；：,.!?;:\"'“”‘’《》（）()\[\]【】\-—·%]+", "", text)


def normalize_char_for_matching(char):
    normalized = normalize_for_matching(char)
    return normalized or None


def text_distance(left, right):
    left = normalize_for_matching(left)
    right = normalize_for_matching(right)

    if not left and not right:
        return 0.0

    if not left or not right:
        return 1.0

    ratio = SequenceMatcher(None, left, right).ratio()
    length_penalty = abs(len(left) - len(right)) / max(len(left), len(right))
    return (1.0 - ratio) + (0.15 * length_penalty)


def map_sentences_to_asr_segments(human_sentences, asr_segments, max_sentences_per_asr):
    n = len(human_sentences)
    m = len(asr_segments)

    if n == 0 or m == 0:
        return []

    max_sentences_per_asr = max(max_sentences_per_asr, (n + m - 1) // m + 2)
    inf = float("inf")
    dp = [[inf] * (n + 1) for _ in range(m + 1)]
    back = [[None] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = 0.0

    for i in range(1, m + 1):
        asr_text = asr_segments[i - 1].get("text", "")

        for j in range(n + 1):
            if dp[i - 1][j] < inf:
                skip_cost = dp[i - 1][j] + 0.9
                if skip_cost < dp[i][j]:
                    dp[i][j] = skip_cost
                    back[i][j] = (j, j)

            start_min = max(0, j - max_sentences_per_asr)
            for k in range(start_min, j):
                if dp[i - 1][k] == inf:
                    continue

                human_text = "".join(human_sentences[k:j])
                cost = dp[i - 1][k] + text_distance(asr_text, human_text)
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    back[i][j] = (k, j)

    groups = []
    i, j = m, n
    while i > 0:
        item = back[i][j]
        if item is None:
            break

        k, current_j = item
        if k != current_j:
            groups.append((i - 1, k, current_j))

        j = k
        i -= 1

    groups.reverse()
    return groups


def build_forced_alignment_chunks(human_sentences, asr_segments, max_sentences_per_asr, padding):
    groups = map_sentences_to_asr_segments(
        human_sentences,
        asr_segments,
        max_sentences_per_asr,
    )

    forced_chunks = []
    total_duration = 0.0

    for asr_index, sentence_start, sentence_end in groups:
        asr_segment = asr_segments[asr_index]
        asr_start = float(asr_segment["start"])
        asr_end = float(asr_segment["end"])
        grouped_sentences = human_sentences[sentence_start:sentence_end]
        start = max(0.0, asr_start - padding)
        end = asr_end + padding

        forced_chunks.append(
            {
                "start": start,
                "end": end,
                "text": "".join(grouped_sentences),
                "sentence_start": sentence_start,
                "sentence_end": sentence_end,
                "asr_text": asr_segment.get("text", "").strip(),
                "asr_start": asr_start,
                "asr_end": asr_end,
            }
        )
        total_duration += max(0.0, end - start)

    return forced_chunks, total_duration


def split_aligned_chunks_to_sentences(aligned_chunks, forced_chunks, human_sentences):
    flat_chars = []
    for segment in aligned_chunks:
        chars = segment.get("chars")
        if chars:
            flat_chars.extend(chars)
            continue

        text = segment.get("text", "")
        start = segment.get("start")
        end = segment.get("end")
        for char in text:
            flat_chars.append({"char": char, "start": start, "end": end})

    sentence_segments = []
    char_cursor = 0

    for chunk in forced_chunks:
        for sentence_index in range(chunk["sentence_start"], chunk["sentence_end"]):
            sentence = human_sentences[sentence_index]
            sentence_chars = flat_chars[char_cursor:char_cursor + len(sentence)]
            char_cursor += len(sentence)

            starts = [char["start"] for char in sentence_chars if char.get("start") is not None]
            ends = [char["end"] for char in sentence_chars if char.get("end") is not None]

            if starts and ends:
                start = min(starts)
                end = max(ends)
            else:
                start = chunk["start"]
                end = chunk["end"]

            sentence_segments.append(
                {
                    "start": start,
                    "end": end,
                    "text": sentence,
                    "alignment": "forced_char",
                }
            )

    return sentence_segments


def build_timestamped_asr_chars(aligned_segments):
    chars = []

    for segment in aligned_segments:
        for char in segment.get("chars") or []:
            normalized = normalize_char_for_matching(char.get("char", ""))
            start = char.get("start")
            end = char.get("end")

            if normalized is None or start is None or end is None:
                continue

            for normalized_char in normalized:
                chars.append(
                    {
                        "char": normalized_char,
                        "start": float(start),
                        "end": float(end),
                    }
                )

    return chars


def map_human_sentences_to_asr_timestamps(human_sentences, aligned_asr_segments, min_match_ratio):
    asr_chars = build_timestamped_asr_chars(aligned_asr_segments)
    asr_text = "".join(char["char"] for char in asr_chars)

    human_sentence_ranges = []
    human_chars = []

    for sentence in human_sentences:
        start = len(human_chars)
        for char in sentence:
            normalized = normalize_char_for_matching(char)
            if normalized is None:
                continue
            human_chars.extend(normalized)
        end = len(human_chars)
        human_sentence_ranges.append((start, end))

    human_text = "".join(human_chars)
    matcher = SequenceMatcher(None, human_text, asr_text, autojunk=False)
    human_to_asr = {}

    for human_start, asr_start, size in matcher.get_matching_blocks():
        for offset in range(size):
            human_to_asr[human_start + offset] = asr_start + offset

    sentence_segments = []

    for i, sentence in enumerate(human_sentences):
        human_start, human_end = human_sentence_ranges[i]
        human_length = human_end - human_start

        if human_length == 0:
            continue

        matched_asr_indexes = [
            human_to_asr[index]
            for index in range(human_start, human_end)
            if index in human_to_asr
        ]
        match_ratio = len(matched_asr_indexes) / human_length

        if match_ratio < min_match_ratio:
            print(f"Skipping sentence {i + 1}: only {match_ratio:.0%} matched to ASR text")
            continue

        starts = [asr_chars[index]["start"] for index in matched_asr_indexes]
        ends = [asr_chars[index]["end"] for index in matched_asr_indexes]

        sentence_segments.append(
            {
                "start": min(starts),
                "end": max(ends),
                "text": sentence,
                "alignment": "asr_text_match",
                "match_ratio": round(match_ratio, 3),
            }
        )

    return sentence_segments


def run_ffmpeg(input_audio, start, end, output_path):
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_audio),
        "-ss", str(start),
        "-to", str(end),
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(output_path),
    ]

    # Do not hide errors while debugging
    subprocess.run(cmd, check=True)


def load_asr_model(args, device):
    try:
        return whisperx.load_model(
            args.model,
            device,
            compute_type=args.compute_type,
            language=args.language,
            task=args.task,
        )
    except TypeError:
        print("WhisperX version does not accept language/task at load time; continuing with defaults.")
        return whisperx.load_model(
            args.model,
            device,
            compute_type=args.compute_type,
        )


def load_alignment_model(language_code, args, device):
    if args.align_model:
        return whisperx.load_align_model(
            language_code=language_code,
            device=device,
            model_name=args.align_model,
        )

    return whisperx.load_align_model(language_code=language_code, device=device)


def clear_cuda_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def resolve_input_path(value, script_dir):
    path = Path(value)
    if path.is_absolute():
        return path

    candidates = [
        path,
        script_dir / path,
        script_dir.parent / path,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return path


def clear_output_files(clips_dir, manifest_path):
    for wav_path in clips_dir.glob("*.wav"):
        wav_path.unlink()

    if manifest_path.exists():
        manifest_path.unlink()


def main():
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser()

    parser.add_argument("--audio", required=True, help="Input audio/video file")
    parser.add_argument("--transcript", required=True, help="Human transcript .txt file")
    parser.add_argument("--out-dir", default=str(script_dir / "aligned_segments"))
    parser.add_argument("--model", default="base")
    parser.add_argument("--align-model", default=None, help="Optional WhisperX alignment model name")
    parser.add_argument("--language", default=None)
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compute-type", default=None, choices=["float16", "float32", "int8"])
    parser.add_argument("--align-padding", type=float, default=0.75)
    parser.add_argument("--max-sentences-per-asr", type=int, default=6)
    parser.add_argument("--min-match-ratio", type=float, default=0.45)

    args = parser.parse_args()
    args.compute_type = args.compute_type or ("float16" if args.device == "cuda" else "int8")

    input_audio = resolve_input_path(args.audio, script_dir)
    transcript_path = resolve_input_path(args.transcript, script_dir)

    if not input_audio.exists():
        raise FileNotFoundError(f"Audio/video file not found: {input_audio}")

    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript file not found: {transcript_path}")

    out_dir = Path(args.out_dir)
    clips_dir = out_dir / "audio"
    clips_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "train.jsonl"
    clear_output_files(clips_dir, manifest_path)

    transcript_text = transcript_path.read_text(encoding="utf-8-sig")
    human_sentences = split_transcript_into_sentences(transcript_text)

    print(f"Human transcript sentences: {len(human_sentences)}")

    if len(human_sentences) == 0:
        raise ValueError("Transcript has 0 sentences. Check your transcript file.")

    print(f"Loading WhisperX ASR model: {args.model}")
    print(f"Device: {args.device}; compute type: {args.compute_type}; batch size: {args.batch_size}")
    model = load_asr_model(args, args.device)

    print("Loading audio for WhisperX...")
    audio = whisperx.load_audio(str(input_audio))

    print("Running WhisperX transcription...")
    result = model.transcribe(audio, batch_size=args.batch_size)

    detected_language = args.language or result.get("language")
    if not detected_language:
        raise ValueError("WhisperX did not return a language. Pass --language, for example --language zh.")

    del model
    clear_cuda_cache()

    print(f"Loading WhisperX alignment model for language: {detected_language}")
    align_model, metadata = load_alignment_model(detected_language, args, args.device)

    print("Running WhisperX alignment on ASR text...")
    result = whisperx.align(
        result["segments"],
        align_model,
        metadata,
        audio,
        args.device,
        return_char_alignments=True,
    )

    del align_model
    clear_cuda_cache()

    aligned_asr_segments = result["segments"]
    aligned_segments = map_human_sentences_to_asr_timestamps(
        human_sentences,
        aligned_asr_segments,
        args.min_match_ratio,
    )

    print(f"WhisperX ASR-aligned segments: {len(aligned_asr_segments)}")
    print(f"Human transcript clips matched from ASR timestamps: {len(aligned_segments)}")

    if len(aligned_segments) == 0:
        raise ValueError("WhisperX returned 0 aligned segments. Check the audio/video file.")

    rows = []

    for i, seg in enumerate(aligned_segments):
        text = seg.get("text", "").strip()

        start = float(seg["start"])
        end = float(seg["end"])
        duration = end - start

        if duration <= 0:
            print(f"Skipping segment {i + 1}: invalid duration")
            continue

        if duration > args.max_seconds:
            print(f"Skipping segment {i + 1}: {duration:.1f}s longer than max {args.max_seconds}s")
            continue

        clip_name = f"{input_audio.stem}_{i + 1:05d}.wav"
        clip_path = clips_dir / clip_name

        print(f"Creating clip {i + 1}/{len(aligned_segments)}: {clip_name}")

        run_ffmpeg(input_audio, start, end, clip_path)

        row = {
            "audio": str(Path("audio") / clip_name),
            "text": text,
            "start": round(start, 2),
            "end": round(end, 2),
            "duration": round(duration, 2),
            "language": args.language,
            "detected_language": detected_language,
            "task": args.task,
            "alignment": seg.get("alignment", "forced_char"),
            "match_ratio": seg.get("match_ratio"),
        }

        rows.append(row)

    with open(manifest_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("Done.")
    print(f"Audio clips saved to: {clips_dir}")
    print(f"Manifest saved to: {manifest_path}")
    print(f"Total usable clips: {len(rows)}")


if __name__ == "__main__":
    main()
