import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util


# NLLB-200 language codes. Add more if your JSONL uses other source_lang codes.
LANG_CODE_MAP = {
    "en": "eng_Latn",
    "ms": "zsm_Latn",      # Malay
    "id": "ind_Latn",      # Indonesian
    "zh": "zho_Hans",      # Chinese Simplified
    "zh-cn": "zho_Hans",
    "zh-tw": "zho_Hant",
    "tl": "tgl_Latn",      # Tagalog / Filipino
    "fil": "tgl_Latn",
    "th": "tha_Thai",      # Thai
    "lo": "lao_Laoo",      # Lao
    "km": "khm_Khmr",      # Khmer
    "my": "mya_Mymr",      # Burmese / Myanmar
    "vi": "vie_Latn",      # Vietnamese
    "hi": "hin_Deva",      # Hindi
    "ta": "tam_Taml",      # Tamil
    "bn": "ben_Beng",      # Bengali
    "ar": "arb_Arab",      # Arabic MSA
    "ja": "jpn_Jpan",      # Japanese
    "ko": "kor_Hang",      # Korean
}


def count_lines(path: str) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    with p.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def skip_non_empty_lines(infile, lines_to_skip: int) -> int:
    """Skip processed non-empty input rows once at startup and return the absolute line number reached."""
    line_num = 0
    skipped = 0

    while skipped < lines_to_skip:
        line = infile.readline()
        if not line:
            break

        line_num += 1
        if line.strip():
            skipped += 1

    return line_num


def load_jsonl_chunk(infile, chunk_size: int, start_line_num: int) -> Tuple[List[Tuple[int, Dict[str, Any], str]], int]:
    """Return a list of (line_num, record, raw_line) from the current input position."""
    chunk = []
    line_num = start_line_num

    while True:
        line = infile.readline()
        if not line:
            break

        line_num += 1
        raw_line = line.rstrip("\n")
        if not raw_line.strip():
            continue

        try:
            record = json.loads(raw_line)
            chunk.append((line_num, record, raw_line))
        except Exception as e:
            # Keep malformed lines as records that will be written to error file later.
            chunk.append((line_num, {"__parse_error__": str(e)}, raw_line))

        if len(chunk) >= chunk_size:
            break

    return chunk, line_num


def translate_batch(
    texts: List[str],
    source_lang: str,
    tokenizer,
    model,
    device: str,
    max_length: int,
) -> List[str]:
    """Translate a batch of texts from source_lang to English using local NLLB."""
    if source_lang not in LANG_CODE_MAP:
        raise ValueError(f"Unsupported source_lang '{source_lang}'. Add it to LANG_CODE_MAP.")

    nllb_source = LANG_CODE_MAP[source_lang]
    nllb_target = "eng_Latn"

    tokenizer.src_lang = nllb_source
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)

    forced_bos_token_id = tokenizer.convert_tokens_to_ids(nllb_target)

    with torch.no_grad():
        generated_tokens = model.generate(
            **encoded,
            forced_bos_token_id=forced_bos_token_id,
            max_length=max_length,
            num_beams=4,
        )

    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(
        description="Back-translate JSONL source_text to English using a local NLLB model, then filter by similarity."
    )
    parser.add_argument("--input", default="my_12000_file.jsonl", help="Input JSONL file")
    parser.add_argument("--accepted", default="accepted_dataset.jsonl", help="Accepted output JSONL")
    parser.add_argument("--rejected", default="rejected_dataset.jsonl", help="Rejected output JSONL")
    parser.add_argument("--errors", default="error_dataset.jsonl", help="Error output JSONL")
    parser.add_argument("--threshold", type=float, default=0.82, help="Similarity threshold")
    parser.add_argument("--batch-size", type=int, default=16, help="Translation batch size")
    parser.add_argument("--chunk-size", type=int, default=256, help="Number of JSONL rows to process per chunk")
    parser.add_argument("--max-length", type=int, default=256, help="Max token length for translation")
    parser.add_argument(
        "--model",
        default="facebook/nllb-200-distilled-600M",
        help="Local/Hugging Face translation model name or path",
    )
    parser.add_argument(
        "--embedder",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="SentenceTransformer model name or path",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Overwrite outputs instead of appending/resuming",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading translation model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, use_safetensors=True).to(device)
    model.eval()

    print(f"Loading similarity model: {args.embedder}")
    embedder = SentenceTransformer(args.embedder, device=device)

    file_mode = "w" if args.no_resume else "a"
    already_processed = 0 if args.no_resume else (
        count_lines(args.accepted) + count_lines(args.rejected) + count_lines(args.errors)
    )

    print(f"Input: {args.input}")
    print(f"Similarity threshold: {args.threshold}")
    print(f"Already processed non-empty lines: {already_processed}")

    accepted_count = count_lines(args.accepted) if not args.no_resume else 0
    rejected_count = count_lines(args.rejected) if not args.no_resume else 0
    error_count = count_lines(args.errors) if not args.no_resume else 0

    with open(args.input, "r", encoding="utf-8") as infile, \
         open(args.accepted, file_mode, encoding="utf-8") as accepted_out, \
         open(args.rejected, file_mode, encoding="utf-8") as rejected_out, \
         open(args.errors, file_mode, encoding="utf-8") as error_out:

        current_line_num = skip_non_empty_lines(infile, already_processed)

        while True:
            chunk, current_line_num = load_jsonl_chunk(infile, args.chunk_size, current_line_num)
            if not chunk:
                break

            # Group valid records by source language for efficient NLLB batching.
            groups: Dict[str, List[Tuple[int, Dict[str, Any], str]]] = {}
            chunk_results: Dict[int, Dict[str, Any]] = {}

            for line_num, record, raw_line in chunk:
                try:
                    if "__parse_error__" in record:
                        raise ValueError(record["__parse_error__"])

                    source_lang = record["source_lang"].lower().strip()
                    source_text = record["source_text"]
                    target_text = record["target_text"]

                    if not source_text or not target_text:
                        raise ValueError("source_text or target_text is empty")

                    groups.setdefault(source_lang, []).append((line_num, record, raw_line))

                except Exception as e:
                    chunk_results[line_num] = {
                        "status": "error",
                        "line_num": line_num,
                        "raw_line": raw_line,
                        "error": str(e),
                    }

            # Translate each language group in batches.
            for source_lang, rows in groups.items():
                for start in range(0, len(rows), args.batch_size):
                    batch_rows = rows[start:start + args.batch_size]
                    texts = [r[1]["source_text"] for r in batch_rows]

                    try:
                        translations = translate_batch(
                            texts,
                            source_lang,
                            tokenizer,
                            model,
                            device,
                            args.max_length,
                        )

                        target_texts = [r[1]["target_text"] for r in batch_rows]
                        target_emb = embedder.encode(target_texts, convert_to_tensor=True, batch_size=args.batch_size)
                        back_emb = embedder.encode(translations, convert_to_tensor=True, batch_size=args.batch_size)
                        similarities = util.cos_sim(target_emb, back_emb).diagonal().tolist()

                        for (line_num, record, raw_line), back_translation, similarity in zip(batch_rows, translations, similarities):
                            chunk_results[line_num] = {
                                "status": "accepted" if similarity >= args.threshold else "rejected",
                                "line_num": line_num,
                                "record": record,
                                "source_lang": source_lang,
                                "source_text": record["source_text"],
                                "target_text": record["target_text"],
                                "back_translation": back_translation,
                                "similarity": float(similarity),
                            }

                    except Exception as e:
                        # If a whole batch fails, save each row to the error file.
                        for line_num, record, raw_line in batch_rows:
                            chunk_results[line_num] = {
                                "status": "error",
                                "line_num": line_num,
                                "raw_line": raw_line,
                                "error": str(e),
                            }

            # Write results in original input order.
            for line_num, _, _ in sorted(chunk, key=lambda x: x[0]):
                result = chunk_results[line_num]

                if result["status"] == "accepted":
                    accepted_out.write(json.dumps(result["record"], ensure_ascii=False) + "\n")
                    accepted_count += 1

                elif result["status"] == "rejected":
                    rejected_record = {
                        "line_num": result["line_num"],
                        "source_lang": result["source_lang"],
                        "source_text": result["source_text"],
                        "target_text": result["target_text"],
                        "back_translation": result["back_translation"],
                        "similarity": result["similarity"],
                    }
                    rejected_out.write(json.dumps(rejected_record, ensure_ascii=False) + "\n")
                    rejected_count += 1

                else:
                    error_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    error_count += 1

            accepted_out.flush()
            rejected_out.flush()
            error_out.flush()

            already_processed += len(chunk)
            print(
                f"Processed: {already_processed} | "
                f"Accepted: {accepted_count} | "
                f"Rejected: {rejected_count} | "
                f"Errors: {error_count}"
            )

    print("Done.")
    print(f"Accepted: {accepted_count}")
    print(f"Rejected: {rejected_count}")
    print(f"Errors: {error_count}")
    print(f"Accepted output: {args.accepted}")
    print(f"Rejected output: {args.rejected}")
    print(f"Error output: {args.errors}")


if __name__ == "__main__":
    main()
