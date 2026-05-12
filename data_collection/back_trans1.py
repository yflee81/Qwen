import json
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer, util

INPUT_FILE = "my_my_1_200.jsonl"
ACCEPTED_FILE = "accepted_dataset.jsonl"
REJECTED_FILE = "rejected_dataset.jsonl"

SIMILARITY_THRESHOLD = 0.82

embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

accepted_count = 0
rejected_count = 0
error_count = 0

with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
     open(ACCEPTED_FILE, "w", encoding="utf-8") as accepted_out, \
     open(REJECTED_FILE, "w", encoding="utf-8") as rejected_out:

    for line_num, line in enumerate(infile, start=1):
        line = line.strip()

        if not line:
            continue

        try:
            record = json.loads(line)

            source_lang = record["source_lang"]
            source_text = record["source_text"]
            target_text = record["target_text"]

            translator = GoogleTranslator(source=source_lang, target="en")
            back_translation = translator.translate(source_text)

            embeddings = embedder.encode(
                [target_text, back_translation],
                convert_to_tensor=True
            )

            similarity = util.cos_sim(embeddings[0], embeddings[1]).item()

            if similarity >= SIMILARITY_THRESHOLD:
                accepted_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                accepted_count += 1
                print(f"Line {line_num}: Accepted | similarity={similarity:.3f}")
            else:
                rejected_record = {
                    "line_num": line_num,
                    "source_lang": source_lang,
                    "source_text": source_text,
                    "target_text": target_text,
                    "back_translation": back_translation,
                    "similarity": similarity
                }

                rejected_out.write(json.dumps(rejected_record, ensure_ascii=False) + "\n")
                rejected_count += 1
                print(f"Line {line_num}: Rejected | similarity={similarity:.3f}")

        except Exception as e:
            error_count += 1
            print(f"Line {line_num}: Error | {e}")

print(f"Accepted: {accepted_count}")
print(f"Rejected: {rejected_count}")
print(f"Errors: {error_count}")
print(f"Accepted output: {ACCEPTED_FILE}")
print(f"Rejected output: {REJECTED_FILE}")