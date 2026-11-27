"""Reproduce 100 prompt examples and a disjoint 500-sentence human baseline.

Source: https://github.com/lfang1/CodeSwitchingResearch
Commit: 80757512bc86c386302269f070b2a5e7fe57dfa2
Original file: 11152019_two_entropies_appended_input_R_v4.csv

Deduplicate code-switched rows by removing whitespace and Unicode punctuation
and lowercasing English, retaining the first source row per key (1428 rows).
Both draws use independent random.Random(2026).sample instances, without
replacement, with candidates in original CSV order. The baseline excludes
the 100 prompt examples by (university, sent_id). Derived sentence text removes
Chinese segmentation spaces and punctuation padding, keeping English word
spaces. Sampling preserves all source characters. In this public source copy,
phone numbers in the unused translation column have been redacted; the
original_sentence column is unchanged. See configs/public_export.json.
"""

import csv
import hashlib
import random
import re
import unicodedata
from pathlib import Path

SEED = 2026
SAMPLE_SIZE = 100
BASELINE_SIZE = 500
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "surprisal_corpus.csv"
DEDUP_OUTPUT = ROOT / "data" / "surprisal_deduplicated.csv"
OUTPUT = ROOT / "data" / "research_sentences.csv"
BASELINE_OUTPUT = ROOT / "data" / "human_baseline.csv"


def duplicate_key(text):
    """Strict whole-sentence key used to obtain 1428 unique source sentences."""
    return "".join(char.lower() for char in text
                   if not char.isspace()
                   and not unicodedata.category(char).startswith("P"))


def normalize_sentence(text):
    """Remove segmentation padding without merging English words."""
    text = re.sub(r"\s+", " ", text).strip()
    chinese = r"\u3400-\u9fff\uf900-\ufaff\U00020000-\U0002ffff\u3007"
    punctuation = "，。！？；：、（）【】《》〈〉「」『』“”‘’…—"
    boundary = chinese + re.escape(punctuation)
    text = re.sub(rf"(?<=[{boundary}]) +| +(?=[{boundary}])", "", text)
    text = re.sub(r" +(?=[.,;:!?%\)\]\}])", "", text)
    return re.sub(r"(?<=[\(\[\{]) +", "", text)


def main():
    data = SOURCE.read_bytes()
    blob_sha = hashlib.sha1(
        b"blob " + str(len(data)).encode() + b"\0" + data
    ).hexdigest()
    if blob_sha != "6ce50b143bee74e7f036c6052064c0290dabaee2":
        raise ValueError("Source CSV differs from the checksummed public redacted copy.")
    with SOURCE.open(encoding="utf-8-sig", newline="") as stream:
        source_rows = [row for row in csv.DictReader(stream)
                       if row["sent_type"] == "code-switch"]
    if len(source_rows) != 1476:
        raise ValueError("Expected 1476 code-switched source rows.")
    candidates = []
    seen = set()
    for row in source_rows:
        key = duplicate_key(row["original_sentence"])
        if key not in seen:
            seen.add(key)
            candidates.append(row)
    if len(candidates) != 1428:
        raise ValueError("Expected 1428 strictly deduplicated source sentences.")
    with DEDUP_OUTPUT.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["university", "sent_id",
                                                    "original_sentence"])
        writer.writeheader()
        for row in candidates:
            writer.writerow({"university": row["university"],
                             "sent_id": row["sent_id"],
                             "original_sentence": normalize_sentence(row["original_sentence"])})
    print(f"Saved {len(candidates)} unique sentences; removed 48 redundant rows: {DEDUP_OUTPUT}")
    # Sample without replacement from the original CSV row order.
    selected = random.Random(SEED).sample(candidates, SAMPLE_SIZE)
    fields = ["sample_id", "group_id", "example_id", "university",
              "sent_id", "original_sentence"]
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(selected):
            writer.writerow({
                "sample_id": index + 1,
                "group_id": index // 10 + 1,
                "example_id": index % 10 + 1,
                **{key: normalize_sentence(row[key]) if key == "original_sentence"
                   else row[key] for key in fields[3:]},
            })
    print(f"Saved {SAMPLE_SIZE} sentences, 10 groups, seed={SEED}: {OUTPUT}")
    selected_ids = {(row["university"], row["sent_id"]) for row in selected}
    remaining = [row for row in candidates
                 if (row["university"], row["sent_id"]) not in selected_ids]
    if len(remaining) != 1328:
        raise ValueError("Expected 1328 candidates after excluding prompt examples.")
    baseline = random.Random(SEED).sample(remaining, BASELINE_SIZE)
    baseline_fields = ["sample_id", "university", "sent_id", "original_sentence"]
    with BASELINE_OUTPUT.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=baseline_fields)
        writer.writeheader()
        for index, row in enumerate(baseline, 1):
            writer.writerow({"sample_id": index,
                             **{key: normalize_sentence(row[key])
                                if key == "original_sentence" else row[key]
                                for key in baseline_fields[1:]}})
    print(f"Saved {BASELINE_SIZE} human baseline sentences, seed={SEED}: {BASELINE_OUTPUT}")


if __name__ == "__main__":
    main()
