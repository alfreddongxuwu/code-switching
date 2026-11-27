"""Build ten fixed prompts locally; no model calls.

Run: python scripts/build_prompts.py
Input: prompt.txt and data/research_sentences.csv
Output: data/prompts.jsonl (one group_id/prompt object per line)
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    template = (ROOT / "prompt.txt").read_text(encoding="utf-8-sig")
    if template.count("{examples}") != 1:
        raise ValueError("Prompt template must contain exactly one {examples}.")
    with (ROOT / "data" / "research_sentences.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 100 or len({(r["university"], r["sent_id"])
                               for r in rows}) != 100:
        raise ValueError("Expected 100 distinct source sentences.")
    groups = defaultdict(list)
    for row in rows:
        if not row["original_sentence"].strip():
            raise ValueError("Empty example sentence.")
        groups[int(row["group_id"])].append(row)
    if set(groups) != set(range(1, 11)):
        raise ValueError("Expected group IDs 1 through 10.")
    records = []
    for group_id in sorted(groups):
        examples = sorted(groups[group_id], key=lambda r: int(r["example_id"]))
        if [int(r["example_id"]) for r in examples] != list(range(1, 11)):
            raise ValueError(f"Group {group_id} must have example IDs 1 through 10.")
        text = "\n".join(row["original_sentence"] for row in examples)
        records.append({"group_id": group_id,
                        "prompt": template.replace("{examples}", text)})
    output = ROOT / "data" / "prompts.jsonl"
    output.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n"
                              for record in records), encoding="utf-8")
    print(f"Saved {len(records)} prompts with 10 examples each: {output}")


if __name__ == "__main__":
    main()
