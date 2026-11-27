"""Audit final corpora and mark excluded analysis rows NA, preserving sentences."""

import csv
import hashlib
import json
import math
import statistics
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/analysis_input_audit"
OLLAMA = ROOT / "outputs/full_large_10groups_5repeats"
GPT = ROOT / "outputs/full_gpt6_luna_10groups_5repeats"
LANGUAGE_REVIEW = ROOT / "configs/language_review.json"
SOURCES = [
    ("human", ROOT / "data/human_baseline.csv"),
    *[(family, OLLAMA / f"models/{family}/accepted.csv")
      for family in ["gemma", "qwen", "llama", "deepseek"]],
    ("gpt", GPT / "models/gpt/accepted.csv"),
]
JOINERS = set("'’‘ʼ-‐‑")
METRICS = ["m_index", "i_index", "burstiness", "span_entropy",
           "mean_english_span_length", "single_word_english_span_proportion"]


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_language_review(source_hashes):
    """Apply reviewed lexical decisions only to the exact reviewed inputs."""
    review = json.loads(LANGUAGE_REVIEW.read_text(encoding="utf-8"))
    normalize = lambda hashes: {key.replace("\\", "/"): value
                                for key, value in hashes.items()}
    if normalize(review["source_sha256"]) != normalize(source_hashes):
        raise ValueError("Inputs changed; update the lexical language review before auditing")
    decisions = {}
    for decision in review["decisions"]:
        key = (decision["family"], decision["data_row"])
        if key in decisions:
            raise ValueError(f"Duplicate language-review decision: {key}")
        if decision["decision"] not in {"keep_english", "non_chinese_english"}:
            raise ValueError(f"Unresolved language-review decision: {key}")
        decisions[key] = decision
    return review, decisions


def is_han(char):
    value = ord(char)
    return (value == 0x3007 or 0x3400 <= value <= 0x4DBF
            or 0x4E00 <= value <= 0x9FFF or 0xF900 <= value <= 0xFAFF
            or 0x20000 <= value <= 0x2FA1F or 0x30000 <= value <= 0x323AF)


def is_latin(char):
    return char.isalpha() and "LATIN" in unicodedata.name(char, "")


def whole_key(text):
    """Existing deduplication convention; retain numbers and non-P symbols."""
    return "".join(char.lower() for char in text if not char.isspace()
                   and not unicodedata.category(char).startswith("P"))


def scan(text):
    """Yield (kind, comparison form, start, end); keep neutral barriers for copies."""
    index = 0
    while index < len(text):
        char = text[index]
        if is_han(char):
            yield ("C", char, index, index + 1)
            index += 1
        elif is_latin(char) or char.isdecimal():
            end = index + 1
            while end < len(text):
                current = text[end]
                if is_latin(current) or current.isdecimal():
                    end += 1
                elif (unicodedata.category(current).startswith("M")
                      and is_latin(text[end - 1])):
                    end += 1
                elif (current in JOINERS and end + 1 < len(text)
                      and (is_latin(text[end - 1]) or text[end - 1].isdecimal())
                      and (is_latin(text[end + 1]) or text[end + 1].isdecimal())):
                    end += 1
                else:
                    break
            surface = text[index:end]
            kind = "E" if any(is_latin(item) for item in surface) else "D"
            yield (kind, whole_key(surface), index, end)
            index = end
        else:
            if not char.isspace() and not unicodedata.category(char).startswith("P"):
                yield ("O", char, index, index + 1)
            index += 1


def metric_values(tokens):
    labels = [kind for kind, _, _, _ in tokens if kind in {"C", "E"}]
    chinese, english = labels.count("C"), labels.count("E")
    reason = ("no_language_tokens" if not labels else
              "no_chinese" if not chinese else "no_english" if not english else "")
    base = {"chinese_characters": chinese, "english_words": english,
            "effective_tokens": len(labels), "na_reason": reason}
    if reason:
        return {**base, **{name: None for name in METRICS}}
    runs = []
    for label in labels:
        if runs and runs[-1][0] == label:
            runs[-1][1] += 1
        else:
            runs.append([label, 1])
    lengths = [length for _, length in runs]
    english_lengths = [length for label, length in runs if label == "E"]
    mean = statistics.mean(lengths)
    sigma = statistics.pstdev(lengths)
    frequency = Counter(lengths)
    values = [
        2 * chinese * english / (chinese ** 2 + english ** 2),
        (len(runs) - 1) / (len(labels) - 1),
        (sigma - mean) / (sigma + mean),
        -sum((count / len(lengths)) * math.log2(count / len(lengths))
             for count in frequency.values()),
        english / len(english_lengths),
        english_lengths.count(1) / len(english_lengths),
    ]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Unexpected non-finite metric on a bilingual row")
    return {**base, **dict(zip(METRICS, values))}


def write_csv(name, rows, fields):
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def mark_analysis_status(entry):
    reasons = []
    if entry["na_reason"]:
        reasons.extend(entry["na_reason"].split(";"))
    if entry["unsupported_letters"]:
        reasons.append("non_chinese_english_letters")
    if entry["language_review_decision"] == "non_chinese_english":
        reasons.append("non_chinese_english_content")
    if entry["prompt_whole_copy"]:
        reasons.append("prompt_whole_copy")
    if entry["prompt_fragment_copy"]:
        reasons.append("prompt_fragment_copy")
    if entry["family"] != "human" and entry["internal_duplicate_occurrence"] > 1:
        reasons.append("internal_duplicate_after_first")
    entry["_na_reasons"] = list(dict.fromkeys(reasons))
    entry["na_reason"] = ";".join(entry["_na_reasons"])
    entry["analysis_status"] = "NA" if reasons else "valid"
    entry["included_in_analysis"] = not reasons
    if reasons:
        for name in METRICS:
            entry[name] = "NA"


def fragment_matches(output, prompts, index):
    tokens = output["_tokens"]
    sequence = [(kind, value) for kind, value, _, _ in tokens]
    best = {}
    seen = set()
    for start in range(len(sequence) - 4):
        for prompt_index, prompt_start in index.get(tuple(sequence[start:start + 5]), []):
            prompt_sequence = prompts[prompt_index]["_sequence"]
            left, other_left = start, prompt_start
            right, other_right = start + 5, prompt_start + 5
            while left and other_left and sequence[left - 1] == prompt_sequence[other_left - 1]:
                left -= 1
                other_left -= 1
            while (right < len(sequence) and other_right < len(prompt_sequence)
                   and sequence[right] == prompt_sequence[other_right]):
                right += 1
                other_right += 1
            signature = (prompt_index, left, right, other_left, other_right)
            if signature in seen:
                continue
            seen.add(signature)
            units = sum(kind in {"C", "E"} for kind, _ in sequence[left:right])
            if units >= 5 and (prompt_index not in best or units > best[prompt_index]["units"]):
                prompt_tokens = prompts[prompt_index]["_tokens"]
                best[prompt_index] = {
                    "units": units,
                    "output_excerpt": output["sentence"][tokens[left][2]:tokens[right - 1][3]],
                    "prompt_excerpt": prompts[prompt_index]["original_sentence"][
                        prompt_tokens[other_left][2]:prompt_tokens[other_right - 1][3]],
                }
    return best


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source_hashes = {str(path.relative_to(ROOT)): digest(path) for _, path in SOURCES}
    review, language_decisions = load_language_review(source_hashes)
    applied_decisions, reviewed_word_forms = set(), set()
    prompts = read_csv(ROOT / "data/research_sentences.csv")
    if len(prompts) != 100:
        raise ValueError("Expected 100 prompt examples")
    by_group, indexes = defaultdict(list), {}
    for prompt in prompts:
        prompt["_tokens"] = list(scan(prompt["original_sentence"]))
        prompt["_sequence"] = [(kind, value) for kind, value, _, _ in prompt["_tokens"]]
        by_group[int(prompt["group_id"])].append(prompt)
    for group, examples in by_group.items():
        if len(examples) != 10:
            raise ValueError("Expected ten examples in each prompt group")
        index = defaultdict(list)
        for number, prompt in enumerate(examples):
            sequence = prompt["_sequence"]
            for start in range(len(sequence) - 4):
                index[tuple(sequence[start:start + 5])].append((number, start))
        indexes[group] = index

    rows, duplicates, copies, summaries, original_model_rows = [], [], [], [], {}
    for family, path in SOURCES:
        source = read_csv(path)
        if len(source) != 500:
            raise ValueError(f"Expected 500 rows: {path}")
        if family != "human":
            batches = Counter((int(row["group_id"]), int(row["repeat_id"])) for row in source)
            if set(batches) != {(group, repeat) for group in range(1, 11) for repeat in range(1, 6)}:
                raise ValueError(f"Unexpected batches: {family}")
            if set(batches.values()) != {10}:
                raise ValueError(f"Unexpected batch sizes: {family}")
            original_model_rows[family] = source
        else:
            human_ids = {(row["university"], row["sent_id"]) for row in source}
            prompt_ids = {(row["university"], row["sent_id"]) for row in prompts}
            if human_ids & prompt_ids:
                raise ValueError("Human baseline overlaps prompt examples")
        family_rows, keyed = [], defaultdict(list)
        for number, row in enumerate(source, 1):
            sentence = row.get("sentence", row.get("original_sentence", ""))
            tokens = list(scan(sentence))
            reviewed_word_forms.update(sentence[start:end].lower()
                                       for kind, _, start, end in tokens if kind == "E")
            decision = language_decisions.get((family, number), {})
            if decision:
                if decision["sentence"] != sentence:
                    raise ValueError(f"Language-review sentence mismatch: {family}, {number}")
                if decision["excerpt"] not in sentence:
                    raise ValueError(f"Language-review excerpt mismatch: {family}, {number}")
                applied_decisions.add((family, number))
            unsupported = "".join(dict.fromkeys(
                char for char in sentence if char.isalpha() and not is_han(char) and not is_latin(char)))
            entry = {
                "family": family, "model": row.get("model", "human"),
                "data_row": number, "group_id": row.get("group_id", ""),
                "repeat_id": row.get("repeat_id", ""), "attempt_id": row.get("attempt_id", ""),
                "sentence_id": row.get("sentence_id", row.get("sample_id", "")),
                "university": row.get("university", ""), "sent_id": row.get("sent_id", ""),
                "source_csv": str(path.relative_to(ROOT)), "sentence": sentence,
                "unsupported_letters": unsupported, **metric_values(tokens),
                "language_review_decision": decision.get("decision", ""),
                "language_review_excerpt": decision.get("excerpt", ""),
                "language_review_basis": decision.get("basis", ""),
                "language_review_note": decision.get("note", ""),
                "language_review_reference": decision.get("reference", ""),
                "internal_duplicate_group": "", "internal_duplicate_occurrence": 0,
                "internal_duplicate_group_size": 0,
                "analysis_status": "", "included_in_analysis": False,
                "prompt_whole_copy": False,
                "prompt_fragment_copy": False, "longest_prompt_copy_units": 0,
                "_tokens": tokens,
            }
            key = whole_key(sentence)
            keyed[key].append(entry)
            if family != "human":
                group = int(row["group_id"])
                examples = by_group[group]
                matches = fragment_matches(entry, examples, indexes[group])
                exact = {i for i, prompt in enumerate(examples)
                         if whole_key(prompt["original_sentence"]) == key}
                entry["prompt_whole_copy"] = bool(exact)
                entry["prompt_fragment_copy"] = bool(matches)
                entry["longest_prompt_copy_units"] = max(
                    (match["units"] for match in matches.values()), default=0)
                for prompt_index in sorted(set(matches) | exact):
                    prompt = examples[prompt_index]
                    match = matches.get(prompt_index, {})
                    copies.append({
                        **entry, "prompt_sample_id": prompt["sample_id"],
                        "prompt_example_id": prompt["example_id"],
                        "whole_copy": prompt_index in exact,
                        "fragment_copy": prompt_index in matches,
                        "matched_units": match.get("units", 0),
                        "output_excerpt": match.get("output_excerpt", ""),
                        "prompt_excerpt": match.get("prompt_excerpt", ""),
                        "prompt_sentence": prompt["original_sentence"],
                    })
            family_rows.append(entry)
        duplicate_groups = [items for items in keyed.values() if len(items) > 1]
        for number, items in enumerate(duplicate_groups, 1):
            group_id = f"{family}_D{number:03d}"
            for position, item in enumerate(items, 1):
                item["internal_duplicate_group"] = group_id
                item["internal_duplicate_occurrence"] = position
                item["internal_duplicate_group_size"] = len(items)
                duplicates.append({**item, "group_size": len(items),
                                   "occurrence": position, "normalized_key": whole_key(item["sentence"])})
        for entry in family_rows:
            mark_analysis_status(entry)
        summaries.append({
            "family": family, "model": family_rows[0]["model"], "rows": len(source),
            "na_rows": sum(bool(row["na_reason"]) for row in family_rows),
            "na_reasons": dict(Counter(reason for row in family_rows
                                       for reason in row["_na_reasons"])),
            "bilingual_rows": sum(row["chinese_characters"] > 0 and row["english_words"] > 0
                                  for row in family_rows),
            "valid_analysis_rows": sum(row["included_in_analysis"] for row in family_rows),
            "rows_with_unsupported_letters": sum(bool(row["unsupported_letters"]) for row in family_rows),
            "rows_with_non_chinese_english_content": sum(
                row["language_review_decision"] == "non_chinese_english" for row in family_rows),
            "internal_duplicate_groups": len(duplicate_groups),
            "internal_duplicate_rows": sum(map(len, duplicate_groups)),
            "internal_duplicate_excess": sum(len(items) - 1 for items in duplicate_groups),
            "prompt_whole_copy_rows": sum(row["prompt_whole_copy"] for row in family_rows),
            "prompt_fragment_copy_rows": sum(row["prompt_fragment_copy"] for row in family_rows),
            "longest_prompt_copy_units": max(row["longest_prompt_copy_units"] for row in family_rows),
        })
        rows.extend(family_rows)

    if applied_decisions != set(language_decisions):
        raise ValueError("Some language-review decisions did not match any input row")
    if len(reviewed_word_forms) != review["reviewed_latin_word_forms"]:
        raise ValueError("Latin word-form inventory differs from the reviewed inventory")
    expected = [row for family in ["gemma", "qwen", "llama", "deepseek"]
                for row in original_model_rows[family]]
    canonical = lambda records: Counter(tuple(sorted(row.items())) for row in records)
    if canonical(read_csv(OLLAMA / "accepted.csv")) != canonical(expected):
        raise ValueError("Combined Ollama CSV differs from model CSVs")
    if canonical(read_csv(GPT / "accepted.csv")) != canonical(original_model_rows["gpt"]):
        raise ValueError("Combined GPT CSV differs from model CSV")
    for _, path in SOURCES:
        if digest(path) != source_hashes[str(path.relative_to(ROOT))]:
            raise ValueError("An input changed during the audit")
    lookup = {(row["family"], row["data_row"]): row for row in rows}
    for detail in duplicates + copies:
        detail.update(lookup[(detail["family"], detail["data_row"])])
    fields = [key for key in rows[0] if not key.startswith("_")]
    write_csv("sentence_audit.csv", rows, fields)
    write_csv("na_rows.csv", [row for row in rows if row["na_reason"]], fields)
    write_csv("analysis_valid_rows.csv",
              [row for row in rows if row["included_in_analysis"]], fields)
    write_csv("language_review.csv", [row for row in rows if row["language_review_decision"]], fields)
    write_csv("internal_duplicates.csv", duplicates,
              fields + ["group_size", "occurrence", "normalized_key"])
    write_csv("prompt_copies.csv", copies, fields + [
        "prompt_sample_id", "prompt_example_id", "whole_copy", "fragment_copy",
        "matched_units", "output_excerpt", "prompt_excerpt", "prompt_sentence"])
    write_csv("summary.csv", summaries, list(summaries[0]))
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": source_hashes,
        "language_review": {
            "config": str(LANGUAGE_REVIEW.relative_to(ROOT)),
            "config_sha256": digest(LANGUAGE_REVIEW),
            "reviewed_latin_word_forms": len(reviewed_word_forms),
            "decision_counts": dict(Counter(item["decision"] for item in language_decisions.values())),
            "unresolved_decisions": 0,
        },
        "integrity": {"all_six_sources_have_500_rows": True,
                      "each_model_has_50_batches_of_ten": True,
                      "combined_csvs_equal_model_csvs": True,
                      "human_baseline_disjoint_from_prompts": True,
                      "lexical_review_matches_input_files": True,
                      "all_language_review_decisions_applied": True,
                      "audit_did_not_change_source_files": True},
        "methods": {
            "metric_units": "One Han character or one Latin/alphanumeric word; preserve internal apostrophes and hyphens.",
            "metric_filter": "Ignore separators, standalone numbers and symbols. All six metrics NA for monolingual rows, unsupported-script or reviewed third-language rows, prompt copies, and internal repeats after the first occurrence.",
            "language_identification": "Script screening plus review of all distinct Latin word forms and exceptional sentence contexts. Named entities, brands, abbreviations, loanwords and recognizable English misspellings count as English; ambiguous forms require explicit adjudication. See configs/language_review.json.",
            "contact_strings": "Keep existing English-token rules: em in em@160 counts as English, @ and standalone digits are separators; pn-19 counts as one English token.",
            "standard_deviation": "Population SD, ddof=0, for Burstiness and sentence-level descriptive summaries; Welch t-test retains its usual variance definition.",
            "sentence_summary_sd_ddof": 0,
            "whole_duplicates": "Lowercase and remove Unicode whitespace and punctuation; retain numbers and other symbols.",
            "fragment_copies": "At least five consecutive matching Chinese-character/English-word units, within the same prompt's ten examples. Numbers and non-punctuation symbols must match but do not count toward five language units.",
            "duplicates": "Keep the first internal duplicate occurrence and mark later occurrences NA. Prompt whole/fragment copies are NA. Exclude the union of NA rows from both analysis levels; preserve source sentences.",
            "analysis_outputs": "sentence_audit.csv retains all rows with status and six metric columns; na_rows.csv lists exclusions; analysis_valid_rows.csv contains only included rows; language_review.csv records exceptional lexical decisions.",
        },
        "summary": summaries,
    }
    (OUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
