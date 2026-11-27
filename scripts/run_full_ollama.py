"""Two-stage, checkpointed full generation with whole-batch seed reruns.

Each array task owns one model. Repair is allowed only after all four initial
tasks have finished. All dispatched requests and raw responses are retained.
"""

from runtime_config import assert_project

import argparse
import csv
import fcntl
import hashlib
import json
import os
import pwd
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

from run_ollama_pilot import api, gpu_model, log, now

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_ROOT = ROOT
FAMILIES = ["gemma", "qwen", "llama", "deepseek"]


def checked(path):
    path = Path(path)
    path.resolve().relative_to(ROOT)
    if path.is_symlink():
        raise ValueError(f"Refusing symbolic link: {path}")
    return path


def write_json(path, value):
    path = checked(path)
    temporary = checked(path.with_suffix(path.suffix + ".tmp"))
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def append_json(path, value):
    with checked(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def string_array(value):
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def extract_sentences(text):
    """Extract unchanged strings, allowing only unambiguous outer packaging."""
    text = text.strip()
    try:
        value = json.loads(text)
    except ValueError:
        value = None
    else:
        if string_array(value):
            return value, "plain_json", None
        if isinstance(value, dict) and len(value) == 1:
            inner = next(iter(value.values()))
            if string_array(inner):
                return inner, "single_array_in_object", None
        return [], "invalid_structure", "JSON value is not a string array"
    fence = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", text, re.S | re.I)
    if fence:
        try:
            value = json.loads(fence.group(1))
        except ValueError:
            pass
        else:
            if string_array(value):
                return value, "markdown_fence", None
    decoder = json.JSONDecoder()
    candidates = []
    for match in re.finditer(r"\[", text):
        try:
            value, end = decoder.raw_decode(text, match.start())
        except ValueError:
            continue
        if string_array(value):
            candidates.append((match.start(), end, value))
    outer = [item for item in candidates if not any(
        other[0] < item[0] and other[1] >= item[1] for other in candidates
    )]
    if len(outer) == 1:
        return outer[0][2], "array_with_outer_text", None
    return [], "unparsed", "No unique JSON string array could be extracted"


def assess(response, expected):
    sentences, packaging, error = extract_sentences(response.get("response", ""))
    issues = []
    if error:
        issues.append({"type": "parse_error", "detail": error})
    else:
        if len(sentences) != expected:
            issues.append({"type": "count_mismatch", "expected": expected, "actual": len(sentences)})
        no_han, no_latin, empty = [], [], []
        for index, sentence in enumerate(sentences, 1):
            if not sentence.strip():
                empty.append(index)
            if not re.search(r"[\u3400-\u9fff\uf900-\ufaff\U00020000-\U0002ffff\u3007]", sentence):
                no_han.append(index)
            if not re.search(r"[A-Za-z]", sentence):
                no_latin.append(index)
        for name, items in [("no_han", no_han), ("no_latin", no_latin), ("empty_item", empty)]:
            if items:
                issues.append({"type": name, "items": items})
    if response.get("done_reason") != "stop":
        issues.append({"type": "incomplete_generation", "done_reason": response.get("done_reason")})
    return {"accepted": not issues, "sentences": sentences, "packaging": packaging, "issues": issues}


def seed_for(group_id, repeat_id, attempt_id, generation, policy):
    offset = policy["repeats_per_group"] * (group_id - 1) + repeat_id - 1
    return generation["seed_rule"]["base_seed"] + policy["retry_seed_stride"] * attempt_id + offset


def read_inputs():
    config = json.loads((ROOT / "configs/osc.json").read_text())
    generation = json.loads((ROOT / "configs/generation.json").read_text())
    policy = json.loads((ROOT / "configs/full_run.json").read_text())
    models = [item for item in config["models"] if item["provider"] == "ollama"]
    if [item["family"] for item in models] != FAMILIES:
        raise ValueError("Expected the four configured Ollama families")
    if [item["model"] for item in models] != ["gemma3:27b", "qwen3:32b", "llama3:70b", "deepseek-llm:67b"]:
        raise ValueError("Unexpected full-run models")
    if policy["groups"] != list(range(1, 11)) or policy["repeats_per_group"] != 5:
        raise ValueError("Expected ten groups and five repeats")
    if policy["max_reruns_per_batch"] != 3 or policy["retry_seed_stride"] != 50:
        raise ValueError("Expected three reruns and disjoint 50-seed blocks")
    if policy["batch_size"] != 10 or policy["retry_unit"] != "whole_batch" or policy["selection"] != "first_accepted_attempt":
        raise ValueError("Unexpected acceptance policy")
    if generation["conversation_history"] or policy["history"]:
        raise ValueError("Conversation history is not allowed")
    if not re.fullmatch(r"[a-z][a-z0-9_]+", policy["run_name"]):
        raise ValueError("Invalid run directory name")
    prompts = [json.loads(line) for line in (ROOT / "data/prompts.jsonl").read_text().splitlines()]
    if [item["group_id"] for item in prompts] != policy["groups"]:
        raise ValueError("Expected one prompt for each group in source order")
    return config, generation, policy, models, {item["group_id"]: item["prompt"] for item in prompts}


def all_slots(policy):
    return [(group, repeat) for group in policy["groups"] for repeat in range(1, policy["repeats_per_group"] + 1)]


def best_attempt(records, slot):
    matches = [(key[2], value) for key, value in records.items() if key[:2] == slot and value.get("accepted")]
    return min(matches, key=lambda item: item[0])[1] if matches else None


def freeze_inputs(run_dir, config, generation, policy):
    names = ["configs/osc.json", "configs/generation.json", "configs/full_run.json", "data/prompts.jsonl", "scripts/run_full_ollama.py", "scripts/run_ollama_pilot.py", "run/full_large_array.sbatch"]
    snapshot = {"config": config, "generation": generation, "policy": policy,
                "file_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}}
    with checked(run_dir / ".config.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        path = checked(run_dir / "config_snapshot.json")
        if path.exists():
            if json.loads(path.read_text()) != snapshot:
                raise RuntimeError("Frozen full-run inputs changed; refusing to mix versions")
        else:
            write_json(path, snapshot)


def load_records(model_dir):
    records = {}
    ledger = checked(model_dir / "requests.jsonl")
    if not ledger.exists():
        return records
    for line in ledger.read_text().splitlines():
        request = json.loads(line)
        key = (request["group_id"], request["repeat_id"], request["attempt_id"])
        if key in records:
            raise RuntimeError(f"Duplicate dispatched request: {key}")
        path = checked(model_dir / "attempts" / f"g{key[0]:02d}_r{key[1]:02d}_a{key[2]:02d}.json")
        if path.exists():
            records[key] = json.loads(path.read_text())
        else:
            record = {"request": request, "accepted": False, "sentences": [], "packaging": "unknown",
                      "issues": [{"type": "interrupted_request", "detail": "Dispatched request has no saved response; its seed will not be reused"}]}
            write_json(path, record)
            records[key] = record
    return records


def save_model_summary(model_dir, entry, records, policy, repair_finished=None):
    if repair_finished is None:
        previous = checked(model_dir / "summary.json")
        repair_finished = json.loads(previous.read_text()).get("repair_finished", False) if previous.exists() else False
    slots = all_slots(policy)
    accepted = [(slot, best_attempt(records, slot)) for slot in slots]
    initial = [records[(g, r, 0)] for g, r in slots if (g, r, 0) in records]
    summary = {
        "model": entry["model"], "family": entry["family"], "updated_at_utc": now(),
        "initial_finished": len(initial) == len(slots), "initial_requests": len(initial),
        "initial_accepted_batches": sum(item.get("accepted", False) for item in initial),
        "dispatched_requests": len(records), "rerun_requests": sum(key[2] > 0 for key in records),
        "accepted_batches": sum(record is not None for _, record in accepted),
        "accepted_sentences": policy["batch_size"] * sum(record is not None for _, record in accepted),
        "repair_finished": repair_finished,
        "unresolved": [{"group_id": slot[0], "repeat_id": slot[1],
                        "attempts": sum(key[:2] == slot for key in records),
                        "last_issues": max((value for key, value in records.items() if key[:2] == slot),
                                           key=lambda value: value["request"]["attempt_id"], default={}).get("issues", [])}
                       for slot, record in accepted if record is None],
    }
    write_json(model_dir / "summary.json", summary)
    fields = ["model", "group_id", "repeat_id", "attempt_id", "seed", "sentence_id", "sentence"]
    target = checked(model_dir / "accepted.csv")
    temporary = checked(target.with_suffix(".csv.tmp"))
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for slot, record in accepted:
            if record is None:
                continue
            request = record["request"]
            for index, sentence in enumerate(record["sentences"], 1):
                writer.writerow({"model": entry["model"], "group_id": slot[0], "repeat_id": slot[1],
                                 "attempt_id": request["attempt_id"], "seed": request["seed"],
                                 "sentence_id": index, "sentence": sentence})
    os.replace(temporary, target)
    return summary


def aggregate(run_dir):
    with checked(run_dir / ".aggregate.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        summaries = []
        for family in FAMILIES:
            path = checked(run_dir / "models" / family / "summary.json")
            if path.exists():
                summaries.append(json.loads(path.read_text()))
        summary = {"updated_at_utc": now(), "by_model": summaries,
                   "first_pass_complete": len(summaries) == 4 and all(s["initial_finished"] for s in summaries),
                   "repair_complete": len(summaries) == 4 and all(s["repair_finished"] for s in summaries),
                   "all_models_have_500": len(summaries) == 4 and all(s["accepted_sentences"] == 500 for s in summaries)}
        write_json(run_dir / "summary.json", summary)
        target = checked(run_dir / "accepted.csv")
        temporary = checked(target.with_suffix(".csv.tmp"))
        fields = ["model", "group_id", "repeat_id", "attempt_id", "seed", "sentence_id", "sentence"]
        with temporary.open("w", encoding="utf-8-sig", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fields); writer.writeheader()
            for family in FAMILIES:
                source = checked(run_dir / "models" / family / "accepted.csv")
                if source.exists():
                    with source.open(encoding="utf-8-sig", newline="") as stream:
                        writer.writerows(csv.DictReader(stream))
        os.replace(temporary, target)


def process_model(entry, phase, run_dir, prompts, generation, policy, port):
    model_dir = checked(run_dir / "models" / entry["family"])
    model_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    checked(model_dir / "attempts").mkdir(mode=0o700, exist_ok=True)
    with checked(model_dir / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        records = load_records(model_dir)
        slots = all_slots(policy)
        if phase == "repair":
            for family in FAMILIES:
                other = checked(run_dir / "models" / family / "summary.json")
                if not other.exists() or not json.loads(other.read_text())["initial_finished"]:
                    raise RuntimeError("All four initial model runs must finish before repair")
        work = [(g, r, 0) for g, r in slots if (g, r, 0) not in records] if phase == "initial" else [
            (g, r, attempt) for attempt in range(1, policy["max_reruns_per_batch"] + 1)
            for g, r in slots if (g, r, attempt) not in records and best_attempt(records, (g, r)) is None]
        if not work:
            save_model_summary(model_dir, entry, records, policy, repair_finished=True if phase == "repair" else None)
            aggregate(run_dir)
            log(f"{entry['model']} {phase}: no generation work remains")
            return
        base = f"http://127.0.0.1:{port}"
        for attempt in range(60):
            try:
                version = api(base, "/api/version", timeout=2)["version"]; break
            except OSError:
                if attempt == 59: raise
                time.sleep(1)
        info = api(base, "/api/show", {"model": entry["model"]})
        preload = {"model": entry["model"], "stream": False, "keep_alive": "10m", "options": generation["options"]}
        if entry["family"] == "qwen": preload["think"] = generation["qwen3_think"]
        api(base, "/api/generate", preload, timeout=600)
        loaded = gpu_model(base, entry["model"])
        write_json(model_dir / f"manifest_{phase}.json", {
            "model": entry["model"], "phase": phase, "started_at_utc": now(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"), "hostname": socket.gethostname(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "ollama_version": version,
            "loaded_model": loaded, "model_details": info.get("details"), "template": info.get("template"),
            "system": info.get("system"), "model_parameters": info.get("parameters"),
        })
        log(f"GPU verified: {entry['model']}, {loaded['size_vram'] / 1024**3:.2f} GiB in VRAM")
        for group, repeat, attempt in work:
            if phase == "repair" and best_attempt(records, (group, repeat)) is not None:
                continue
            seed = seed_for(group, repeat, attempt, generation, policy)
            payload = {"model": entry["model"], "prompt": prompts[group], "stream": False, "keep_alive": "10m",
                       "options": dict(generation["options"], seed=seed)}
            if entry["family"] == "qwen": payload["think"] = generation["qwen3_think"]
            request = {"model": entry["model"], "group_id": group, "repeat_id": repeat, "attempt_id": attempt,
                       "seed": seed, "started_at_utc": now(), "payload": payload,
                       "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                       "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
                       "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID")}
            append_json(model_dir / "requests.jsonl", request)
            log(f"Generating {entry['model']} g{group:02d} r{repeat} attempt {attempt} seed {seed}")
            record = {"request": request}
            try:
                response = api(base, "/api/generate", payload, timeout=generation["request_timeout_seconds"])
                record["response"] = response
                record.update(assess(response, policy["batch_size"]))
                record["loaded_model"] = gpu_model(base, entry["model"])
            except Exception as exc:
                record.update({"accepted": False, "sentences": [], "packaging": "unknown",
                               "issues": [{"type": "request_error", "detail": str(exc)}]})
            record["finished_at_utc"] = now()
            path = model_dir / "attempts" / f"g{group:02d}_r{repeat:02d}_a{attempt:02d}.json"
            if path.exists(): raise RuntimeError("Refusing to overwrite an existing attempt")
            write_json(path, record)
            records[(group, repeat, attempt)] = record
            summary = save_model_summary(model_dir, entry, records, policy)
            log(f"Saved: accepted={record['accepted']}; batch_count={len(record['sentences'])}; accepted_batches={summary['accepted_batches']}/50")
        summary = save_model_summary(model_dir, entry, records, policy, repair_finished=True if phase == "repair" else None)
        aggregate(run_dir)
        log(f"Finished {entry['model']} {phase}: {summary['accepted_sentences']} accepted sentences; {len(summary['unresolved'])} unresolved batches")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["initial", "repair"])
    parser.add_argument("--model-index", type=int)
    parser.add_argument("--port", type=int)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    config, generation, policy, models, prompts = read_inputs()
    if args.plan:
        print(json.dumps({"run_name": policy["run_name"], "models": [m["model"] for m in models],
                          "initial_requests_per_model": 50, "initial_requests_total": 200,
                          "max_reruns_per_batch": 3, "maximum_total_requests": 800,
                          "seed_blocks": [[seed_for(1, 1, a, generation, policy), seed_for(10, 5, a, generation, policy)] for a in range(4)]}, indent=2))
        return
    assert_project(ROOT)
    if args.phase is None or args.model_index not in range(4) or args.port is None or not 1024 <= args.port <= 65535:
        raise ValueError("Provide phase, model index 0-3 and a local server port")
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("A Slurm GPU allocation is required")
    if os.environ.get("OLLAMA_MODELS") != config["ollama_models_dir"]:
        raise RuntimeError("Unexpected model store")
    run_dir = checked(ROOT / "outputs" / policy["run_name"])
    run_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    freeze_inputs(run_dir, config, generation, policy)
    process_model(models[args.model_index], args.phase, run_dir, prompts, generation, policy, args.port)


if __name__ == "__main__":
    main()
