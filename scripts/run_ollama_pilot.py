"""Run groups 1-3, repeat 1, once for each configured Ollama model on GPU.

Raw requests and responses are retained. No response is repaired or retried.
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
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_ROOT = ROOT


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(message):
    print(f"{now()} {message}", flush=True)


def checked_path(path):
    path = Path(path)
    path.resolve().relative_to(ROOT)
    if path.is_symlink():
        raise ValueError(f"Refusing symbolic link: {path}")
    return path


def write_json(path, value):
    checked_path(path)
    temporary = checked_path(path.with_suffix(path.suffix + ".tmp"))
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def append_json(path, value):
    checked_path(path)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def api(base, endpoint, payload=None, timeout=30):
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
    request = urllib.request.Request(
        base + endpoint, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    if result.get("error"):
        raise RuntimeError(result["error"])
    return result


def gpu_model(base, model):
    for item in api(base, "/api/ps").get("models", []):
        if model in (item.get("name"), item.get("model")):
            if item.get("size_vram", 0) <= 0:
                raise RuntimeError(f"{model} has no weights in GPU memory")
            return item
    raise RuntimeError(f"{model} is not loaded")


def validate(text, expected):
    issues = []
    sentences = []
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError) as exc:
        return [], [f"Invalid JSON: {exc}"]
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        return [], ["Response is not an array of strings"]
    if len(parsed) != expected:
        issues.append(f"Expected {expected} strings, received {len(parsed)}")
    for index, text in enumerate(parsed, 1):
        both = bool(re.search(r"[\u3400-\u9fff\uf900-\ufaff\U00020000-\U0002ffff\u3007]", text)) and bool(
            re.search(r"[A-Za-z]+", text)
        )
        if not text.strip():
            issues.append(f"Sentence {index} is empty")
        if not both:
            issues.append(f"Sentence {index} does not contain both Han characters and Latin letters")
        sentences.append({"sentence_id": index, "sentence": text, "contains_both_languages": both})
    return sentences, issues


def run(port):
    assert_project(ROOT)
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("A Slurm job with an allocated GPU is required")
    if not 1024 <= port <= 65535:
        raise ValueError("Invalid local server port")
    config = json.loads((ROOT / "configs/osc.json").read_text())
    generation = json.loads((ROOT / "configs/generation.json").read_text())
    if config["project_dir"] != str(ROOT):
        raise ValueError("Configured project directory differs from this checkout")
    if os.environ.get("OLLAMA_MODELS") != config["ollama_models_dir"]:
        raise ValueError("Unexpected Ollama model directory")
    if generation["conversation_history"] is not False:
        raise ValueError("This experiment requires independent requests without history")
    prompts = [json.loads(line) for line in (ROOT / "data/prompts.jsonl").read_text().splitlines()]
    selected = [item for item in prompts if item["group_id"] in (1, 2, 3)]
    if [item["group_id"] for item in selected] != [1, 2, 3]:
        raise ValueError("Expected exactly one prompt for each of groups 1, 2 and 3")
    models = [item for item in config["models"] if item["provider"] == "ollama"]
    if [item["family"] for item in models] != ["gemma", "qwen", "llama", "deepseek"]:
        raise ValueError("Unexpected pilot model selection")
    run_name = config["pilot_run_name"]
    if not re.fullmatch(r"[a-z][a-z0-9_]+", run_name):
        raise ValueError("Pilot run name must be a simple directory name")
    run_dir = checked_path(ROOT / "outputs" / run_name)
    run_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_path = checked_path(run_dir / ".run.lock")
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        requests_path = checked_path(run_dir / "requests.jsonl")
        if requests_path.exists() and requests_path.stat().st_size:
            raise RuntimeError("Pilot already contains dispatched requests; refusing to repeat them")
        return execute(run_dir, selected, models, generation, port)


def execute(run_dir, prompts, models, generation, port):
    base = f"http://127.0.0.1:{port}"
    for attempt in range(60):
        try:
            version = api(base, "/api/version", timeout=2)["version"]
            break
        except OSError:
            if attempt == 59:
                raise
            time.sleep(1)
    manifest = {
        "run_name": run_dir.name, "started_at_utc": now(), "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "hostname": socket.gethostname(), "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "ollama_version": version, "generation": generation,
        "groups": [1, 2, 3], "repeat_id": 1, "expected_generation_requests": 12,
        "response_format": "Existing prompt instruction only; no constrained decoding",
        "file_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ["configs/osc.json", "configs/generation.json", "data/prompts.jsonl", "scripts/run_ollama_pilot.py"]
        },
        "nvidia_smi": subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,uuid,name,memory.total", "--format=csv,noheader"
        ], text=True).strip(),
        "models": [],
    }
    write_json(run_dir / "manifest.json", manifest)
    summary = {"run_name": run_dir.name, "slurm_job_id": manifest["slurm_job_id"], "requests": []}
    sentence_rows = []
    for entry in models:
        model = entry["model"]
        log(f"Loading {model} on the allocated GPU")
        info = api(base, "/api/show", {"model": model})
        context_lengths = [value for key, value in info.get("model_info", {}).items() if key.endswith(".context_length")]
        if context_lengths and generation["options"]["num_ctx"] > min(context_lengths):
            raise RuntimeError(f"Configured context exceeds {model}'s recorded context length")
        preload = {"model": model, "stream": False, "keep_alive": "5m", "options": generation["options"]}
        if entry["family"] == "qwen":
            preload["think"] = generation["qwen3_think"]
        api(base, "/api/generate", preload, timeout=600)
        loaded = gpu_model(base, model)
        log(f"GPU verified for {model}: {loaded['size_vram'] / 1024**3:.2f} GiB in VRAM")
        manifest["models"].append({
            "model": model, "details": info.get("details"), "parameters": info.get("parameters"),
            "template": info.get("template"), "system": info.get("system"),
            "loaded_model": loaded,
        })
        write_json(run_dir / "manifest.json", manifest)
        raw_dir = checked_path(run_dir / "raw" / entry["family"])
        raw_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        for prompt in prompts:
            group_id = prompt["group_id"]
            seed = generation["seed_rule"]["base_seed"] + generation["seed_rule"]["repeats_per_group"] * (group_id - 1)
            payload = {
                "model": model, "prompt": prompt["prompt"], "stream": False, "keep_alive": "5m",
                "options": {**generation["options"], "seed": seed},
            }
            if entry["family"] == "qwen":
                payload["think"] = generation["qwen3_think"]
            request = {"model": model, "group_id": group_id, "repeat_id": 1, "seed": seed, "started_at_utc": now(), "payload": payload}
            append_json(run_dir / "requests.jsonl", request)
            log(f"Generating {model}, group {group_id}, repeat 1, seed {seed}")
            record = {"request": request}
            item = {"model": model, "group_id": group_id, "repeat_id": 1, "seed": seed}
            try:
                response = api(base, "/api/generate", payload, timeout=generation["request_timeout_seconds"])
                record["response"] = response
                sentences, issues = validate(response.get("response", ""), generation["expected_sentences_per_response"])
                if response.get("done_reason") != "stop":
                    issues.append(f"Generation ended with {response.get('done_reason')}")
                gpu = gpu_model(base, model)
                record.update({"validation_issues": issues, "parsed_sentences": sentences, "loaded_model": gpu})
                item.update({
                    "status": "valid" if not issues else "validation_issues", "sentence_count": len(sentences),
                    "issues": issues, "gpu_size_vram": gpu["size_vram"],
                    "done_reason": response.get("done_reason"), "prompt_eval_count": response.get("prompt_eval_count"),
                    "eval_count": response.get("eval_count"),
                })
                for sentence in sentences:
                    sentence_rows.append({**{key: item[key] for key in ["model", "group_id", "repeat_id", "seed"]}, **sentence})
            except Exception as exc:
                record["error"] = str(exc)
                item.update({"status": "request_error", "sentence_count": 0, "issues": [str(exc)]})
            record["finished_at_utc"] = now()
            write_json(raw_dir / f"group_{group_id:02d}_repeat_01.json", record)
            summary["requests"].append(item)
            summary.update({
                "finished_requests": len(summary["requests"]),
                "valid_requests": sum(row["status"] == "valid" for row in summary["requests"]),
                "parsed_sentences": len(sentence_rows), "updated_at_utc": now(),
            })
            write_json(run_dir / "summary.json", summary)
            with checked_path(run_dir / "sentences.csv").open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["model", "group_id", "repeat_id", "seed", "sentence_id", "sentence", "contains_both_languages"])
                writer.writeheader()
                writer.writerows(sentence_rows)
            log(f"Saved {model}, group {group_id}: {item['status']}; {item['sentence_count']} parsed sentences")
    manifest["finished_at_utc"] = now()
    write_json(run_dir / "manifest.json", manifest)
    summary["all_12_requests_finished"] = len(summary["requests"]) == 12
    write_json(run_dir / "summary.json", summary)
    log(f"Pilot complete: {summary['finished_requests']} requests, {summary['valid_requests']} valid, {len(sentence_rows)} parsed sentences")
    return 1 if any(item["status"] == "request_error" for item in summary["requests"]) else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, type=int)
    args = parser.parse_args()
    sys.exit(run(args.port))
