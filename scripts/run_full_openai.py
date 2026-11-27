"""Checkpointed GPT corpus generation: 50 initial calls, then bounded repairs."""

from runtime_config import assert_project

import fcntl
import hashlib
import json
import os
import pwd
import re
import shutil
import socket
import stat
import urllib.error
import urllib.request
import uuid

import run_full_ollama as full


def assess_response(response, expected):
    messages = [item for item in response.get("output", []) if item.get("type") == "message"]
    text = "".join(part.get("text", "") for item in messages for part in item.get("content", [])
                   if part.get("type") == "output_text")
    refusal = any(part.get("type") == "refusal" for item in messages for part in item.get("content", []))
    completed = response.get("status") == "completed" and not response.get("error") and not refusal
    assessment = full.assess({"response": text, "done_reason": "stop" if completed else response.get("status")}, expected)
    return assessment


def post_once(endpoint, payload, key, timeout):
    request = urllib.request.Request(endpoint, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as stream:
            return {"response": json.load(stream), "api_request_id": stream.headers.get("x-request-id")}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace").replace(key, "[REDACTED]")
        raw = re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED_KEY]", raw)
        try:
            detail = json.loads(raw)
        except ValueError:
            detail = {"body": raw}
        return {"operational_error": {"http_status": exc.code, "detail": detail}}
    except Exception as exc:
        return {"operational_error": {"type": type(exc).__name__}}


def save_summary(run_dir, model_dir, records, config, phase, finished=False):
    summary = full.save_model_summary(model_dir, {"model": config["model"], "family": "gpt"},
                                      records, config, repair_finished=finished)
    summary.update({"run_name": config["run_name"], "phase": phase, "run_finished": finished,
        "seed_controlled": False, "conversation_history": False, "pilot_included": False,
        "target_sentences": 500, "maximum_requests": 200,
        "operational_error_requests": sum(bool(record.get("operational_error")) for record in records.values()),
        "usage": {field: sum(record.get("response", {}).get("usage", {}).get(field, 0) for record in records.values())
                  for field in ["input_tokens", "output_tokens", "total_tokens"]}})
    full.write_json(run_dir / "summary.json", summary)
    temporary = full.checked(run_dir / "accepted.csv.tmp")
    shutil.copyfile(full.checked(model_dir / "accepted.csv"), temporary)
    os.replace(temporary, full.checked(run_dir / "accepted.csv"))
    return summary


def run(config, generation, prompts, key):
    run_dir = full.checked(full.ROOT / "outputs" / config["run_name"])
    model_dir = full.checked(run_dir / "models/gpt")
    (model_dir / "attempts").mkdir(mode=0o700, parents=True, exist_ok=True)
    with full.checked(model_dir / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        names = ["configs/openai_full_run.json", "configs/generation.json", "data/prompts.jsonl",
                 "scripts/run_full_openai.py", "scripts/run_full_ollama.py", "scripts/run_ollama_pilot.py",
                 "run/openai_full.sbatch"]
        snapshot = {"config": config, "generation": generation,
                    "file_sha256": {name: hashlib.sha256((full.ROOT / name).read_bytes()).hexdigest() for name in names}}
        frozen_path = full.checked(run_dir / "config_snapshot.json")
        if frozen_path.exists():
            if json.loads(frozen_path.read_text()) != snapshot:
                raise RuntimeError("Frozen GPT inputs changed; refusing to mix versions")
        else:
            full.write_json(frozen_path, snapshot)
        records = full.load_records(model_dir)
        if any(key[0] not in config["groups"] or key[1] not in range(1, 6) or key[2] not in range(4) for key in records):
            raise RuntimeError("Unexpected saved GPT attempts")
        ledger_ids = [record["request"]["independent_call_id"] for record in records.values()]
        if len(set(ledger_ids)) != len(ledger_ids):
            raise RuntimeError("Duplicate independent call identifiers")
        manifest = {"started_at_utc": full.now(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                    "hostname": socket.gethostname(), "model": config["model"],
                    "seed_controlled": False, "parameters": config["parameters"]}
        full.write_json(run_dir / f"manifest_{os.environ.get('SLURM_JOB_ID', 'offline')}.json", manifest)
        slots = full.all_slots(config)
        for phase in ["initial", "repair"]:
            if phase == "initial":
                work = [(g, r, 0) for g, r in slots if (g, r, 0) not in records]
            else:
                if not all((g, r, 0) in records for g, r in slots):
                    raise RuntimeError("All 50 initial attempts must finish before repair")
                work = [(g, r, attempt) for attempt in range(1, 4) for g, r in slots
                        if (g, r, attempt) not in records and full.best_attempt(records, (g, r)) is None]
            save_summary(run_dir, model_dir, records, config, phase)
            for group, repeat, attempt in work:
                if full.best_attempt(records, (group, repeat)) is not None:
                    continue
                payload = dict(config["parameters"], model=config["model"], input=prompts[group])
                request = {"model": config["model"], "group_id": group, "repeat_id": repeat,
                    "attempt_id": attempt, "seed": None, "independent_call_id": uuid.uuid4().hex,
                    "started_at_utc": full.now(), "endpoint": config["endpoint"], "payload": payload,
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
                full.append_json(model_dir / "requests.jsonl", request)
                full.log(f"Generating GPT g{group:02d} r{repeat} attempt {attempt}; independent call, no seed control")
                record = {"request": request}
                result = post_once(config["endpoint"], payload, key, generation["request_timeout_seconds"])
                record.update(result)
                if result.get("operational_error"):
                    record.update({"accepted": False, "sentences": [], "packaging": "unknown",
                        "issues": [{"type": "operational_error", "detail": result["operational_error"]}]})
                else:
                    record.update(assess_response(result["response"], config["batch_size"]))
                record["finished_at_utc"] = full.now()
                path = full.checked(model_dir / "attempts" / f"g{group:02d}_r{repeat:02d}_a{attempt:02d}.json")
                if path.exists():
                    raise RuntimeError("Refusing to overwrite an existing GPT attempt")
                full.write_json(path, record)
                records[(group, repeat, attempt)] = record
                summary = save_summary(run_dir, model_dir, records, config, phase)
                if result.get("operational_error"):
                    raise RuntimeError("OpenAI operational error recorded; stopped without automatic resend")
                full.log(f"Saved: accepted={record['accepted']}; count={len(record['sentences'])}; accepted_batches={summary['accepted_batches']}/50")
        summary = save_summary(run_dir, model_dir, records, config, "finished", finished=True)
        full.log(f"GPT full run finished: {summary['accepted_sentences']} accepted sentences; {len(summary['unresolved'])} unresolved batches")


def main():
    root = full.ROOT
    assert_project(root)
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Submit the formal run as a Slurm job")
    config = json.loads((root / "configs/openai_full_run.json").read_text())
    generation = json.loads((root / "configs/generation.json").read_text())
    if config["endpoint"] != "https://api.openai.com/v1/responses" or config["model"] != "gpt-6-luna":
        raise ValueError("Unexpected endpoint or GPT model")
    if config["groups"] != list(range(1, 11)) or config["repeats_per_group"] != 5 or config["batch_size"] != 10:
        raise ValueError("Expected ten groups, five repeats, and ten strings per batch")
    if config["max_reruns_per_batch"] != 3 or config["selection"] != "first_accepted_attempt" or config["retry_unit"] != "whole_batch":
        raise ValueError("Unexpected repair policy")
    if config["seed"] is not None or config["history"] or config["pilot_included"]:
        raise ValueError("Unexpected seed, history, or pilot inclusion setting")
    expected = {"temperature": generation["options"]["temperature"], "top_p": generation["options"]["top_p"],
                "max_output_tokens": generation["options"]["num_predict"], "reasoning": {"effort": "none"},
                "store": False, "truncation": "disabled"}
    if config["parameters"] != expected:
        raise ValueError("Unexpected generation parameters")
    if not re.fullmatch(r"[a-z][a-z0-9_]+", config["run_name"]):
        raise ValueError("Invalid run name")
    source = [json.loads(line) for line in (root / "data/prompts.jsonl").read_text().splitlines()]
    if [item["group_id"] for item in source] != config["groups"]:
        raise ValueError("Unexpected prompt groups")
    keyfile = full.checked(root / ".secrets/openai_api_key")
    if keyfile.stat().st_uid != os.getuid() or stat.S_IMODE(keyfile.stat().st_mode) != 0o600:
        raise RuntimeError("Key file must be owned by this user with permissions 600")
    key = keyfile.read_text().strip()
    if len(key) < 20 or any(character.isspace() for character in key):
        raise RuntimeError("Invalid key file")
    run(config, generation, {item["group_id"]: item["prompt"] for item in source}, key)


if __name__ == "__main__":
    main()
