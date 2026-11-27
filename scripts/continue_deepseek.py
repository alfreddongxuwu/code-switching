"""Continue only the unresolved DeepSeek batches, preserving the original run."""

from runtime_config import assert_project

import argparse
import fcntl
import hashlib
import json
import os
import pwd
import shutil
import socket
import time

import run_full_ollama as full


def continue_batches(run_dir, extension, generation, policy, entry, prompts, port):
    model_dir = full.checked(run_dir / "models/deepseek")
    extension_dir = full.checked(run_dir / "continuations" / extension["continuation_id"])
    extension_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    targets = [tuple(slot) for slot in extension["target_slots"]]
    first, last = extension["first_attempt_id"], extension["last_attempt_id"]
    if (first, last) != (4, 6) or len(set(targets)) != 11:
        raise ValueError("This continuation authorizes only attempts 4-6 for eleven slots")
    effective_policy = dict(policy, max_reruns_per_batch=last)
    source_names = ["configs/deepseek_continuation_01.json", "scripts/continue_deepseek.py", "run/deepseek_continuation.sbatch"]
    frozen = {"extension": extension, "file_sha256": {
        name: hashlib.sha256((full.ROOT / name).read_bytes()).hexdigest() for name in source_names}}
    with full.checked(model_dir / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        records = full.load_records(model_dir)
        if any(key[2] >= first and (key[:2] not in targets or key[2] > last) for key in records):
            raise RuntimeError("Unexpected continuation attempts already exist")
        snapshot_path = extension_dir / "snapshot.json"
        if snapshot_path.exists():
            snapshot = json.loads(snapshot_path.read_text())
            if snapshot["frozen"] != frozen:
                raise RuntimeError("Continuation inputs changed; refusing to mix versions")
        else:
            unresolved = [slot for slot in full.all_slots(policy) if full.best_attempt(records, slot) is None]
            if unresolved != targets or len(records) != extension["expected_prior_requests"]:
                raise RuntimeError("The run no longer matches the authorized continuation baseline")
            if any((g, r, attempt) not in records for g, r in targets for attempt in range(first)):
                raise RuntimeError("An unresolved target is missing an earlier attempt")
            summary = json.loads((model_dir / "summary.json").read_text())
            if summary["accepted_batches"] != extension["expected_prior_accepted_batches"]:
                raise RuntimeError("Unexpected prior accepted batch count")
            baseline_files = ["accepted.csv", "summary.json", "audit.json", "config_snapshot.json"]
            baseline_files += [f"models/{family}/{name}" for family in full.FAMILIES
                               for name in ["summary.json", "accepted.csv"]]
            baseline_files += ["models/deepseek/requests.jsonl", "models/deepseek/manifest_repair.json"]
            hashes = {}
            for name in baseline_files:
                source = full.checked(run_dir / name)
                target = full.checked(extension_dir / "baseline" / name)
                target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
            full.write_json(snapshot_path, {"frozen": frozen, "created_at_utc": full.now(),
                "baseline_file_sha256": hashes,
                "prior_attempt_sha256": {str(path.relative_to(run_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted((model_dir / "attempts").glob("*.json"))}})
        work = [(g, r, attempt) for attempt in range(first, last + 1) for g, r in targets
                if (g, r, attempt) not in records and full.best_attempt(records, (g, r)) is None]

        def checkpoint(finished):
            summary = full.save_model_summary(model_dir, entry, records, effective_policy, repair_finished=finished)
            full.aggregate(run_dir)
            added = [value for key, value in records.items() if key[2] >= first]
            full.write_json(extension_dir / "summary.json", {
                "continuation_id": extension["continuation_id"], "updated_at_utc": full.now(),
                "finished": finished, "new_requests": len(added), "maximum_new_requests": 33,
                "accepted_sentences_before": 390, "accepted_sentences_now": summary["accepted_sentences"],
                "unresolved": summary["unresolved"], "slurm_job_id": os.environ.get("SLURM_JOB_ID")})
            return summary

        if not work:
            checkpoint(True)
            full.log("No authorized continuation work remains")
            return
        checkpoint(False)
        base = f"http://127.0.0.1:{port}"
        for attempt in range(60):
            try:
                version = full.api(base, "/api/version", timeout=2)["version"]
                break
            except OSError:
                if attempt == 59:
                    raise
                time.sleep(1)
        info = full.api(base, "/api/show", {"model": entry["model"]})
        full.api(base, "/api/generate", {"model": entry["model"], "stream": False,
                 "keep_alive": "10m", "options": generation["options"]}, timeout=600)
        loaded = full.gpu_model(base, entry["model"])
        full.write_json(extension_dir / f"manifest_{os.environ.get('SLURM_JOB_ID', 'offline')}.json", {
            "model": entry["model"], "started_at_utc": full.now(), "ollama_version": version,
            "hostname": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "loaded_model": loaded,
            "model_details": info.get("details"), "template": info.get("template"),
            "system": info.get("system"), "model_parameters": info.get("parameters")})
        full.log(f"GPU verified: {entry['model']}, {loaded['size_vram'] / 1024**3:.2f} GiB in VRAM")
        for group, repeat, attempt in work:
            if full.best_attempt(records, (group, repeat)) is not None:
                continue
            seed = full.seed_for(group, repeat, attempt, generation, policy)
            payload = {"model": entry["model"], "prompt": prompts[group], "stream": False,
                       "keep_alive": "10m", "options": dict(generation["options"], seed=seed)}
            request = {"model": entry["model"], "group_id": group, "repeat_id": repeat,
                       "attempt_id": attempt, "seed": seed, "started_at_utc": full.now(),
                       "payload": payload, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                       "continuation_id": extension["continuation_id"]}
            full.append_json(model_dir / "requests.jsonl", request)
            full.log(f"Generating {entry['model']} g{group:02d} r{repeat} attempt {attempt} seed {seed}")
            record = {"request": request}
            try:
                response = full.api(base, "/api/generate", payload, timeout=generation["request_timeout_seconds"])
                record["response"] = response
                record.update(full.assess(response, policy["batch_size"]))
                record["loaded_model"] = full.gpu_model(base, entry["model"])
            except Exception as exc:
                record.update({"accepted": False, "sentences": [], "packaging": "unknown",
                               "issues": [{"type": "request_error", "detail": str(exc)}]})
            record["finished_at_utc"] = full.now()
            path = model_dir / "attempts" / f"g{group:02d}_r{repeat:02d}_a{attempt:02d}.json"
            if path.exists():
                raise RuntimeError("Refusing to overwrite an existing attempt")
            full.write_json(path, record)
            records[(group, repeat, attempt)] = record
            summary = checkpoint(False)
            full.log(f"Saved: accepted={record['accepted']}; accepted_batches={summary['accepted_batches']}/50")
        summary = checkpoint(True)
        full.log(f"Continuation finished: {summary['accepted_sentences']} accepted sentences; {len(summary['unresolved'])} unresolved batches")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    config, generation, policy, models, prompts = full.read_inputs()
    extension = json.loads((full.ROOT / "configs/deepseek_continuation_01.json").read_text())
    assert_project(full.ROOT)
    if config["project_dir"] != str(full.ROOT) or extension["run_name"] != policy["run_name"]:
        raise RuntimeError("Unexpected project or run")
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("A Slurm GPU allocation is required")
    if os.environ.get("OLLAMA_MODELS") != config["ollama_models_dir"] or not 1024 <= args.port <= 65535:
        raise RuntimeError("Unexpected model store or server port")
    entry = models[3]
    if entry["model"] != extension["model"] or entry["family"] != "deepseek":
        raise RuntimeError("Only DeepSeek continuation is authorized")
    run_dir = full.checked(full.ROOT / "outputs" / policy["run_name"])
    full.freeze_inputs(run_dir, config, generation, policy)
    continue_batches(run_dir, extension, generation, policy, entry, prompts, args.port)


if __name__ == "__main__":
    main()
