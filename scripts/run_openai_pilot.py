"""Make at most one OpenAI pilot request, keeping credentials out of outputs."""

from runtime_config import assert_project

import fcntl
import hashlib
import json
import os
import pwd
import re
import stat
import urllib.error
import urllib.request

import run_full_ollama as full


def main():
    root = full.ROOT
    assert_project(root)
    config = json.loads((root / "configs/openai_pilot.json").read_text())
    if config["endpoint"] != "https://api.openai.com/v1/responses" or config["model"] != "gpt-6-luna":
        raise ValueError("Unexpected OpenAI endpoint or model")
    if config["automatic_retries"] != 0 or config["group_id"] != 1 or config["repeat_id"] != 1:
        raise ValueError("Only one first-group pilot request is authorized")
    generation = json.loads((root / "configs/generation.json").read_text())
    parameters = config["parameters"]
    if (parameters["temperature"] != generation["options"]["temperature"]
            or parameters["top_p"] != generation["options"]["top_p"]
            or parameters["max_output_tokens"] != generation["options"]["num_predict"]
            or parameters["reasoning"] != {"effort": "none"}):
        raise ValueError("Pilot settings do not match the chosen generation settings")
    prompts = [json.loads(line) for line in (root / "data/prompts.jsonl").read_text().splitlines()]
    prompt = next(item["prompt"] for item in prompts if item["group_id"] == config["group_id"])
    payload = dict(parameters, model=config["model"], input=prompt)
    if set(payload) != {"model", "input", "temperature", "top_p", "max_output_tokens", "reasoning", "store", "truncation"}:
        raise ValueError("Unexpected request parameters")
    if payload["store"] is not False or payload["truncation"] != "disabled":
        raise ValueError("Unexpected storage or context truncation settings")
    keyfile = full.checked(root / ".secrets/openai_api_key")
    if keyfile.stat().st_uid != os.getuid() or stat.S_IMODE(keyfile.stat().st_mode) != 0o600:
        raise RuntimeError("Key file must be owned by this user with permissions 600")
    key = keyfile.read_text().strip()
    if len(key) < 20 or any(character.isspace() for character in key):
        raise RuntimeError("Invalid key file")
    run_dir = full.checked(root / "outputs" / config["run_name"])
    run_dir.mkdir(mode=0o700, exist_ok=True)
    with full.checked(run_dir / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = full.checked(run_dir / "requests.jsonl")
        if ledger.exists() and ledger.stat().st_size:
            raise RuntimeError("Pilot request already dispatched; inspect saved results instead of repeating it")
        names = ["configs/openai_pilot.json", "configs/generation.json", "data/prompts.jsonl",
                 "scripts/run_openai_pilot.py", "scripts/run_full_ollama.py", "scripts/run_ollama_pilot.py"]
        full.write_json(run_dir / "config_snapshot.json", {
            "config": config, "created_at_utc": full.now(),
            "file_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}})
        request_record = {"started_at_utc": full.now(), "endpoint": config["endpoint"],
                          "group_id": 1, "repeat_id": 1, "seed": None, "payload": payload}
        full.append_json(ledger, request_record)
        full.write_json(run_dir / "request.json", request_record)
        request = urllib.request.Request(config["endpoint"],
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=generation["request_timeout_seconds"]) as stream:
                response = json.load(stream)
                request_id = stream.headers.get("x-request-id")
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace").replace(key, "[REDACTED]")
            raw = re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED_KEY]", raw)
            try:
                detail = json.loads(raw)
            except ValueError:
                detail = {"body": raw}
            full.write_json(run_dir / "error.json", {"http_status": exc.code, "finished_at_utc": full.now(), "detail": detail})
            full.write_json(run_dir / "summary.json", {"api_success": False, "http_status": exc.code,
                "automatic_retries": 0, "finished_at_utc": full.now()})
            full.log(f"OpenAI pilot failed: HTTP {exc.code}. Error recorded without credentials; no retry issued.")
            return
        except Exception as exc:
            full.write_json(run_dir / "error.json", {"type": type(exc).__name__, "finished_at_utc": full.now()})
            full.log(f"OpenAI pilot connection failed: {type(exc).__name__}; no retry issued.")
            return
        full.write_json(run_dir / "response.json", response)
        text = "".join(content.get("text", "") for item in response.get("output", [])
                       if item.get("type") == "message" for content in item.get("content", [])
                       if content.get("type") == "output_text")
        refusal = any(content.get("type") == "refusal" for item in response.get("output", [])
                      if item.get("type") == "message" for content in item.get("content", []))
        complete = response.get("status") == "completed" and not response.get("error") and not refusal
        assessment = full.assess({"response": text, "done_reason": "stop" if complete else response.get("status")},
                                 config["expected_sentences"])
        full.write_json(run_dir / "assessment.json", assessment)
        full.checked(run_dir / "responses.txt").write_text(text + "\n", encoding="utf-8")
        summary = {"finished_at_utc": full.now(), "api_success": True, "response_status": response.get("status"),
            "requested_model": config["model"], "returned_model": response.get("model"),
            "response_id": response.get("id"), "request_id": request_id, "usage": response.get("usage"),
            "sentence_count": len(assessment["sentences"]), "accepted": assessment["accepted"],
            "issues": assessment["issues"], "seed_controlled": False,
            "effective_settings": {name: response.get(name) for name in parameters},
            "automatic_retries": 0, "included_in_formal_corpus": False}
        full.write_json(run_dir / "summary.json", summary)
        full.log(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
