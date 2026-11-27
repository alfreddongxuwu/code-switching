"""Download the configured Ollama models and record verified local digests.

Run on OSC through run/download_models.sbatch. No inference requests are made.
"""

from runtime_config import assert_project

import argparse
import hashlib
import json
import os
import pwd
import re
import shutil
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_ROOT = ROOT


def log(message):
    print(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {message}", flush=True)


def inside_project(path):
    path = Path(path)
    path.resolve().relative_to(ROOT)
    if path.is_symlink():
        raise ValueError(f"Refusing symbolic link: {path}")
    return path


def api(base, endpoint, payload=None, timeout=120):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        base + endpoint, data=data, headers={"Content-Type": "application/json"}
    )
    return urllib.request.urlopen(request, timeout=timeout)


def pull(base, model):
    for attempt in range(3):
        try:
            previous = None
            final_status = None
            with api(base, "/api/pull", {"model": model, "stream": True}, timeout=600) as response:
                for raw in response:
                    event = json.loads(raw)
                    if event.get("error"):
                        raise RuntimeError(event["error"])
                    status = event.get("status", "")
                    total = event.get("total", 0)
                    progress = int(10 * event.get("completed", 0) / total) if total else None
                    marker = (status, event.get("digest"), progress)
                    if marker != previous:
                        suffix = f" {progress * 10}%" if progress is not None else ""
                        log(f"{model}: {status}{suffix}")
                        previous = marker
                    final_status = status
            if final_status != "success":
                raise RuntimeError(f"Pull ended without success: {final_status}")
            return
        except Exception as exc:
            log(f"{model}: attempt {attempt + 1} failed: {exc}")
            if attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))


def verify(store, model, checked):
    family, tag = model.split(":", 1)
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", family + tag):
        raise ValueError(f"Unsupported model name: {model}")
    manifest = inside_project(store / "manifests/registry.ollama.ai/library" / family / tag)
    manifest_bytes = manifest.read_bytes()
    document = json.loads(manifest_bytes)
    descriptors = [document["config"]] + document["layers"]
    blobs = []
    for descriptor in descriptors:
        digest = descriptor["digest"]
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            raise ValueError(f"Unexpected blob digest: {digest}")
        blob = inside_project(store / "blobs" / digest.replace(":", "-"))
        size = blob.stat().st_size
        if size != descriptor["size"]:
            raise ValueError(f"Incorrect file size: {blob}")
        if digest not in checked:
            hasher = hashlib.sha256()
            with blob.open("rb") as stream:
                for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    hasher.update(chunk)
            if hasher.hexdigest() != digest.split(":", 1)[1]:
                raise ValueError(f"SHA-256 mismatch: {blob}")
            checked.add(digest)
        blobs.append({"digest": digest, "bytes": size, "sha256_verified": True})
    return {
        "model": model,
        "status": "verified",
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "bytes": sum(blob["bytes"] for blob in blobs),
        "blobs": blobs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    assert_project(ROOT)
    config = json.loads((ROOT / "configs/osc.json").read_text())
    if config["project_dir"] != str(ROOT):
        raise ValueError("Configured project directory does not match this checkout.")
    store = inside_project(config["ollama_models_dir"])
    if store != ROOT / "ollama_models":
        raise ValueError("Model store must be this project's ollama_models directory.")
    store.mkdir(mode=0o700, exist_ok=True)
    models = [item["model"] for item in config["models"] if item["provider"] == "ollama"]
    if len(models) != 4 or len(set(models)) != 4:
        raise ValueError("Expected the four configured Ollama models.")
    report_dir = inside_project(ROOT / config.get("model_inventory_dir", "outputs/model_downloads"))
    report_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    report_path = inside_project(report_dir / "inventory.json")
    report = {"checked_at_utc": None, "model_store": str(store), "models": []}
    base = None
    if not args.verify_only:
        if not args.port or not 1024 <= args.port <= 65535:
            raise ValueError("Provide the local Ollama server port.")
        if os.environ.get("OLLAMA_MODELS") != str(store):
            raise ValueError("OLLAMA_MODELS must explicitly name this project's model store.")
        minimum_free_gib = config.get("download_min_free_gib", 25)
        if shutil.disk_usage(store).free < minimum_free_gib * 1024**3:
            raise RuntimeError(f"Less than {minimum_free_gib} GiB free space available for model downloads.")
        base = f"http://127.0.0.1:{args.port}"
        for attempt in range(60):
            try:
                with api(base, "/api/version", timeout=2) as response:
                    report["ollama_version"] = json.load(response)["version"]
                break
            except OSError:
                if attempt == 59:
                    raise
                time.sleep(1)
    checked = set()
    for model in models:
        try:
            if base:
                pull(base, model)
            log(f"Verifying all SHA-256 digests for {model}")
            result = verify(store, model, checked)
            if base:
                with api(base, "/api/show", {"model": model}) as response:
                    result["details"] = json.load(response).get("details", {})
            log(f"Verified {model}: {result['bytes'] / 1024**3:.2f} GiB")
        except Exception as exc:
            result = {"model": model, "status": "error", "error": str(exc)}
            log(f"ERROR {model}: {exc}")
        report["models"].append(result)
        report["checked_at_utc"] = datetime.now(timezone.utc).isoformat()
        temporary = inside_project(report_path.with_suffix(".json.tmp"))
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        os.replace(temporary, report_path)
    log(f"Inventory saved to {report_path}")
    return 0 if all(item["status"] == "verified" for item in report["models"]) else 1


if __name__ == "__main__":
    sys.exit(main())
