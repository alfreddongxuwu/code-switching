"""Validate a user's configured cluster checkout without publishing identities.

Fill in configs/osc.json before model generation. OSC_USERNAME and
OSC_PROJECT_DIR may override those two fields for the current process.
"""

import getpass
import json
import os
from pathlib import Path


def assert_project(root):
    root = Path(root).resolve()
    config = json.loads((root / "configs/osc.json").read_text(encoding="utf-8"))
    username = os.environ.get("OSC_USERNAME", config["username"])
    project = os.environ.get("OSC_PROJECT_DIR", config["project_dir"])
    if username == "YOUR_USERNAME" or project.startswith("/path/to/"):
        raise RuntimeError("Configure your own username and project_dir in configs/osc.json before generation")
    if getpass.getuser() != username or Path(project).expanduser().resolve() != root:
        raise RuntimeError("Run only as the configured user inside the configured project directory")
