"""Interactively save an OpenAI API key without echoing it or using shell history."""

from runtime_config import assert_project

import getpass
import os
from pathlib import Path
import pwd
import sys
import tempfile


def main():
    root = Path(__file__).resolve().parents[1]
    assert_project(root)
    if not sys.stdin.isatty():
        raise SystemExit("Run this command directly in your interactive OSC terminal.")
    directory = root / ".secrets"
    if directory.is_symlink():
        raise SystemExit("Refusing a symbolic link for the secrets directory.")
    directory.resolve().relative_to(root)
    directory.mkdir(mode=0o700, exist_ok=True)
    if directory.stat().st_uid != os.getuid():
        raise SystemExit("The secrets directory is not owned by the current user.")
    directory.chmod(0o700)
    target = directory / "openai_api_key"
    if target.is_symlink():
        raise SystemExit("Refusing a symbolic link for the key file.")
    key = getpass.getpass("Paste your OpenAI API key (hidden), then press Enter: ").strip()
    if len(key) < 20 or any(character.isspace() for character in key):
        raise SystemExit("No key saved: the pasted value looks incomplete or contains whitespace.")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory,
                                         prefix=".openai_", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(key + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o600)
        os.replace(temporary, target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    print("API key saved. Directory permissions: 700; file permissions: 600.")
    print("The key has not been printed. You can return to the chat now.")


if __name__ == "__main__":
    main()
