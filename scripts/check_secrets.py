"""Fail if any value from .env appears in a file git tracks or would commit (P0-T03).

Usage, from the repository root:

    python scripts/check_secrets.py

Checks tracked files plus untracked files that are not gitignored (so new notebooks and
outputs are covered before they are committed). Prints file names only, never the values.
Exit code 0 = no leaks found.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from desertification.config import find_project_root
from desertification.security import find_leaks, parse_env_text, secrets_to_check


def candidate_files(root: Path) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in out.stdout.splitlines() if line and line != ".env"]


def main() -> int:
    root = find_project_root()
    env_file = root / ".env"
    if not env_file.is_file():
        print("No .env file found: nothing to check.")
        return 0
    secrets = secrets_to_check(parse_env_text(env_file.read_text(encoding="utf-8")))
    if not secrets:
        print(".env has no values long enough to check.")
        return 0

    files = {}
    for rel in candidate_files(root):
        path = root / rel
        if path.is_file():
            files[rel] = path.read_bytes().decode("utf-8", errors="ignore")
    leaks = find_leaks(files, secrets)
    print(f"Checked {len(files)} files against {len(secrets)} value(s) from .env.")
    if leaks:
        print("SECRET FOUND in:")
        for rel in leaks:
            print(f"  {rel}")
        return 1
    print("No secrets found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
