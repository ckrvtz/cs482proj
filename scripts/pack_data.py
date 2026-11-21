"""Zip the local data cache (data/raw, data/interim) into data-cache.zip at the repo root.

Use this to move downloaded data to another machine without re-downloading it.
The archive is gitignored and never pushed. Restore it with scripts/unpack_data.py.

    python scripts/pack_data.py            # refuses to replace an existing archive
    python scripts/pack_data.py --force    # replace it
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from desertification.config import find_project_root, get_path, load_config
from desertification.datacache import pack


def git_commit(root) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="replace an existing archive")
    args = parser.parse_args()

    root = find_project_root()
    cfg = load_config()
    cache_dirs = cfg["data_cache"]["dirs"]
    archive = get_path(cfg, "data_archive")
    try:
        manifest = pack(root, archive, cache_dirs, git_commit(root), overwrite=args.force)
    except FileExistsError:
        print(f"{archive.name} already exists. Use --force to replace it.")
        return 1
    size_mb = archive.stat().st_size / 1e6
    print(
        f"Packed {manifest['n_files']} files ({manifest['total_bytes'] / 1e6:.1f} MB) from "
        f"{', '.join(cache_dirs)} into {archive.name} ({size_mb:.1f} MB)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
