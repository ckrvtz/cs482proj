"""Restore data-cache.zip (made by scripts/pack_data.py) into data/raw and data/interim.

Files already present with identical content are skipped. Files present with different
content are left alone and listed as conflicts unless --overwrite is given. Every file is
checked against the archive's SHA-256 manifest.

    python scripts/unpack_data.py --dry-run       # show what would happen
    python scripts/unpack_data.py                 # restore missing files
    python scripts/unpack_data.py --overwrite     # also replace files that differ
    python scripts/unpack_data.py path/to/other.zip
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from desertification.config import find_project_root, get_path, load_config
from desertification.datacache import unpack


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("archive", nargs="?", help="zip to restore (default: data-cache.zip)")
    parser.add_argument("--overwrite", action="store_true", help="replace files that differ")
    parser.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = parser.parse_args()

    root = find_project_root()
    cfg = load_config()
    archive = Path(args.archive) if args.archive else get_path(cfg, "data_archive")
    if not archive.is_file():
        print(f"Archive not found: {archive}")
        return 1
    report = unpack(
        archive, root, cfg["data_cache"]["dirs"], overwrite=args.overwrite, dry_run=args.dry_run
    )
    write, over = ("Would write", "would overwrite") if args.dry_run else ("Wrote", "overwrote")
    print(
        f"{write} {len(report.written)} new files; {over} {len(report.overwritten)}; "
        f"skipped {len(report.skipped_identical)} identical."
    )
    if report.conflicts:
        print(
            f"{len(report.conflicts)} local files differ from the archive and were left alone "
            "(use --overwrite to replace them):"
        )
        for name in report.conflicts[:20]:
            print(f"  {name}")
    if report.rejected:
        print(f"Ignored {len(report.rejected)} entries outside data/raw or data/interim:")
        for name in report.rejected[:20]:
            print(f"  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
