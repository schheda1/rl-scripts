"""
Remove the pipeline temp directory for the current user (or a specified path).

Usage:
  python scripts/rl/clean_tmp.py                         # removes /tmp/rl_pipeline_{user}
  python scripts/rl/clean_tmp.py --tmp-dir /tmp/my_dir   # removes specified path
  python scripts/rl/clean_tmp.py --dry-run               # show what would be deleted
"""

import argparse
import getpass
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean pipeline temp directory.")
    p.add_argument(
        "--tmp-dir",
        type=str,
        default=f"/tmp/rl_pipeline_{getpass.getuser()}",
        help="Path to the temp directory to remove (default: /tmp/rl_pipeline_{user})",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be deleted without actually deleting.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tmp_dir = Path(args.tmp_dir)

    if not tmp_dir.exists():
        print(f"Nothing to clean — {tmp_dir} does not exist.")
        return

    # Count files and total size for reporting
    files = list(tmp_dir.rglob("*"))
    n_files = sum(1 for f in files if f.is_file())
    total_bytes = sum(f.stat().st_size for f in files if f.is_file())
    total_mb = total_bytes / (1024 * 1024)

    print(f"{'[DRY RUN] ' if args.dry_run else ''}Removing: {tmp_dir}")
    print(f"  {n_files} files, {total_mb:.1f} MB")

    if args.dry_run:
        for f in sorted(files):
            print(f"  {f}")
        print("Dry run complete — nothing deleted.")
        return

    shutil.rmtree(tmp_dir)
    print("Done.")


if __name__ == "__main__":
    main()
