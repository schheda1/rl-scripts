#!/usr/bin/env python3
"""
Patch HeCBench *-cuda benchmarks that have reference.h to guard CPU
verification code under #ifdef VERIFY / #endif.

Semantics:
  - Without -DVERIFY (default, training mode): CPU reference is skipped entirely.
    The GPU kernel still runs and can be timed.
  - With    -DVERIFY (correctness mode):        CPU reference runs and GPU output
    is validated against it.

What gets guarded:
  1. The  #include "reference.h"  line in every .cu file that includes it.
  2. Every call site of a function declared/defined in reference.h, plus the
     immediately following verification block (comparison loops, asserts,
     validate_result calls, PASS/FAIL prints, early-return on failure).

Already-guarded benchmarks (those already using #ifdef VERIFY, e.g. ace-cuda)
are detected and skipped automatically.

Usage:
    python patch_verify.py                             # patch all *-cuda
    python patch_verify.py --dry-run                   # preview only
    python patch_verify.py --bench entropy-cuda        # single benchmark
    python patch_verify.py --hecbench-src /path/to/src
"""

import argparse
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GUARD_OPEN  = "#ifdef VERIFY\n"
GUARD_CLOSE = "#endif // VERIFY\n"

# Identifiers that look like function names but aren't reference calls.
_SKIP_NAMES = {
    "if", "for", "while", "switch", "return", "sizeof", "decltype",
    "static_assert", "assert", "min", "max", "abs", "printf", "fprintf",
    "malloc", "free", "memcpy", "memset", "cudaMalloc", "cudaFree",
    "cudaMemcpy", "std", "void", "int", "float", "double", "bool",
}

# Lines containing these patterns signal the END of a verification block.
_BLOCK_END_RE = re.compile(
    r'printf\s*\(.*\b(PASS|FAIL|pass|fail|error|Error|ok(?!\w)|OK|'
    r'correct|incorrect|mismatch|wrong|failed|passed)\b'
    r'|validate_result\s*\('
    r'|checkResult\s*\('
    r'|check_result\s*\('
    r'|verify_result\s*\('
    r'|\breturn\s+(?:1|-1|false|true|error|err|EXIT_FAILURE)\s*;'
    r'|\bfprintf\s*\(\s*stderr',
    re.IGNORECASE,
)

# Lines containing these signal the START of unrelated (GPU) code — stop scanning.
_GPU_CODE_RE = re.compile(
    r'\bcuda(?:Malloc|Memcpy|Free|Event|Stream|DeviceSync|Launch)\b'
    r'|<<<\s*\w'
    r'|\b__global__\b'
    r'|\bcublas\b'
    r'|\bcusparse\b',
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# reference.h parsing
# ---------------------------------------------------------------------------

def extract_fn_names(ref_h: Path) -> list[str]:
    """
    Return the set of callable function names defined in reference.h.

    Strategy: collect every identifier that immediately precedes '(' on a
    non-comment, non-preprocessor line.  Filter out known non-function names.
    This is intentionally broad — false positives are harmless (we only act
    when the name also appears as a call in main.cu).
    """
    content = ref_h.read_text(errors="replace")
    names: set[str] = set()
    for line in content.splitlines():
        s = line.strip()
        if not s or s.startswith("//") or s.startswith("*") or s.startswith("#"):
            continue
        for m in re.finditer(r'\b([A-Za-z_]\w*)\s*\(', s):
            name = m.group(1)
            if name not in _SKIP_NAMES and len(name) > 2:
                names.add(name)
    return sorted(names)


# ---------------------------------------------------------------------------
# Source-file helpers
# ---------------------------------------------------------------------------

def _is_comment_or_blank(line: str) -> bool:
    s = line.strip()
    return not s or s.startswith("//") or s.startswith("*") or s.startswith("/*")


def _ifdef_depth_at(lines: list[str], idx: int) -> int:
    """Return the #ifdef VERIFY nesting depth at line idx (0 = not inside)."""
    depth = 0
    for i in range(idx):
        s = lines[i].strip()
        if re.match(r"#ifdef\s+VERIFY|#if\s+defined\s*\(\s*VERIFY\s*\)", s):
            depth += 1
        elif s.startswith("#endif") and depth > 0:
            depth -= 1
    return depth


def _is_call(line: str, fn: str) -> bool:
    """True if `line` contains a call to `fn` (not a definition or comment)."""
    s = line.strip()
    if _is_comment_or_blank(s):
        return False
    # Must have fn followed by ( somewhere on the line
    if not re.search(r'\b' + re.escape(fn) + r'\s*\(', line):
        return False
    # Exclude function definitions:  <type> fn(
    if re.search(
        r'(?:void|int|float|double|bool|auto|static|inline|template)\s[^;]*\b'
        + re.escape(fn) + r'\s*\(',
        line,
    ):
        return False
    return True


def _find_block_end(lines: list[str], start: int) -> int:
    """
    Scan forward from `start` to find the last line of the verification block.

    Stops at:
      - A PASS/FAIL printf, validate_result, return-on-failure, fprintf(stderr
      - Two consecutive blank lines after at least 3 lines scanned
      - A line containing GPU API calls (indicates we've left the verify section)
      - Hard limit: 50 lines from start
    """
    n = len(lines)
    limit = min(start + 50, n)
    end = start
    blank_run = 0
    scanned = 0

    for i in range(start, limit):
        line = lines[i]
        s = line.strip()
        scanned += 1

        # GPU code means we've left the verification section
        if _GPU_CODE_RE.search(line) and scanned > 1:
            break

        if not s:
            blank_run += 1
            if blank_run >= 2 and scanned > 4:
                break
        else:
            blank_run = 0

        end = i

        if _BLOCK_END_RE.search(line):
            # Consume one more closing brace if it immediately follows
            for j in range(i + 1, min(i + 4, n)):
                if lines[j].strip() == "}":
                    end = j
                elif lines[j].strip() and not _is_comment_or_blank(lines[j]):
                    break
            return end

    return end


# ---------------------------------------------------------------------------
# Core patcher
# ---------------------------------------------------------------------------

def patch_file(src: Path, fn_names: list[str], dry_run: bool) -> list[str]:
    """
    Patch a single source file.  Returns a list of human-readable change
    descriptions (empty list = no changes made).
    """
    original = src.read_text(errors="replace")
    lines = original.splitlines(keepends=True)
    changes: list[str] = []

    # ------------------------------------------------------------------
    # Pass 1: guard  #include "reference.h"
    # ------------------------------------------------------------------
    new_lines: list[str] = []
    for i, line in enumerate(lines):
        if re.match(r'\s*#include\s+"reference\.h"\s*', line):
            prev = lines[i - 1].strip() if i > 0 else ""
            if "#ifdef VERIFY" in prev or "defined(VERIFY)" in prev:
                new_lines.append(line)  # already guarded
            else:
                new_lines.append(GUARD_OPEN)
                new_lines.append(line)
                new_lines.append(GUARD_CLOSE)
                changes.append(f'  guarded #include "reference.h" (line {i + 1})')
        else:
            new_lines.append(line)
    lines = new_lines

    # ------------------------------------------------------------------
    # Pass 2: collect call-site blocks (bottom-up to preserve indices)
    # ------------------------------------------------------------------
    blocks: list[tuple[int, int, str]] = []  # (start, end, fn_name)

    for fn in fn_names:
        for idx, line in enumerate(lines):
            if not _is_call(line, fn):
                continue
            if _ifdef_depth_at(lines, idx) > 0:
                continue  # already inside #ifdef VERIFY
            end = _find_block_end(lines, idx)
            blocks.append((idx, end, fn))

    # Merge overlapping / adjacent blocks
    blocks.sort()
    merged: list[tuple[int, int, str]] = []
    for start, end, fn in blocks:
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end), merged[-1][2])
        else:
            merged.append((start, end, fn))

    # Insert guards from bottom to top
    for start, end, fn in reversed(merged):
        if _ifdef_depth_at(lines, start) > 0:
            continue
        lines.insert(end + 1, GUARD_CLOSE)
        lines.insert(start, GUARD_OPEN)
        changes.append(
            f"  guarded call to {fn}() — lines {start + 1}–{end + 1}"
        )

    # ------------------------------------------------------------------
    # Write if changed
    # ------------------------------------------------------------------
    new_content = "".join(lines)
    if new_content != original:
        if not dry_run:
            src.write_text(new_content)
    else:
        changes = []  # no net change

    return changes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Guard CPU verification in HeCBench *-cuda benchmarks under #ifdef VERIFY"
    )
    ap.add_argument(
        "--hecbench-src",
        default="HeCBench/src",
        help="Path to HeCBench/src (default: HeCBench/src)",
    )
    ap.add_argument(
        "--bench",
        default=None,
        help="Process only this benchmark directory name (e.g. entropy-cuda)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing any files",
    )
    args = ap.parse_args()

    src_dir = Path(args.hecbench_src)
    if not src_dir.is_dir():
        sys.exit(f"ERROR: {src_dir} is not a directory")

    if args.bench:
        candidates = [src_dir / args.bench]
    else:
        candidates = sorted(src_dir.glob("*-cuda"))

    patched = 0
    already_done = 0
    no_ref_h = 0
    uncertain: list[str] = []

    for bench_dir in candidates:
        if not bench_dir.is_dir():
            print(f"[SKIP] {bench_dir.name} — not a directory")
            continue

        ref_h = bench_dir / "reference.h"
        if not ref_h.exists():
            no_ref_h += 1
            continue

        # Find .cu files that include reference.h
        cu_files = [
            f for f in bench_dir.glob("*.cu")
            if '"reference.h"' in f.read_text(errors="replace")
        ]
        if not cu_files:
            print(f"[WARN]  {bench_dir.name} — reference.h exists but not #included in any .cu")
            uncertain.append(bench_dir.name)
            continue

        fn_names = extract_fn_names(ref_h)
        if not fn_names:
            print(f"[WARN]  {bench_dir.name} — could not extract function names from reference.h")
            uncertain.append(bench_dir.name)
            continue

        bench_changes: list[str] = []
        for cu in cu_files:
            file_changes = patch_file(cu, fn_names, dry_run=args.dry_run)
            if file_changes:
                bench_changes.append(f"  {cu.name}:")
                bench_changes.extend(file_changes)

        if bench_changes:
            tag = "[DRY]  " if args.dry_run else "[PATCH]"
            print(f"{tag} {bench_dir.name}")
            for c in bench_changes:
                print(c)
            patched += 1
        else:
            already_done += 1

    print()
    print(f"Results:")
    print(f"  Patched:        {patched}")
    print(f"  Already done:   {already_done}")
    print(f"  No reference.h: {no_ref_h}  (skipped — not Tier 1)")
    if uncertain:
        print(f"  Needs review:   {len(uncertain)}")
        for u in uncertain:
            print(f"    {u}")
    if args.dry_run:
        print()
        print("DRY RUN — no files were written.")


if __name__ == "__main__":
    main()
