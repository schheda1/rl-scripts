"""
Extract the published heuristic's per-loop decisions, by compilation only.

The heuristic announces every decision on the compiler's STDERR:

    UnrollAndUnmergeHeuristic::<loopIdx>;<unroll_factor>

(that is how the original artifact captures them — executors/executor.py calls
parse_simple_heuristic_output(err) on the build output). So its choices can be
recovered with a compile and a regex. No benchmark execution, no GPU timing, no
change to the LLVM pass.

WHY THAT MATTERS
----------------
The reward table already holds the measured value of every (loop, unmerge,
factor) cell. Once the heuristic's decisions are known, its quality follows from
a table lookup:

    fired where no-op was optimal   -> harmful firing
    declined where a transform won  -> miss
    fired on the right category     -> compare its factor's reward to the oracle

That makes the heuristic evaluable on all ~138 eligible benchmarks against
ground truth, where the original study reported 16 with none.

INDEX SCOPING — the reason this compiles per FILE
-------------------------------------------------
The original harness passes `--enable-uu-heuristic` alone, so the pass runs over
the whole build and its printed loopIdx is numbered per translation unit; on a
multi-file benchmark those indices collide. The RL pipeline always scopes with
--uu-match-filename / --uu-match-targettriple, and its loop_records are numbered
under that scoping. To make the two directly comparable this compiles once per
(filename, triple) that owns eligible loops, with the SAME match flags.

If the heuristic ignores those flags, the symptom is decisions whose loop_idx is
absent from that file's loop_records — reported as `in_records=0`, and summarised
at the end. A high count there means the scoping assumption is wrong and the
lookup must not be trusted; see --no-scope to test the unscoped behaviour.

Usage:
  python3 heuristic_decisions.py RUN_DIR --out heuristic_decisions.csv
  python3 heuristic_decisions.py RUN_DIR --benchmarks contract-cuda mandelbrot-cuda
"""

import argparse
import csv
import json
import logging
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hecbench import ARCH, HECBENCH_SRC, _make, discover_benchmarks

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("heuristic")

# MIRROR: executors/executor.py:parse_simple_heuristic_output
PREFIX = "UnrollAndUnmergeHeuristic::"
# clang accepts unknown -mllvm options with only a warning, so a mis-ported flag
# name would produce zero decisions everywhere — indistinguishable from "the
# heuristic declined everything". Detect it rather than report a silent no-op.
_UNKNOWN_ARG = re.compile(r"Unknown command line argument|did you mean", re.I)
LINE = re.compile(r"^UnrollAndUnmergeHeuristic::(\d+);(-?\d+)\s*$")
FIELDS = ["benchmark", "filename", "triple", "loop_idx", "factor",
          "in_records", "compile_ok"]


def _write(path, rows) -> None:
    """Rewrite the whole CSV. Called per benchmark so a killed run keeps its work."""
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def parse_decisions(stderr: str) -> list:
    """[(loop_idx, factor)] announced by the heuristic during this compile."""
    out = []
    for line in (stderr or "").splitlines():
        m = LINE.match(line.strip())
        if m:
            out.append((int(m.group(1)), int(m.group(2))))
        elif line.strip().startswith(PREFIX):
            # Same prefix, unexpected payload — surface it rather than drop it.
            log.warning("unparsed heuristic line: %s", line.strip()[:120])
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path,
                   help="Run directory holding eligible_benchmarks.json")
    p.add_argument("--benchmarks", nargs="+", default=None,
                   help="Restrict to these benchmark names (default: all eligible)")
    p.add_argument("--out", type=Path, default=None,
                   help="Output CSV (default: RUN_DIR/heuristic_decisions.csv)")
    p.add_argument("--arch", default=ARCH)
    p.add_argument("--hecbench-src", default=None,
                   help="MUST be the same tree the reward cache was measured "
                        "on, or the loop indices describe different code.")
    p.add_argument("--no-scope", action="store_true",
                   help="Compile once per benchmark with a bare "
                        "--enable-uu-heuristic, as the original harness does, "
                        "instead of once per file with the match flags. Use to "
                        "test whether the pass honours the scoping at all.")
    p.add_argument("--timeout", type=int, default=600,
                   help="Per-compile timeout in seconds (default: 600)")
    args = p.parse_args()

    elig_f = args.run_dir / "eligible_benchmarks.json"
    if not elig_f.exists():
        sys.exit(f"missing {elig_f}")
    elig = json.loads(elig_f.read_text())
    eligible = set(elig.get("eligible", []))
    records = elig.get("loop_records", {})
    if args.benchmarks:
        eligible &= set(args.benchmarks)
        missing = set(args.benchmarks) - eligible
        if missing:
            log.warning("not eligible, skipped: %s", sorted(missing))
    if not eligible:
        sys.exit("no eligible benchmarks selected")

    src = Path(args.hecbench_src) if args.hecbench_src else HECBENCH_SRC
    paths = {b.name: b for b in discover_benchmarks(src)}
    log.info("Source tree: %s", src)
    log.info("%d eligible benchmarks; %d found in the tree",
             len(eligible), len(eligible & set(paths)))

    workdir = Path(tempfile.mkdtemp(prefix="uu_heuristic_"))
    rows, n_compiles, n_failed_compiles = [], 0, 0
    unknown_flag = [False]
    out = args.out or (args.run_dir / "heuristic_decisions.csv")

    for bench in sorted(eligible):
        bpath = paths.get(bench)
        if bpath is None:
            log.warning("%s not found in the source tree — skipped", bench)
            continue
        recs = records.get(bench, [])
        # (filename, triple) pairs that own eligible loops, and the loop indices
        # each one owns — used both to scope the compile and to check the
        # returned indices are ones we know about.
        scopes: dict = {}
        for r in recs:
            scopes.setdefault((r["filename"], r["triple"]), set()).add(
                int(r["loop_idx"]))
        if args.no_scope:
            scopes = {("", ""): {int(r["loop_idx"]) for r in recs}}
        if not scopes:
            log.warning("%s has no loop_records — skipped", bench)
            continue

        work = workdir / bench
        if work.exists():
            shutil.rmtree(work)
        shutil.copytree(bpath, work)

        for (fname, triple), known in sorted(scopes.items()):
            flags = "-mllvm --enable-uu-heuristic"
            if fname:
                flags += f" -mllvm --uu-match-filename={fname}"
            if triple:
                flags += f" -mllvm --uu-match-targettriple={triple}"
            try:
                r = _make(work, extra_cflags=flags, arch=args.arch,
                          timeout=args.timeout)
                stderr, ok = r.stderr, r.returncode == 0
            except subprocess.TimeoutExpired as e:
                # Decisions are printed DURING compilation, so a timeout may
                # still carry a complete set — keep whatever was emitted.
                stderr = (e.stderr or b"").decode(errors="replace") \
                    if isinstance(e.stderr, bytes) else (e.stderr or "")
                ok = False
                log.warning("%s [%s] compile TIMED OUT after %ds", bench,
                            fname or "whole build", args.timeout)
            n_compiles += 1
            if not ok:
                n_failed_compiles += 1
            if _UNKNOWN_ARG.search(stderr or ""):
                unknown_flag[0] = True
            decisions = parse_decisions(stderr)
            log.info("%-32s %-22s %2d decision(s)%s", bench,
                     fname or "(unscoped)", len(decisions),
                     "" if ok else "  [compile failed]")
            if not decisions:
                # A benchmark the heuristic declines entirely is a RESULT, not a
                # gap — record it so the "declined everywhere" case is countable.
                rows.append({"benchmark": bench, "filename": fname,
                             "triple": triple, "loop_idx": "", "factor": "",
                             "in_records": "", "compile_ok": int(ok)})
            for li, fac in decisions:
                rows.append({"benchmark": bench, "filename": fname,
                             "triple": triple, "loop_idx": li, "factor": fac,
                             "in_records": int(li in known),
                             "compile_ok": int(ok)})
        shutil.rmtree(work, ignore_errors=True)
        _write(out, rows)      # flush per benchmark: a 138-benchmark run must
                               # survive being killed partway

    _write(out, rows)

    fired = [r for r in rows if r["loop_idx"] != ""]
    unknown = [r for r in fired if r["in_records"] == 0]
    benches_fired = {r["benchmark"] for r in fired}
    factors = sorted({r["factor"] for r in fired})
    log.info("")
    log.info("=== summary ===")
    log.info("  compiles run          : %d  (%d failed)", n_compiles,
             n_failed_compiles)
    log.info("  decisions found       : %d across %d benchmark(s)",
             len(fired), len(benches_fired))
    log.info("  benchmarks with none  : %d",
             len({r['benchmark'] for r in rows}) - len(benches_fired))
    log.info("  distinct factors used : %s", factors)
    if unknown:
        log.warning("  loop_idx NOT in loop_records: %d of %d (%.0f%%)",
                    len(unknown), len(fired), 100 * len(unknown) / len(fired))
        log.warning("  -> the scoping assumption may be wrong; do NOT join "
                    "these against the reward cache until this is understood.")
        for r in unknown[:8]:
            log.warning("     %s %s loop_idx=%s", r["benchmark"], r["filename"],
                        r["loop_idx"])
    else:
        log.info("  every loop_idx matched loop_records — indices are aligned")
    if not fired:
        log.error("  NO DECISIONS AT ALL across every benchmark.")
        log.error("  Before reading this as 'the heuristic declines "
                  "everything', confirm the pass is actually running: clang "
                  "accepts an unknown -mllvm flag with a warning, so a renamed "
                  "flag looks exactly like this.")
    if unknown_flag[0]:
        log.error("  clang reported an UNKNOWN command line argument — "
                  "--enable-uu-heuristic is probably not the flag this LLVM "
                  "build exposes. Results are meaningless until that is fixed.")
    log.info("  written: %s", out)
    shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
