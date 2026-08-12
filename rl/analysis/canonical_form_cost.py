"""
Which downstream optimisations does UU switch off, and what did the loop gain?

One question, one table. For every eligible loop whose ORACLE-BEST action is a
transform (265 of 443; the other 178 are best left alone and have nothing to
compare), compile it twice and diff the optimisation remarks:

    1. baseline   plain -O3                       -> what fires normally
    2. best       -O3 + UU at the oracle action   -> what still fires

`opts_lost` = Passed remarks present at baseline and absent under UU: the
optimisations this transform cost. Beside it sits `gain`, the measured speedup
of that same action — so the table reads "it disabled N optimisations and was
still worth +X%".

530 compiles, CPU only, no GPU.

Scope, deliberately: no control compile (that is a different question — whether
merely ENABLING the pass breaks the build — and analyze_compile_failures.py
already answers it); no sampling; no outcome classes. One action per loop, the
one the oracle says is right.

The oracle action comes from `offline_data.oracle_of_gated`, which is the same
deadzone rule `label_loops.py` uses. Not re-derived here: a second definition
that drifts would put this table's population out of step with every reported
number. Every loop in scope therefore gained more than the deadzone by
construction — `gain` says how much, and is never a yes/no.

Usage (on the server, where LLVM and HeCBench live):
  TARGET_ARCH=sm_80 python3 canonical_form_cost.py \\
      --run-dir /path/to/run_sweep_1 --hecbench-src /path/to/HeCBench/src \\
      --out-dir uu_canonical
"""

import argparse
import csv
import json
import logging
import shutil
import subprocess
import statistics as st
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from analyze_compile_failures import _compile, parse_remarks     # noqa: E402
from hecbench import (ARCH, HECBENCH_SRC, _build_extra_cflags,   # noqa: E402
                      discover_benchmarks)
from offline_data import NOOP, oracle_of_gated                   # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("uu-canonical")

FAILURE_VALUES = (-0.16, -0.161)


def is_failure(v: float) -> bool:
    return any(abs(v - x) < 1e-9 for x in FAILURE_VALUES)


def scope(run_dir: Path, deadzone: float) -> list:
    """
    One case per loop whose gated-oracle best action is a transform.

    A compile failure is dropped from the table before the oracle is taken: it is
    stored at the failure penalty, so leaving it in cannot change the argmax, but
    it could be returned AS the best action on a loop where nothing else ran.
    The free no-op is injected at exactly 0.0, as build_tables does — omitting it
    would make the oracle claim a transform is required where declining already
    wins.
    """
    rewards = json.loads((run_dir / "reward_cache.json").read_text())["rewards"]
    elig = json.loads((run_dir / "eligible_benchmarks.json").read_text())
    keep = set(elig.get("eligible", []))

    # filename and triple are REQUIRED: the UU pass is scoped by
    # -uu-match-filename / -uu-match-targettriple, and without them the loop
    # index also matches same-numbered loops in other TUs and on the host
    # sub-compilation — a different transform from the measured one.
    meta = {}
    for bench, recs in elig.get("loop_records", {}).items():
        if bench in keep:
            for rec in recs:
                meta[(bench, int(rec["loop_idx"]))] = (rec.get("filename", ""),
                                                       rec.get("triple", ""))

    tables = {}
    for key, val in rewards.items():
        p = key.split("|")
        if len(p) != 4 or p[0] not in keep:
            continue
        v = float(val)
        if is_failure(v):
            continue
        tables.setdefault((p[0], int(p[1])), {})[(int(p[2]), int(p[3]))] = v

    cases, no_meta = [], 0
    for (bench, li), t in sorted(tables.items()):
        t = dict(t)
        t[NOOP] = 0.0
        action, gain = oracle_of_gated(t, deadzone)
        if action == NOOP:
            continue
        if (bench, li) not in meta:
            no_meta += 1
            continue
        fn, tri = meta[(bench, li)]
        cases.append({"benchmark": bench, "loop_idx": li,
                      "unmerge": action[0], "factor": action[1],
                      "filename": fn, "triple": tri, "gain": gain})
    if no_meta:
        log.warning("%d loops had no record (no filename/triple) — skipped",
                    no_meta)
    return cases


def _compile_guarded(work: Path, flags: str, arch: str, rec: Path):
    """
    _compile, but a build timeout is a result and not a crash.

    hecbench._make passes timeout=300 to subprocess.run, which RAISES
    TimeoutExpired; neither it nor _compile catches it. Uncaught, one slow
    benchmark out of 265 would abort the whole sweep after hours of work.
    """
    try:
        return _compile(work, flags, arch, rec)
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"


def run_case(bench: Path, c: dict, arch: str, workdir: Path) -> dict:
    """Two compiles, one diff."""
    work = workdir / ("%s_%d" % (bench.name, c["loop_idx"]))
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(bench, work)
    # Record dirs live OUTSIDE the work tree — see _compile. Both compiles run
    # in the SAME tree, which is safe because hecbench._make runs `make clean`
    # first: the second build genuinely recompiles rather than reusing objects
    # and emitting no records.
    base_rec = workdir / ("rec_%s_base" % work.name)
    uu_rec = workdir / ("rec_%s_uu" % work.name)

    ok_base, _ = _compile_guarded(work, "", arch, base_rec)
    flags = _build_extra_cflags(enable_uu=True, filename=c["filename"],
                                triple=c["triple"], loop_indices=[c["loop_idx"]],
                                unmerge_flags=[c["unmerge"]],
                                unroll_factors=[c["factor"]])
    ok_uu, _ = _compile_guarded(work, flags, arch, uu_rec)

    b, u = parse_remarks(base_rec), parse_remarks(uu_rec)
    lost = sorted(((("%s/%s" % (k[1], k[2])), b[k] - u.get(k, 0))
                   for k in b if k[0] == "Passed" and b[k] > u.get(k, 0)),
                  key=lambda kv: -kv[1])
    gained = sorted(((("%s/%s" % (k[1], k[2])), u[k] - b.get(k, 0))
                     for k in u if k[0] == "Passed" and u[k] > b.get(k, 0)),
                    key=lambda kv: -kv[1])
    # A Missed remark that is NEW under UU is the strongest mechanism evidence
    # there is: the pass ran, looked at the loop, and declined — LLVM usually
    # says why ("could not determine number of loop iterations"), which is
    # canonical-form damage stated in the compiler's own words.
    missed = sorted(((("%s/%s" % (k[1], k[2])), u[k] - b.get(k, 0))
                     for k in u if k[0] == "Missed" and u[k] > b.get(k, 0)),
                    key=lambda kv: -kv[1])

    # usable=0 means the DIFF IS MEANINGLESS, not that nothing was lost. If the
    # UU build failed or timed out, every baseline remark is absent from the UU
    # side and the row would otherwise read as a total wipe-out of downstream
    # optimisation. Same if the baseline emitted no records at all.
    usable = int(bool(ok_base and ok_uu and b))
    if not usable:
        log.warning("  %s loop=%d: UNUSABLE diff (baseline_ok=%s uu_ok=%s "
                    "baseline_remarks=%d) — excluded from the summary",
                    bench.name, c["loop_idx"], ok_base, ok_uu, sum(b.values()))

    shutil.rmtree(work, ignore_errors=True)
    shutil.rmtree(base_rec, ignore_errors=True)
    shutil.rmtree(uu_rec, ignore_errors=True)
    return {"baseline_ok": int(ok_base), "uu_ok": int(ok_uu), "usable": usable,
            "remarks_baseline": sum(b.values()), "remarks_uu": sum(u.values()),
            "n_opts_lost": sum(n for _, n in lost),
            "n_opts_gained": sum(n for _, n in gained),
            "n_missed_new": sum(n for _, n in missed),
            "lost_passes": ";".join("%s x%d" % (k, n) for k, n in lost[:8]),
            "missed_new": ";".join("%s x%d" % (k, n) for k, n in missed[:8]),
            "_lost": lost, "_missed": missed}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True,
                   help="holds reward_cache.json and eligible_benchmarks.json")
    p.add_argument("--hecbench-src", default=None)
    p.add_argument("--arch", default=ARCH)
    p.add_argument("--deadzone", type=float, default=0.005)
    p.add_argument("--limit", type=int, default=0, help="max loops (0 = all)")
    p.add_argument("--out-dir", type=Path, default=Path("uu_canonical"))
    args = p.parse_args()

    src = Path(args.hecbench_src) if args.hecbench_src else HECBENCH_SRC
    disc = {b.name: b for b in discover_benchmarks(src)}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="uu_canonical_"))

    cases = scope(args.run_dir, args.deadzone)
    if args.limit:
        cases = cases[:args.limit]
    log.info("%d loops whose oracle-best action is a transform -> %d compiles",
             len(cases), 2 * len(cases))

    rows = []
    for i, c in enumerate(cases, 1):
        bench = disc.get(c["benchmark"]) or (src / c["benchmark"])
        if not bench.is_dir():
            log.warning("skip %s — not found", c["benchmark"])
            continue
        log.info("[%d/%d] %s loop=%d  unmerge=%d factor=%d  gain=%+.4f",
                 i, len(cases), c["benchmark"], c["loop_idx"], c["unmerge"],
                 c["factor"], c["gain"])
        r = run_case(bench, c, args.arch, workdir)
        rows.append(dict(c, **r))
        log.info("     -> lost %d, gained %d, new-missed %d%s",
                 r["n_opts_lost"], r["n_opts_gained"], r["n_missed_new"],
                 "" if r["usable"] else "   [UNUSABLE]")

    cols = ["benchmark", "loop_idx", "unmerge", "factor", "gain",
            "n_opts_lost", "n_opts_gained", "n_missed_new", "remarks_baseline",
            "remarks_uu", "lost_passes", "missed_new", "usable", "baseline_ok",
            "uu_ok", "filename", "triple"]
    with open(args.out_dir / "canonical_form_cost.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore", restval="")
        w.writeheader()
        w.writerows(rows)
    (args.out_dir / "canonical_form_cost.json").write_text(json.dumps(rows, indent=2))

    if not rows:
        log.warning("no rows")
        return
    # Every statistic below is over USABLE rows only. A row whose UU build failed
    # has zero UU remarks, so its "lost" count is the entire baseline set — left
    # in, a handful of them would dominate the mean and manufacture the result.
    good = [r for r in rows if r["usable"]]
    log.info("")
    log.info("=" * 68)
    log.info("  %d loops attempted, %d usable (%d excluded: build failed or no "
             "baseline records)", len(rows), len(good), len(rows) - len(good))
    if not good:
        log.warning("  nothing usable — check the LLVM build and TARGET_ARCH")
        return
    lost_any = [r for r in good if r["n_opts_lost"] > 0]
    rest = [r for r in good if r["n_opts_lost"] == 0]
    log.info("  %d of %d (%.0f%%) lost at least one optimisation",
             len(lost_any), len(good), 100 * len(lost_any) / len(good))
    log.info("  optimisations lost per loop: mean %.1f  median %.0f  max %d",
             st.fmean([r["n_opts_lost"] for r in good]),
             st.median([r["n_opts_lost"] for r in good]),
             max(r["n_opts_lost"] for r in good))
    log.info("")
    log.info("  THE CROSSING — was it worth it?")
    log.info("    mean gain, loops that lost something : %+.4f  (n=%d)",
             st.fmean([r["gain"] for r in lost_any]) if lost_any else float("nan"),
             len(lost_any))
    log.info("    mean gain, loops that lost nothing   : %+.4f  (n=%d)",
             st.fmean([r["gain"] for r in rest]) if rest else float("nan"),
             len(rest))
    log.info("    every loop here gained more than the deadzone by construction "
             "— the oracle chose this action. The question is the magnitude.")

    for key, label in (("_lost", "passes that STOPPED firing under UU"),
                       ("_missed", "NEW 'missed' remarks under UU (the reason, "
                                   "in the compiler's words)")):
        tally = Counter()
        for r in good:
            for name, n in r[key]:
                tally[name] += n
        if tally:
            log.info("")
            log.info("  %s", label)
            for name, n in tally.most_common(15):
                log.info("    %6d  %s", n, name)
    log.info("")
    log.info("  CAVEAT for the write-up: clang runs the pipeline for BOTH the "
             "device (nvptx) and host sub-compilations and both emit records. "
             "The diff is symmetric so it stays valid, but a lost remark is not "
             "by itself proof a DEVICE optimisation was blocked — confirm "
             "against the per-function names before claiming one.")
    log.info("Written: %s", args.out_dir.resolve())
    shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
