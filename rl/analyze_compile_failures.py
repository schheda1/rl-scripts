"""
CPU-only analysis of UU compile failures and regressions.  NO GPU REQUIRED —
runs on any machine with this LLVM build.

Answers two questions the RL results cannot:

  1. Is a failure caused by *enabling* the UU pass at all, or by the specific
     (loop, unmerge, factor) transform?
     The UU function pass runs simplifyLoop / formLCSSARecursively over EVERY
     loop whenever it is enabled, so a build can break with no loop targeted.
     Control: compile with --enable-uu and a NON-MATCHING loop index.
       - control fails  -> ENABLEMENT (canonicalisation) issue, benchmark-wide
       - control passes -> TRANSFORM  issue, specific to that (loop, action)
     NOTE: bare --enable-uu is NOT a control — with an empty -uu-opt-loop-idx
     every loop is transformed.

  2. Which downstream optimisations did UU block?
     Compiles baseline and UU with -fsave-optimization-record and diffs the
     remark sets (Pass/Name/Function).  Applies to loops that compiled but got
     SLOWER as well — often the actual explanation for a regression.

Input: compile_failures_<ckpt>.csv from a training run (benchmark, loop_idx,
unmerge, factor), or --loop to analyse one case directly.

Usage:
  python3 analyze_compile_failures.py --failures checkpoints/run/compile_failures_best.csv \\
      --hecbench-src /path/to/HeCBench/src --out-dir analysis_uu
  python3 analyze_compile_failures.py --loop contract-cuda:14:1:4 --hecbench-src ...
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
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from hecbench import (
    ARCH, HECBENCH_SRC, _build_extra_cflags, _make, discover_benchmarks,
    extract_error_signature,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("uu-analysis")

# A loop index no real compilation will contain — used for the control compile.
CONTROL_IDX = 999999


def _compile(bench: Path, cflags: str, arch: str, opt_record: "Path | None" = None):
    """Compile with optional -fsave-optimization-record. Returns (ok, stderr)."""
    if opt_record is not None:
        cflags = (f"{cflags} -fsave-optimization-record "
                  f"-foptimization-record-file={opt_record}")
    r = _make(bench, extra_cflags=cflags, arch=arch)
    return r.returncode == 0, r.stderr


def parse_remarks(path: Path) -> Counter:
    """
    Parse an LLVM optimization-record YAML into a Counter of
    (kind, Pass, Name) — dependency-free line scan, no pyyaml needed.
    Records look like:  --- !Missed / Pass: loop-vectorize / Name: MissedDetails
    """
    out: Counter = Counter()
    if not path.exists():
        return out
    kind = p = n = None
    for line in path.read_text(errors="replace").splitlines():
        m = re.match(r"^---\s+!(\w+)", line)
        if m:
            if kind and p:
                out[(kind, p, n or "")] += 1
            kind, p, n = m.group(1), None, None
            continue
        m = re.match(r"^Pass:\s+(\S+)", line)
        if m:
            p = m.group(1).strip("'\"")
            continue
        m = re.match(r"^Name:\s+(\S+)", line)
        if m:
            n = m.group(1).strip("'\"")
    if kind and p:
        out[(kind, p, n or "")] += 1
    return out


def analyse_case(bench: Path, loop_idx: int, unmerge: int, factor: int,
                 arch: str, workdir: Path) -> dict:
    """Run the control + transform compiles for one case and diff remarks."""
    res = {"benchmark": bench.name, "loop_idx": loop_idx,
           "unmerge": unmerge, "factor": factor}
    work = workdir / f"{bench.name}_{loop_idx}_{unmerge}_{factor}"
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(bench, work)

    base_rec = work / "baseline.opt.yaml"
    uu_rec   = work / "uu.opt.yaml"

    # 1. baseline (plain -O3)
    ok_base, err_base = _compile(work, "", arch, base_rec)
    res["baseline_ok"] = ok_base
    if not ok_base:
        # Should not happen — benchmarks with failing baselines never enter the
        # pipeline — but record it rather than mis-attributing to UU.
        res["category"] = "baseline_broken"
        res["error"] = extract_error_signature(err_base)
        return res

    # 2. CONTROL: UU enabled, no loop targeted -> isolates canonicalisation
    ctrl_flags = _build_extra_cflags(enable_uu=True, loop_indices=[CONTROL_IDX],
                                     unmerge_flags=[0], unroll_factors=[1])
    ok_ctrl, err_ctrl = _compile(work, ctrl_flags, arch)
    res["control_ok"] = ok_ctrl

    # 3. the actual transform
    tf_flags = _build_extra_cflags(enable_uu=True, loop_indices=[loop_idx],
                                   unmerge_flags=[unmerge], unroll_factors=[factor])
    ok_tf, err_tf = _compile(work, tf_flags, arch, uu_rec)
    res["transform_ok"] = ok_tf

    if not ok_ctrl:
        res["category"] = "enablement"      # UU pass alone breaks the build
        res["error"] = extract_error_signature(err_ctrl)
    elif not ok_tf:
        res["category"] = "transform"       # this (loop, action) breaks it
        res["error"] = extract_error_signature(err_tf)
    else:
        res["category"] = "compiles"        # for slowdown cases: diff remarks
        res["error"] = ""

    # 4. remark diff (only meaningful when both sides produced a record)
    b, u = parse_remarks(base_rec), parse_remarks(uu_rec)
    if b or u:
        lost   = {k: b[k] - u.get(k, 0) for k in b if b[k] > u.get(k, 0)}
        gained = {k: u[k] - b.get(k, 0) for k in u if u[k] > b.get(k, 0)}
        res["remarks_baseline"] = sum(b.values())
        res["remarks_uu"] = sum(u.values())
        # Passed remarks lost under UU = optimisations that stopped firing
        res["opts_lost"] = sorted(
            (f"{k[1]}/{k[2]}", v) for k, v in lost.items() if k[0] == "Passed"
        )[:20]
        res["opts_gained"] = sorted(
            (f"{k[1]}/{k[2]}", v) for k, v in gained.items() if k[0] == "Passed"
        )[:20]
        res["missed_new"] = sorted(
            (f"{k[1]}/{k[2]}", v) for k, v in gained.items() if k[0] == "Missed"
        )[:20]
    shutil.rmtree(work, ignore_errors=True)
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--failures", help="compile_failures_<ckpt>.csv from a run")
    p.add_argument("--loop", help="single case as benchmark:loop_idx:unmerge:factor")
    p.add_argument("--hecbench-src", default=None)
    p.add_argument("--arch", default=ARCH)
    p.add_argument("--out-dir", default="uu_analysis")
    p.add_argument("--limit", type=int, default=0, help="max cases (0 = all)")
    args = p.parse_args()

    src = Path(args.hecbench_src) if args.hecbench_src else HECBENCH_SRC
    disc = {b.name: b for b in discover_benchmarks(src)}
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="uu_analysis_"))

    cases = []
    if args.loop:
        b, li, um, f = args.loop.split(":")
        cases = [{"benchmark": b, "loop_idx": int(li),
                  "unmerge": int(um), "factor": int(f)}]
    elif args.failures:
        with open(args.failures) as fh:
            cases = [{"benchmark": r["benchmark"], "loop_idx": int(r["loop_idx"]),
                      "unmerge": int(r["unmerge"]), "factor": int(r["factor"])}
                     for r in csv.DictReader(fh)
                     if not r["benchmark"].startswith("OVERALL")]
    else:
        p.error("pass --failures or --loop")

    # One control compile per benchmark is enough to classify enablement issues,
    # so process cases benchmark-by-benchmark and short-circuit once a benchmark
    # is known to be enablement-broken.
    cases.sort(key=lambda c: (c["benchmark"], c["loop_idx"]))
    if args.limit:
        cases = cases[:args.limit]
    log.info("%d cases to analyse (CPU only, no GPU)", len(cases))

    results, enablement_broken = [], {}
    for i, c in enumerate(cases, 1):
        bench = disc.get(c["benchmark"]) or (src / c["benchmark"])
        if not bench.is_dir():
            log.warning("skip %s — not found", c["benchmark"]); continue
        if enablement_broken.get(c["benchmark"]):
            results.append({**c, "category": "enablement",
                            "error": enablement_broken[c["benchmark"]],
                            "baseline_ok": True, "control_ok": False,
                            "transform_ok": False})
            continue
        log.info("[%d/%d] %s loop=%d unmerge=%d factor=%d",
                 i, len(cases), c["benchmark"], c["loop_idx"], c["unmerge"], c["factor"])
        r = analyse_case(bench, c["loop_idx"], c["unmerge"], c["factor"],
                         args.arch, workdir)
        if r.get("category") == "enablement":
            enablement_broken[c["benchmark"]] = r.get("error", "")
        results.append(r)
        log.info("     -> %s  %s", r.get("category"), (r.get("error") or "")[:90])

    (out_dir / "uu_failure_analysis.json").write_text(json.dumps(results, indent=2))
    with open(out_dir / "uu_failure_analysis.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["benchmark", "loop_idx", "unmerge",
                                          "factor", "category", "baseline_ok",
                                          "control_ok", "transform_ok", "error"],
                           extrasaction="ignore", restval="")
        w.writeheader(); w.writerows(results)

    cat = Counter(r.get("category") for r in results)
    log.info("")
    log.info("=== Categories ===")
    for k, v in cat.most_common():
        log.info("  %-16s %d", k, v)
    log.info("  enablement-broken benchmarks: %s",
             sorted(enablement_broken) or "none")
    sigs = Counter(r.get("error", "") for r in results if r.get("error"))
    if sigs:
        log.info("=== Top error signatures ===")
        for s, n in sigs.most_common(8):
            log.info("  %4d x %s", n, s[:110])
    lost = Counter()
    for r in results:
        for name, n in r.get("opts_lost", []):
            lost[name] += n
    if lost:
        log.info("=== Optimisations that stopped firing under UU ===")
        for name, n in lost.most_common(10):
            log.info("  %5d  %s", n, name)
    log.info("Written: %s", out_dir.resolve())
    shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
