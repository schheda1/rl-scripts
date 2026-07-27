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

Input: compile_failures_<ckpt>.csv from a training run, with columns
benchmark, loop_idx, unmerge, factor, filename, triple.  filename/triple are
REQUIRED for a faithful reproduction — the UU pass is scoped by
-uu-match-filename / -uu-match-targettriple, so without them the loop index
also matches same-numbered loops in other TUs and on the host sub-compilation.
CSVs written before those columns existed still run, with a warning.

Usage:
  python3 analyze_compile_failures.py --failures checkpoints/run/compile_failures_best.csv \\
      --hecbench-src /path/to/HeCBench/src --out-dir analysis_uu
  python3 analyze_compile_failures.py \\
      --loop contract-cuda:14:1:4:main.cu:nvptx64-nvidia-cuda --hecbench-src ...
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


def _compile(bench: Path, cflags: str, arch: str, record_dir: "Path | None" = None):
    """
    Compile, optionally harvesting optimization remarks.  Returns (ok, stderr).

    `-foptimization-record-file=<path>` is deliberately NOT used: a benchmark
    compiles several translation units, and every one of them would write to
    that single path, leaving only the last.  Plain -fsave-optimization-record
    emits one <tu>.opt.yaml per TU instead; they are moved out to `record_dir`
    immediately, because the next compile in this same work tree would
    otherwise overwrite them (same TU names).  record_dir must live OUTSIDE
    `bench` or the rglob below would re-collect already-harvested files.
    """
    if record_dir is not None:
        cflags = f"{cflags} -fsave-optimization-record"
    r = _make(bench, extra_cflags=cflags, arch=arch)
    if record_dir is not None:
        record_dir.mkdir(parents=True, exist_ok=True)
        for y in bench.rglob("*.opt.yaml"):
            try:
                y.replace(record_dir / f"{y.parent.name}__{y.name}")
            except OSError:
                shutil.copy2(y, record_dir / f"{y.parent.name}__{y.name}")
    return r.returncode == 0, r.stderr


def parse_remarks(path: Path) -> Counter:
    """
    Parse LLVM optimization-record YAML into a Counter of (kind, Pass, Name) —
    dependency-free line scan, no pyyaml needed.  `path` may be a single file or
    a directory, in which case every *.opt.yaml under it is merged.
    Records look like:  --- !Missed / Pass: loop-vectorize / Name: MissedDetails

    CAVEAT for CUDA: clang runs the pipeline for BOTH the device (nvptx) and
    host sub-compilations, and both write records.  The diff stays valid because
    it is symmetric (same TUs on both sides), but a lost remark is not by itself
    proof that a *device* optimisation was blocked — confirm against the
    per-function names before drawing a conclusion in the write-up.
    """
    out: Counter = Counter()
    if path.is_dir():
        files = sorted(path.glob("*.opt.yaml"))
    elif path.exists():
        files = [path]
    else:
        return out
    for f in files:
        kind = p = n = None
        for line in f.read_text(errors="replace").splitlines():
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
                 arch: str, workdir: Path,
                 filename: str = "", triple: str = "") -> dict:
    """
    Run the control + transform compiles for one case and diff remarks.

    `filename` / `triple` must match what training used.  The UU pass is scoped
    by -uu-match-filename and -uu-match-targettriple; drop them and the loop
    index matches a same-numbered loop in every other TU, and on the host
    sub-compilation too — a different transform from the one that failed.
    """
    res = {"benchmark": bench.name, "loop_idx": loop_idx,
           "unmerge": unmerge, "factor": factor,
           "filename": filename, "triple": triple}
    if not filename or not triple:
        log.warning("  %s loop=%d: missing filename/triple — this case is NOT a "
                    "faithful reproduction of the training compile",
                    bench.name, loop_idx)
    work = workdir / f"{bench.name}_{loop_idx}_{unmerge}_{factor}"
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(bench, work)

    # Record dirs live OUTSIDE the work tree — see _compile.
    base_rec = workdir / f"rec_{work.name}_baseline"
    uu_rec   = workdir / f"rec_{work.name}_uu"

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
    # Identical to the transform compile in every respect EXCEPT the loop index,
    # so the only difference is whether a loop is actually transformed.
    ctrl_flags = _build_extra_cflags(enable_uu=True, filename=filename,
                                     triple=triple, loop_indices=[CONTROL_IDX],
                                     unmerge_flags=[0], unroll_factors=[1])
    ok_ctrl, err_ctrl = _compile(work, ctrl_flags, arch)
    res["control_ok"] = ok_ctrl

    # 3. the actual transform — mirrors hecbench.compile_single_loop_ex exactly
    tf_flags = _build_extra_cflags(enable_uu=True, filename=filename,
                                   triple=triple, loop_indices=[loop_idx],
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
    res["remark_files"] = [len(list(base_rec.glob("*.opt.yaml"))) if base_rec.is_dir() else 0,
                           len(list(uu_rec.glob("*.opt.yaml"))) if uu_rec.is_dir() else 0]
    if ok_base and not b:
        log.warning("  %s: baseline compiled but produced NO optimization "
                    "records — the remark diff for this case is empty",
                    bench.name)
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
    shutil.rmtree(base_rec, ignore_errors=True)
    shutil.rmtree(uu_rec, ignore_errors=True)
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
        parts = args.loop.split(":")
        if len(parts) not in (4, 6):
            p.error("--loop must be benchmark:loop_idx:unmerge:factor"
                    "[:filename:triple]")
        b, li, um, f = parts[:4]
        cases = [{"benchmark": b, "loop_idx": int(li),
                  "unmerge": int(um), "factor": int(f),
                  "filename": parts[4] if len(parts) == 6 else "",
                  "triple":   parts[5] if len(parts) == 6 else ""}]
    elif args.failures:
        with open(args.failures) as fh:
            rows = list(csv.DictReader(fh))
        if rows and "filename" not in rows[0]:
            log.warning("%s has no filename/triple columns — it predates the "
                        "schema change. Results will not faithfully reproduce "
                        "the training compiles.", args.failures)
        cases = [{"benchmark": r["benchmark"], "loop_idx": int(r["loop_idx"]),
                  "unmerge": int(r["unmerge"]), "factor": int(r["factor"]),
                  "filename": r.get("filename", ""),
                  "triple":   r.get("triple", "")}
                 for r in rows if not r["benchmark"].startswith("OVERALL")]
    else:
        p.error("pass --failures or --loop")

    # One control compile per benchmark is enough to classify enablement issues,
    # so process cases benchmark-by-benchmark and short-circuit once a benchmark
    # is known to be enablement-broken.
    cases.sort(key=lambda c: (c["benchmark"], c.get("filename", ""), c["loop_idx"]))
    if args.limit:
        cases = cases[:args.limit]
    log.info("%d cases to analyse (CPU only, no GPU)", len(cases))

    results, enablement_broken = [], {}
    for i, c in enumerate(cases, 1):
        bench = disc.get(c["benchmark"]) or (src / c["benchmark"])
        if not bench.is_dir():
            log.warning("skip %s — not found", c["benchmark"]); continue
        # Memoise per (benchmark, file, triple): the control is scoped by
        # -uu-match-filename, so a verdict for one TU says nothing about another.
        _ek = (c["benchmark"], c.get("filename", ""), c.get("triple", ""))
        if enablement_broken.get(_ek):
            results.append({**c, "category": "enablement",
                            "error": enablement_broken[_ek],
                            "baseline_ok": True, "control_ok": False,
                            "transform_ok": False})
            continue
        log.info("[%d/%d] %s loop=%d unmerge=%d factor=%d",
                 i, len(cases), c["benchmark"], c["loop_idx"], c["unmerge"], c["factor"])
        r = analyse_case(bench, c["loop_idx"], c["unmerge"], c["factor"],
                         args.arch, workdir,
                         filename=c.get("filename", ""),
                         triple=c.get("triple", ""))
        if r.get("category") == "enablement":
            enablement_broken[_ek] = r.get("error", "")
        results.append(r)
        log.info("     -> %s  %s", r.get("category"), (r.get("error") or "")[:90])

    (out_dir / "uu_failure_analysis.json").write_text(json.dumps(results, indent=2))
    with open(out_dir / "uu_failure_analysis.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["benchmark", "loop_idx", "unmerge",
                                          "factor", "filename", "triple",
                                          "category", "baseline_ok",
                                          "control_ok", "transform_ok", "error"],
                           extrasaction="ignore", restval="")
        w.writeheader(); w.writerows(results)

    cat = Counter(r.get("category") for r in results)
    log.info("")
    log.info("=== Categories ===")
    for k, v in cat.most_common():
        log.info("  %-16s %d", k, v)
    log.info("  enablement-broken (benchmark, file): %s",
             sorted({(k[0], k[1]) for k in enablement_broken}) or "none")
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
