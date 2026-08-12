"""
Which downstream optimisations does UU switch off, and what did the loop gain?

One question, one table. For every eligible loop whose ORACLE-BEST action is a
transform, compile it twice and diff the optimisation remarks:

    1. baseline   plain -O3                       -> what fires normally
    2. best       -O3 + UU at the oracle action   -> what still fires

Population (531 distinct eligible loops):
      265  oracle-best is a transform   <- IN SCOPE, 530 compiles
      174  oracle-best is no-op         <- nothing to compare against
       92  every measured cell failed   <- no usable table at all

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
import os
import re
import shlex
import collections
import shutil
import subprocess
import statistics as st
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hecbench import (ARCH, HECBENCH_SRC, _build_extra_cflags,   # noqa: E402
                      discover_benchmarks, extract_error_signature)
from offline_data import NOOP, oracle_of_gated                   # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("uu-canonical")

FAILURE_VALUES = (-0.16, -0.161)

# WHY THIS COMPILES THE DEVICE SIDE DIRECTLY INSTEAD OF RUNNING make.
#
# `clang++ -###` on a normal CUDA build shows both cc1 jobs -- nvptx64 and
# x86_64 -- being handed the SAME -opt-record-file, derived from -o. The device
# job runs first and writes it; the host job then overwrites it. Harvesting
# *.opt.yaml from the build tree therefore yields HOST remarks only, and since
# UU is scoped to the device triple the diff is empty for every loop. That is
# not a property of UU; it is the driver clobbering one record with the other.
#
# --cuda-device-only issues exactly ONE cc1 job, so -foptimization-record-file
# is unambiguous. Measured on adamw-cuda loop 0: 405 device records (167 in the
# loop's own kernel, none in `main`), and with UU applied the PTX grows 647 ->
# 853 lines and the records 405 -> 421 -- so the transform fires and the loop
# numbering survives.
CUDA_HOME = os.environ.get("CUDA_HOME", "/usr/local/cuda")
COMPILE_TIMEOUT = 600


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
                meta[(bench, int(rec["loop_idx"]))] = (
                    rec.get("filename", ""), rec.get("triple", ""),
                    rec.get("kernel_parents") or [])

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
        fn, tri, kp = meta[(bench, li)]
        cases.append({"benchmark": bench, "loop_idx": li,
                      "unmerge": action[0], "factor": action[1],
                      "filename": fn, "triple": tri, "kernel_parents": kp,
                      "gain": gain})
    if no_meta:
        log.warning("%d loops had no record (no filename/triple) — skipped",
                    no_meta)
    return cases


def parse_remarks(path: Path) -> tuple:
    """
    LLVM optimisation-record YAML -> Counter of (kind, Pass, Name, Function).

    A local parser rather than analyze_compile_failures.parse_remarks, which
    drops the Function field. Without it the diff cannot be scoped to the loop's
    own kernel, and a module-wide count mixes in every cub/BlockReduce template
    the TU instantiates.

    Same dependency-free line scan; records look like
        --- !Passed
        Pass:            loop-unroll
        Name:            FullyUnrolled
        Function:        _Z17fused_4bit_kernelIfLi64EE...

    Returns (counts, reasons):
      counts  Counter[(kind, Pass, Name, Function)]
      reasons Counter[(Pass, Function, reason)] for Missed records only — the
              Args block, which is where LLVM states WHY it declined ("loop not
              vectorized: could not determine number of loop iterations"). That
              sentence is the mechanism evidence; the pass name alone is not.
    """
    counts, reasons = Counter(), Counter()
    if not path.exists():
        return counts, reasons
    kind = p = n = fn = None
    args, in_args = [], False

    def flush():
        if kind and p:
            counts[(kind, p, n or "", fn or "")] += 1
            if kind == "Missed" and args:
                reasons[(p, fn or "", " ".join(args)[:160])] += 1

    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("--- !"):
            flush()
            kind, p, n, fn = line[5:].strip(), None, None, None
            args, in_args = [], False
        elif line.startswith("Pass:"):
            p = line.split(":", 1)[1].strip().strip("'\"")
        elif line.startswith("Name:"):
            n = line.split(":", 1)[1].strip().strip("'\"")
        elif line.startswith("Function:"):
            fn = line.split(":", 1)[1].strip().strip("'\"")
        elif line.startswith("Args:"):
            in_args = True
        elif in_args:
            m = re.search(r"String:\s*'?([^'\n]*)'?\s*$", line)
            if m and m.group(1).strip():
                args.append(m.group(1).strip())
    flush()
    return counts, reasons


def device_compile(bench: Path, filename: str, arch: str, uu_flags: str,
                   yaml_out: Path, ptx_out: Path):
    """
    Compile ONE translation unit for the device only. Returns (ok, stderr).

    Run from INSIDE the benchmark directory with the source named RELATIVELY,
    exactly as hecbench._make does (cwd=benchmark_dir, the Makefile compiling
    `main.cu`). This is not cosmetic: the UU pass is scoped by
    -uu-match-filename=main.cu, which is compared against the source name the
    module records. Hand clang an ABSOLUTE path and that name becomes the full
    path, the match fails, and the pass silently does nothing — the compile
    succeeds, the PTX is byte-identical to the baseline, and every diff is zero.
    Measured: identical flags fired from the benchmark dir and did not fire from
    outside it.

    Flags otherwise mirror _make (-O3 -std=c++17 -Wall, the CUDA include dir,
    --cuda-gpu-arch) so the code measured here is the code that was measured
    then, plus --cuda-device-only so there is a single cc1 job and the record
    path cannot be clobbered by the host half.
    """
    inc = ["-I.", "-I" + CUDA_HOME + "/include"]
    # A source under a subdirectory ('./kernel/kernel.cu') needs its own
    # directory on the include path; the Makefile supplies that when it builds.
    d = os.path.dirname(filename)
    if d:
        inc.insert(1, "-I" + d)
    cmd = ["clang++", "-x", "cuda", "--cuda-device-only", "-S",
           "-O3", "-std=c++17", "-Wall",
           "--cuda-gpu-arch=" + arch] + inc + [
           "-fsave-optimization-record",
           "-foptimization-record-file=" + str(yaml_out.resolve())]
    cmd += shlex.split(uu_flags)
    cmd += [filename, "-o", str(ptx_out.resolve())]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=COMPILE_TIMEOUT, cwd=str(bench))
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    return r.returncode == 0, r.stderr


def _delta(b: Counter, u: Counter, kind: str, funcs=None):
    """Counts present in `b` but reduced in `u` (or the reverse for gains)."""
    def keep(k):
        return k[0] == kind and (funcs is None or k[3] in funcs)
    return sorted(((("%s/%s" % (k[1], k[2])), b[k] - u.get(k, 0))
                   for k in b if keep(k) and b[k] > u.get(k, 0)),
                  key=lambda kv: -kv[1])


def run_case(bench: Path, c: dict, arch: str, workdir: Path,
             keep: "Path | None" = None) -> dict:
    """Two device-only compiles of one TU, one diff. No make, no tree copy."""
    rel = c["filename"] or "main.cu"
    src = bench / rel
    tag = "%s_%d" % (bench.name, c["loop_idx"])
    base_y, uu_y = workdir / (tag + "_base.yaml"), workdir / (tag + "_uu.yaml")
    base_p, uu_p = workdir / (tag + "_base.ptx"), workdir / (tag + "_uu.ptx")

    if not src.exists():
        # Same key set as the success path. A partial dict would not break the
        # CSV (restval fills it) but would KeyError the moment this row was ever
        # counted as usable, and "never usable" is an invariant one edit away
        # from being false.
        return {"baseline_ok": 0, "uu_ok": 0, "usable": 0, "uu_fired": 0,
                "why_baseline": "missing source %s" % rel, "why_uu": "",
                "remarks_baseline": 0, "remarks_uu": 0, "n_opts_lost": 0,
                "n_opts_gained": 0, "n_missed_new": 0, "n_lost_in_kernel": 0,
                "n_missed_in_kernel": 0, "n_flipped": 0, "lost_passes": "",
                "ptx_lines_baseline": 0, "ptx_lines_uu": 0,
                "missed_new": "", "flipped_passes": "", "top_reason": "",
                "_lost": [], "_missed": [], "_lost_kernel": [],
                "_missed_kernel": [], "_flipped": [], "_why": []}

    ok_base, err_base = device_compile(bench, rel, arch, "", base_y, base_p)
    flags = _build_extra_cflags(enable_uu=True, filename=c["filename"],
                                triple=c["triple"], loop_indices=[c["loop_idx"]],
                                unmerge_flags=[c["unmerge"]],
                                unroll_factors=[c["factor"]])
    ok_uu, err_uu = device_compile(bench, rel, arch, flags, uu_y, uu_p)

    b, br = parse_remarks(base_y)
    u, ur = parse_remarks(uu_y)
    # Identical PTX means the transform never applied to this loop, so a zero
    # diff says nothing about UU. This is the check whose absence let a whole
    # host-pipeline run be reported as a result.
    fired = 0
    ptx_b = ptx_u = 0
    if ok_base and ok_uu and base_p.exists() and uu_p.exists():
        pb, pu = base_p.read_bytes(), uu_p.read_bytes()
        fired = int(pb != pu)
        # Code size, because it is the denominator the Missed counts need.
        # Unrolling by 8 replicates the body, so the SLP vectoriser simply gets
        # more candidates to decline and emits more NotPossible/NotBeneficial
        # remarks. A raw rise in Missed is partly an artifact of there being
        # more code, NOT evidence that anything was blocked. The Passed->absent
        # direction has no such confound.
        ptx_b, ptx_u = pb.count(b"\n"), pu.count(b"\n")

    funcs = set(c.get("kernel_parents") or [])
    lost = _delta(b, u, "Passed")
    gained = _delta(u, b, "Passed")
    missed = _delta(u, b, "Missed")
    lost_k = _delta(b, u, "Passed", funcs) if funcs else []
    missed_k = _delta(u, b, "Missed", funcs) if funcs else []

    # THE DIRECT ANSWER: a pass that PASSED in the baseline and is MISSED under
    # UU. Not "the count went down" — the pass ran, looked at the same code, and
    # explicitly declined. Paired by Pass name, so `loop-vectorize` passing then
    # missing is one entry regardless of the Name field either side used.
    ran = {k[1] for k in b if k[0] == "Passed" and k[3] in funcs}
    now_missed = collections.defaultdict(int)
    for k, v in u.items():
        if k[0] == "Missed" and k[1] in ran and k[3] in funcs:
            now_missed[k[1]] += v - b.get(k, 0)
    flipped = sorted(((p, n) for p, n in now_missed.items() if n > 0),
                     key=lambda kv: -kv[1])
    # and why, in LLVM's words — new Missed reasons inside the kernel
    why = sorted(((("%s: %s" % (k[0], k[2])), v - br.get(k, 0))
                  for k, v in ur.items()
                  if k[1] in funcs and v > br.get(k, 0)),
                 key=lambda kv: -kv[1])

    usable = int(bool(ok_base and ok_uu and b and fired))
    if not usable:
        log.warning("  %s loop=%d: UNUSABLE (baseline_ok=%s uu_ok=%s "
                    "baseline_records=%d uu_fired=%d)", bench.name,
                    c["loop_idx"], ok_base, ok_uu, sum(b.values()), fired)

    if keep is not None:
        keep.mkdir(parents=True, exist_ok=True)
        for f in (base_y, uu_y, base_p, uu_p):
            if f.exists():
                shutil.copy2(f, keep / f.name)
        log.info("     kept records in %s", keep)
    for f in (base_y, uu_y, base_p, uu_p):
        try:
            f.unlink()
        except OSError:
            pass

    def _why(ok, err):
        if ok:
            return ""
        if err == "TIMEOUT":
            return "timeout"
        return extract_error_signature(err)[:120] if err else "unknown"

    return {"baseline_ok": int(ok_base), "uu_ok": int(ok_uu), "usable": usable,
            "uu_fired": fired,
            "why_baseline": _why(ok_base, err_base),
            "why_uu": _why(ok_uu, err_uu),
            "remarks_baseline": sum(b.values()), "remarks_uu": sum(u.values()),
            "ptx_lines_baseline": ptx_b, "ptx_lines_uu": ptx_u,
            "n_opts_lost": sum(n for _, n in lost),
            "n_opts_gained": sum(n for _, n in gained),
            "n_missed_new": sum(n for _, n in missed),
            "n_lost_in_kernel": sum(n for _, n in lost_k),
            "n_missed_in_kernel": sum(n for _, n in missed_k),
            # Kernel-scoped, matching n_lost_in_kernel / n_missed_in_kernel.
            # These were module-wide while the counts beside them were not.
            "lost_passes": ";".join("%s x%d" % (k, n) for k, n in lost_k[:8]),
            "missed_new": ";".join("%s x%d" % (k, n) for k, n in missed_k[:8]),
            "n_flipped": sum(n for _, n in flipped),
            "flipped_passes": ";".join("%s x%d" % (p, n) for p, n in flipped),
            "top_reason": why[0][0] if why else "",
            "_lost": lost, "_missed": missed,
            "_lost_kernel": lost_k, "_missed_kernel": missed_k,
            "_flipped": flipped, "_why": why}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True,
                   help="holds reward_cache.json and eligible_benchmarks.json")
    p.add_argument("--hecbench-src", default=None)
    p.add_argument("--arch", default=ARCH)
    p.add_argument("--deadzone", type=float, default=0.005)
    p.add_argument("--limit", type=int, default=0, help="max loops (0 = all)")
    p.add_argument("--keep-records", type=Path, default=None,
                   help="copy every .opt.yaml here instead of deleting it, so an "
                        "all-zero diff can be diagnosed (use with --limit 1)")
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
        r = run_case(bench, c, args.arch, workdir, args.keep_records)
        rows.append(dict(c, **r))
        log.info("     -> module lost %d / missed %d | in-kernel lost %d / "
                 "missed %d%s", r["n_opts_lost"], r["n_missed_new"],
                 r["n_lost_in_kernel"], r["n_missed_in_kernel"],
                 "" if r["usable"] else "   [UNUSABLE]")

    cols = ["benchmark", "loop_idx", "unmerge", "factor", "gain",
            "n_lost_in_kernel", "n_missed_in_kernel",
            "n_opts_lost", "n_opts_gained", "n_missed_new",
            "remarks_baseline", "remarks_uu", "ptx_lines_baseline",
            "ptx_lines_uu", "lost_passes", "missed_new",
            "n_flipped", "flipped_passes", "top_reason",
            "usable", "uu_fired", "baseline_ok", "uu_ok",
            "why_baseline", "why_uu", "filename", "triple"]
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
    n_nofire = sum(1 for r in rows if r["baseline_ok"] and r["uu_ok"]
                   and not r["uu_fired"])
    if n_nofire:
        log.warning("")
        log.warning("  *** %d rows compiled both ways but produced IDENTICAL PTX "
                    "— UU did not fire at that loop index, so their zero diff "
                    "says nothing. Excluded. ***", n_nofire)
    # ONE list to read: passes that fired in the baseline and stopped, inside
    # the loop's own kernel. Everything else here is context for it.
    lk = [r for r in good if r["n_lost_in_kernel"] > 0]
    log.info("  %d of %d loops (%.0f%%) lost an optimisation inside their own "
             "kernel", len(lk), len(good), 100 * len(lk) / len(good))
    log.info("")
    log.info("  PASSES THAT STOPPED FIRING IN THE KERNEL")
    tally = Counter()
    for r in good:
        for name, n in r["_lost_kernel"]:
            tally[name] += n
    for name, n in tally.most_common(15):
        log.info("    %6d  %s", n, name)
    if not tally:
        log.info("    (none)")

    log.info("")
    log.info("  and was it worth it")
    rest = [r for r in good if r["n_lost_in_kernel"] == 0]
    log.info("    mean gain, loops that lost one  : %+.4f  (n=%d)",
             st.fmean([r["gain"] for r in lk]) if lk else float("nan"), len(lk))
    log.info("    mean gain, loops that lost none : %+.4f  (n=%d)",
             st.fmean([r["gain"] for r in rest]) if rest else float("nan"),
             len(rest))

    # Context only. Unrolling replicates the body, so passes get more candidates
    # to decline and Missed counts rise with code size, not with damage.
    grow = 100 * st.fmean([(r["ptx_lines_uu"] - r["ptx_lines_baseline"])
                           / max(r["ptx_lines_baseline"], 1) for r in good])
    mt = Counter()
    for r in good:
        for name, n in r["_missed_kernel"]:
            mt[name] += n
    if mt:
        log.info("")
        log.info("  context, NOT damage — PTX is %+.0f%% larger, so passes have "
                 "more to examine and decline:", grow)
        log.info("    %s", ", ".join("%s %d" % (k, v) for k, v in mt.most_common(4)))
    nf = sum(r["n_flipped"] for r in good)
    if nf:
        log.info("  %d of those are strict Passed->Missed flips on the same pass.",
                 nf)
    log.info("")
    log.info("  CSV: read n_lost_in_kernel and lost_passes. Both are scoped to "
             "the loop's kernel; n_opts_lost is the whole TU and includes cub "
             "and other templates the loop may never reach.")
    log.info("Written: %s", args.out_dir.resolve())
    shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
