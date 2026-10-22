"""
Locked-clock kernel timer: Nsight Compute at the GPU's base clock (`--profiler ncu`;
design in collection_dev_2026-10-21/README §12, measurements in CLOCK_LOCK.md).
Stdlib only.  The baseline step (train._measure_one_baseline) and the pipeline's GPU
stage (collect_pipeline.exec_worker) both measure through `measure`, so the two sides
of a reward come from the same code.

A RUN is one execution of the application under ncu.  Inside it, launches skip+1 ..
skip+count of ONE kernel, selected by its exact mangled name, are profiled, each as a
separate serialized measurement with the SM clock locked (`--clock-control base`); with
`--kill` the application ends after the last of them.  A loop with several parent
kernels gets one run per parent: `--launch-count` is global across a regex, so one
kernel could use up the count of another.

Per run: the mean `gpu__time_duration` over the profiled launches, their spread (CV),
and per launch the clock (`sm__cycles_elapsed.avg.per_second`), SM cycles and executed
instructions (recorded, a wrong-work signal; not a reject criterion).  The clock is
checked for what the ratio needs, not against a fixed value (the base clock differs
between A100 boards): every launch of a run within CLOCK_TOL of the run's median (the
lock held), and a cell's run within CLOCK_TOL of the clock its baseline recorded for
that parent (both sides of the ratio at one clock); a baseline's runs, of its first.

The four metrics are collected in one pass (no replay, no save/restore, caches in their
natural state with `--cache-control none`).  Keep the set fixed and never add
`--section`/`--set`: another metric can turn it into a multi-pass replay, which changes
both the time and the cache state.  ncu cannot count launches under `--kill`; launch
counts come from the one nsys census run per benchmark in the baseline step.
"""

import csv
import os
import re
import shlex
import statistics
import subprocess
import time

from store_common import run_hard_timeout

METRICS = ("gpu__time_duration.sum", "sm__cycles_elapsed.avg",
           "sm__cycles_elapsed.avg.per_second", "smsp__inst_executed.sum")
SKIP, COUNT = 2, 20        # launches skipped (cold start) / profiled per run; same on both sides
CLOCK_TOL = 0.01           # clock agreement: launches of a run, a cell and its baseline
CV_MAX = 0.01              # a run whose per-launch CV is above this gets one more run
RUNS_DIFF_MAX = 0.005      # baselines: two runs further apart than this get one more run

# Unit factors to ms / MHz.  `--csv` implies base units (nsecond, cycle/second); the
# scaled forms are accepted in case a version prints them.  An unknown unit is read
# as the base unit: a wrong clock factor then fails the clock check loudly, and a
# common time factor cancels in the reward.
_TO_MS = {"nsecond": 1e-6, "ns": 1e-6, "usecond": 1e-3, "us": 1e-3, "msecond": 1.0,
          "ms": 1.0, "second": 1e3, "s": 1e3}
_TO_MHZ = {"cycle/second": 1e-6, "hz": 1e-6, "cycle/usecond": 1.0, "mhz": 1.0,
           "cycle/msecond": 1e-3, "khz": 1e-3, "cycle/nsecond": 1e3, "ghz": 1e3}


def timer_stamp(profiler: str, count: int = COUNT) -> dict:
    """What a baseline and a reward were measured with; two caches score against each
    other only if this is equal (an nsys baseline must never score an ncu cell).  The
    clock is not in it: each baseline parent records its own, and cells must match it."""
    if profiler != "ncu":
        return {"profiler": profiler}
    return {"profiler": "ncu", "skip": SKIP, "count": count, "metrics": list(METRICS)}


def launch_window(n_launches, count: int = COUNT) -> "tuple[int, int]":
    """(skip, count) for a kernel launched *n_launches* times per run (None: unknown).
    A kernel launched fewer than SKIP + 1 times keeps its last launch; count 0 = all."""
    if not n_launches:
        return SKIP, count
    skip = min(SKIP, n_launches - 1)
    return skip, (min(count, n_launches - skip) if count else 0)


def command(run_cmd: str, kernel: str, log_file: str, skip: int, count: int) -> str:
    """One ncu run profiling launches skip+1..skip+count of *kernel* (exact mangled
    name); count 0 profiles every launch and lets the application finish."""
    limit = (f"--launch-skip {skip} --launch-count {count} --kill yes" if count
             else f"--launch-skip {skip}")
    return ("ncu --clock-control base --cache-control none "
            "--target-processes application-only "
            "--kernel-name-base mangled --print-kernel-base mangled "
            f"-k {shlex.quote('regex:^' + re.escape(kernel) + '$')} {limit} "
            f"--check-exit-code no --metrics {','.join(METRICS)} "
            f"--csv --page raw --print-units base --log-file {shlex.quote(log_file)} "
            f"{run_cmd}")


def _num(s: str):
    try:
        return float(s.replace(",", ""))
    except (AttributeError, ValueError):
        return None


def parse_log(text: str) -> "tuple[list, list]":
    """`--csv --page raw` log -> ([{"kernel", "ms", "mhz", "cycles", "inst"} per profiled
    launch], [==ERROR== lines]).  The header is the first row naming all METRICS; a row
    right after it whose time is not a number holds the units; every later row with a
    numeric time and clock is a launch.  Tool lines (==PROF==, ==WARNING==) are skipped."""
    errors = [l.strip() for l in text.splitlines() if l.startswith("==ERROR==")]
    rows = [l for l in text.splitlines() if l.strip() and not l.startswith("==")]
    start = next((i for i, l in enumerate(rows) if all(m in l for m in METRICS)), None)
    if start is None:      # nothing profiled, or a layout this parser does not know
        return [], errors + (["ncu log: no metrics header; first line: " + rows[0][:200]]
                             if rows else [])
    table = list(csv.reader(rows[start:]))
    head = [h.strip() for h in table[0]]
    if not all(m in head for m in METRICS):
        return [], errors + ["ncu log: no column per metric in " + rows[start][:200]]
    col = {m: head.index(m) for m in METRICS}
    kcol = head.index("Kernel Name") if "Kernel Name" in head else None
    # ncu 2026.2.1's raw page also has every other raw metric; this one says whether
    # the four were collected in one pass (no replay), as they should be.
    pcol = (head.index("profiler__replayer_passes") if "profiler__replayer_passes" in head
            else None)
    units = (table[1] if len(table) > 1 and len(table[1]) == len(head)
             and _num(table[1][col[METRICS[0]]]) is None else None)
    unit = (lambda m: units[col[m]].strip().lower()) if units else (lambda m: "")
    t_f = _TO_MS.get(unit(METRICS[0]), 1e-6)
    f_f = _TO_MHZ.get(unit(METRICS[2]), 1e-6)
    out = []
    for r in table[1:]:
        if len(r) != len(head):
            continue
        t, hz = _num(r[col[METRICS[0]]]), _num(r[col[METRICS[2]]])
        if t is None or hz is None:
            continue
        out.append({"kernel": r[kcol] if kcol is not None else "", "ms": t * t_f,
                    "mhz": hz * f_f, "cycles": _num(r[col[METRICS[1]]]),
                    "inst": _num(r[col[METRICS[3]]]),
                    "passes": _num(r[pcol]) if pcol is not None else None})
    return out, errors


def _median(vals):
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def gpu_temp(env: dict):
    """Temperature (C) of the run's GPU (CUDA_VISIBLE_DEVICES, nvidia-smi numbering), or
    None.  At a locked SM clock a memory-bound kernel still slows with temperature
    (debayer +0.36 % from 27 to 44 C, GPU node 2026-10-22), so it is recorded per run:
    a baseline and its cells timed at different temperatures can be told apart."""
    try:
        r = subprocess.run(["nvidia-smi", "-i", env.get("CUDA_VISIBLE_DEVICES", ""),
                            "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        return float(r.stdout.split()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def summarize(launches: list) -> dict:
    """One run's launches -> {"status", "ms", "cv", "n", "mhz", "cycles", "inst"}.
    status: ok / no_output (nothing profiled) / clock_unlocked (the launches' clocks
    differ by more than CLOCK_TOL: the lock did not hold for the whole run)."""
    if not launches:
        return {"status": "no_output"}
    ms = [l["ms"] for l in launches]
    mhz = [l["mhz"] for l in launches]
    mean = statistics.fmean(ms)
    off = max(mhz) - min(mhz) > CLOCK_TOL * statistics.median(mhz)
    return {"status": "clock_unlocked" if off else "ok", "ms": mean,
            "cv": statistics.pstdev(ms) / mean if mean > 0 else 0.0, "n": len(ms),
            "mhz": statistics.median(mhz), "mhz_range": [min(mhz), max(mhz)],
            "cycles": _median(l["cycles"] for l in launches),
            "inst": _median(l["inst"] for l in launches),
            "passes": max((l["passes"] for l in launches if l.get("passes") is not None),
                          default=None)}


def measure(run_cmd: str, cwd, env: dict, timeout: int, kernel: str, skip: int,
            count: int, stem: str, n_runs: int, n_base: int = 0, baseline: bool = False,
            ref_mhz: "float | None" = None) -> dict:
    """*n_runs* runs (a baseline at least 2) of one kernel -> {"status", "ms" (median of
    the per-run means), "runs", "cv" (largest within-run), "n" (fewest launches),
    "mhz", "cycles", "inst", "passes", "temp" (C before each good run), "exec_s",
    "error"}.

    status ok / no_output / clock_unlocked / short (fewer launches than *n_base*, the
    baseline's: the application died before reaching them).  Two failed runs in a row
    end the measurement (with the good runs so far, if any); a run off the clock (its
    launches disagree, or its clock is not *ref_mhz*, the baseline's, or for a baseline
    its first run's) is repeated once, a second one ends the measurement.  One more run
    when a run's CV > CV_MAX or, for a baseline, two runs differ by more than
    RUNS_DIFF_MAX.  Each log is deleted once read; ncu's own ==ERROR== lines (it prints
    some to the console, not to the log) are kept in "error"."""
    want = max(n_runs, 2) if baseline else max(n_runs, 1)
    good, exec_s, errors = [], [], []
    status, clock_retried, extended, in_a_row = "no_output", False, False, 0
    for attempt in range(want + 4):          # bound: 1 extended + 1 clock retry + 2 fails
        if len(good) >= want:
            break
        log_file = f"{stem}_run{attempt}.csv"
        temp = gpu_temp(env)
        t0 = time.time()
        ok, out = run_hard_timeout(command(run_cmd, kernel, log_file, skip, count), cwd=cwd,
                                   timeout=timeout, env=env, capture=True, scope_dir=str(cwd))
        errors += [l.strip() for l in out.splitlines() if l.startswith("==ERROR==")]
        exec_s.append(round(time.time() - t0, 2))
        try:
            with open(log_file, errors="replace") as fh:
                text = fh.read()
            os.remove(log_file)
        except OSError:
            text = ""
        launches, errs = parse_log(text) if ok else ([], [])  # a killed run's log is partial
        errors += errs
        r = summarize(launches)
        r["temp"] = temp
        ref = ref_mhz or (good[0]["mhz"] if good else None)
        if r["status"] == "clock_unlocked":
            errors.append("clock varied within the run: %.0f-%.0f MHz" % tuple(r["mhz_range"]))
        elif r["status"] == "ok" and ref and abs(r["mhz"] - ref) > CLOCK_TOL * ref:
            r["status"] = "clock_unlocked"
            errors.append("clock %.0f MHz, %s %.0f MHz" % (
                r["mhz"], "baseline's" if ref_mhz else "first run's", ref))
        elif r["status"] == "ok" and r["n"] < n_base:
            r["status"] = "short"
        status = r["status"]
        if status == "ok":
            good.append(r)
            in_a_row = 0
            spread = (any(g["cv"] > CV_MAX for g in good) or
                      (baseline and len(good) >= 2 and
                       (max(g["ms"] for g in good) - min(g["ms"] for g in good))
                       > RUNS_DIFF_MAX * min(g["ms"] for g in good)))
            if len(good) == want and spread and not extended:
                extended, want = True, want + 1
        elif status == "clock_unlocked":
            if clock_retried:
                break
            clock_retried = True
        else:
            in_a_row += 1
            if in_a_row == 2:
                break
    res = {"status": "ok" if good else status, "exec_s": exec_s,
           "error": "; ".join(dict.fromkeys(errors))[:300] or None}
    if status == "clock_unlocked" and clock_retried:
        res["status"] = "clock_unlocked"        # the lock failed twice: trust no run of it
        res["mhz_seen"] = r.get("mhz_range")
    if res["status"] != "ok":
        return res
    runs = [g["ms"] for g in good]
    return {**res, "ms": statistics.median(runs), "runs": runs,
            "cv": max(g["cv"] for g in good), "n": min(g["n"] for g in good),
            **{k: _median(g[k] for g in good) for k in ("mhz", "cycles", "inst")},
            "passes": max((g["passes"] for g in good if g.get("passes") is not None),
                          default=None),
            "temp": [g["temp"] for g in good]}


def scope(kernel_parents: list, baseline: dict) -> tuple:
    """A loop's ncu scope -> (plan, baseline_ms), or (None, reason).  plan: {parent:
    {"skip", "count", "n_base", "w", "inst", "mhz"}} over the parents that launch (by the
    census, `launches.parents`).  w: share of the parent's launches per run (the
    census; the baseline's profiled counts if the census lacks one), so baseline_ms
    is the launch-weighted mean per-launch time; a one-parent loop has w = 1.
    reason: no_parent, or ncu_<status> of a parent the baseline could not measure
    (e.g. ncu_not_in_binary: the IR's mangled name is not the binary's)."""
    census = (baseline.get("launches") or {}).get("parents")
    ent = (baseline.get("ncu") or {}).get("parents") or {}
    ps = [p for p in dict.fromkeys(kernel_parents or []) if p
          and not (census is not None and census.get(p) == 0)]
    if not ps:
        return None, "no_parent"
    for p in ps:
        if "ms" not in (ent.get(p) or {}):
            return None, "ncu_" + (ent.get(p) or {}).get("failed", "not_measured")
    c = [(census or {}).get(p) for p in ps]
    if not all(c):
        c = [ent[p]["n"] for p in ps]
    plan = {p: {"skip": ent[p]["skip"], "count": ent[p]["count"], "n_base": ent[p]["n"],
                "w": ci / sum(c), "inst": ent[p].get("inst"), "mhz": ent[p].get("mhz")}
            for p, ci in zip(ps, c)}
    return plan, sum(v["w"] * ent[p]["ms"] for p, v in plan.items())


def combine(plan: dict, results: dict) -> "tuple[float, list]":
    """(cell_ms, per-run ms) over a loop's parents, with the plan's weights; cell_ms
    uses each parent's median, the per-run list pairs the parents' runs in order."""
    cell = sum(v["w"] * results[p]["ms"] for p, v in plan.items())
    n = min(len(results[p]["runs"]) for p in plan)
    runs = [sum(v["w"] * results[p]["runs"][i] for p, v in plan.items()) for i in range(n)]
    return cell, runs


def off_clock(baselines: dict, clock_mhz: float) -> list:
    """["bench parent MHz"] for timed baseline parents not within CLOCK_TOL of
    *clock_mhz* (collect_pipeline --ncu-clock-mhz: an optional sanity bound)."""
    return [f"{b} {p} {e['mhz']:.0f}" for b, v in sorted(baselines.items())
            for p, e in sorted(((v.get("ncu") or {}).get("parents") or {}).items())
            if "mhz" in e and abs(e["mhz"] - clock_mhz) > CLOCK_TOL * clock_mhz]
