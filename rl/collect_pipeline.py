"""
Decoupled, pipelined reward-cell collector — the GPU-throughput successor to
collect_cells.py.

Problem it solves: collect_cells runs compile -> nsys profile -> nsys stats/parse
-> feature-extract in SERIES on one GPU worker, so the GPU idles through every CPU
phase.  Here the GPU does ONLY `nsys profile`; compile, stats+parse, and post-unmerge
feature extraction run on separate CPU pools connected by bounded queues.  The reward
methodology is IDENTICAL to collect_cells (same 20 cold nsys runs, same command, same
median-of-sums, same clip/deadzone, same failure semantics) — this changes the
SCHEDULING, not the numbers.  Output is the same reward_cache.json (via save_cache), so
it is a drop-in for downstream/train and resumes from an existing cache.

Pipeline (per node, replicated per GPU; SHARED queues → self-balancing across GPUs):
    feeder(main) --work_q--> [compile ×C] --Q12--> [exec ×G, one/GPU] --Q23-->
        [parse ×P] --result_q--> main (writes reward_cache.json incrementally)
    feeder(main) --post_q--> [postfeat ×F]  (CPU-only, independent of the above)
Defaults per GPU: exec=1, compile=2, parse=3 (override with --exec/--compile/--parse,
which are PER-GPU counts; total = per-GPU × #GPUs).

Build isolation: each cell builds in its OWN dir under --shm-root (/dev/shm), so the
mandatory `make clean` in _make can never wipe another cell's binary.  The bundle is a
"structural copy" of the benchmark — build inputs (source/Makefile/headers, or any small
file) are COPIED; large data files are SYMLINKED to the on-disk original (page cache
serves them, so shm stays tiny and the metric — kernel time — is unaffected by data
location).  exec deletes the bundle right after its 20 runs; parse deletes the reports.

Bounded queues (Q12, Q23) bound shm/tmp and provide backpressure.  Memory monitor logs
per-queue depth + shm/tmp footprint + system RAM at --mem-log-interval to a JSONL file.

INVARIANTS (must match collect_cells / the existing cache): n_runs cold profiled runs per
cell with the SAME profiler command the single-cell measurement uses (hecbench's
_profile_once_nsys / _profile_once_rocprof — see exec_command):
  NVIDIA  `nsys profile --trace=cuda --sample=none --cpuctxsw=none` per run (exec, GPU),
          then `nsys stats --report=cuda_gpu_kern_sum` per report (parse, CPU);
  AMD     legacy `rocprof --stats -d <dir> -o <dir>/r.csv` per run (exec, GPU) — rocprof
          writes the pre-aggregated <dir>/r.stats.csv itself, so parse is a file read.
reward = clip_-1((baseline_ms - median(sum_kernel_times))/max(baseline_ms,1e-9)) then
deadzone; compile_failed/timeout → cached penalty; measure_failed → NOT cached (retried;
a cell whose own runs failed twice is skipped until --retry-failed).
--n-runs REQUIRED and must match the run.  Pin the profiler version across a collection.

--profiler ncu, the default on NVIDIA (README §12, CLOCK_LOCK.md): each cell is timed as the
loop's parent kernel(s) by exact mangled name, launches 3..22 per run, at the locked
base clock (ncu_timer).  The GPU stage runs ncu once per parent and reads its CSV (the
spread decides whether to run once more), so parse only scores; with a run of a few
seconds the job is compile-bound (raise --compile).  reward = clip/deadzone of
(baseline - cell) / baseline over the launch-weighted mean per-launch time; a run off
the clock (clock_unlocked) is not cached and counts like no_output; a run that ends
before its baseline's launch count is a launch_mismatch.  The timer is stamped into
both caches, and a run refuses caches timed another way.  --n-runs 2 (each run: 20
launches; a cell's clock must match the one its baseline recorded).

AMD specifics: the GPU stage pins BOTH CUDA_VISIBLE_DEVICES and HIP_VISIBLE_DEVICES (a HIP
binary ignores the CUDA variable); bundles are laid out <key>/<bench> so hecbench._make's
sibling-header symlink for -hip benchmarks fires (store_common.bundle_paths); the target
is taken from hecbench.IS_HIP, i.e. TARGET_ARCH — --arch must agree with it.

Usage (baselines first, once per collection; then any number of cell jobs):
  python3 collect_pipeline.py CKPT_DIR --split test --n-runs 20 \\
      --compile-failure-penalty -0.16 --reward-deadzone 0.005 --baselines-only
  python3 collect_pipeline.py CKPT_DIR --split test --n-runs 20 \\
      --compile-failure-penalty -0.16 --reward-deadzone 0.005 \\
      --exec 1 --compile 2 --parse 3
  python3 collect_pipeline.py CKPT_DIR --split test --n-runs 20 ... --dry-run
  python3 collect_pipeline.py --selftest        # pure-logic checks, no GPU/benchmarks
"""

import argparse
import json
import logging
import os
import queue as _queue
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# NOTE: all heavy imports (torch.multiprocessing, hecbench, train, collect_cells)
# are LAZY — done inside main() and the worker functions that use them — so
# `--selftest` runs on stdlib alone, and CPU workers never import torch eagerly.
# The reused helpers (valid_cells/missing_cells/save_cache, compile/parse/nsys
# functions) are imported where used so the schedule change cannot drift from the
# methodology.

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("pipeline")

# Shared, stdlib-only leaf (safe for --selftest; no torch): cell key, bundle copy,
# shm/tmp accounting, and the pre-compiled-store fast path.
from store_common import (cellkey, sanitize, bundle_paths, structural_copy, dir_bytes,
                          stage_from_store, is_cell_done, load_manifest, toolchain_stamp,
                          benchmark_fingerprint, run_hard_timeout, device_code_hash,
                          run_target_rel)
import ncu_timer                   # stdlib-only too

_STOP = None                       # queue sentinel (picklable, and no work item is None)
# A cell whose runs failed (no profiler output / launch-count mismatch) this many
# times is skipped by later runs unless --retry-failed; so is a benchmark whose
# baseline failed this many times.
MAX_MEASURE_ATTEMPTS = 2
# This many cells in a row without profiler output on one GPU, over at least
# GPU_DEAD_BENCHES benchmarks, means the GPU or the profiler is failing: the job
# stops instead of letting that GPU drain the shared queue.
GPU_DEAD_CELLS, GPU_DEAD_BENCHES = 5, 3


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested by --selftest; no I/O, no GPU)
# ---------------------------------------------------------------------------

def compute_reward(baseline_ms: float, modified_ms: float, deadzone: float) -> float:
    """EXACT copy of collect_cells' reward math (worker lines 317-321)."""
    reward = max((baseline_ms - modified_ms) / max(baseline_ms, 1e-9), -1.0)
    if deadzone > 0 and abs(reward) < deadzone:
        reward = 0.0
    return reward


def _limit_by_benchmark(items: list, n: int) -> list:
    """Take at most n items, ROUND-ROBIN across benchmarks, so a small --limit
    validation sample spreads over many benchmarks (exercises many distinct
    compile/run paths) instead of draining one benchmark's cells.  Deterministic:
    preserves first-seen benchmark order and each benchmark's internal order.
    Returns the SAME list object (identity) when n<=0 or n>=len — callers rely on
    this to mean 'no limit'."""
    if n <= 0 or n >= len(items):
        return items
    from collections import OrderedDict
    groups: "OrderedDict[str, list]" = OrderedDict()
    for it in items:
        groups.setdefault(it["benchmark_name"], []).append(it)
    lists = list(groups.values())
    out: list = []
    idx = 0
    while len(out) < n:
        progressed = False
        for lst in lists:
            if idx < len(lst):
                out.append(lst[idx])
                progressed = True
                if len(out) >= n:
                    break
        if not progressed:                 # every benchmark exhausted
            break
        idx += 1
    return out


def _sys_mem_gb() -> tuple:
    """(available_gb, total_gb) from /proc/meminfo; (0,0) if unavailable."""
    try:
        info = {}
        with open("/proc/meminfo") as fh:
            for line in fh:
                k, _, v = line.partition(":")
                info[k.strip()] = int(v.split()[0])          # kB
        return (round(info.get("MemAvailable", 0) / 1048576, 2),
                round(info.get("MemTotal", 0) / 1048576, 2))
    except Exception:
        return (0.0, 0.0)


def resolve_kernel_scope(kernel_parents: list, baseline_cache: dict, bench: str):
    """(kernel_filter, baseline_ms) — the coupled pair, EXACT copy of collect_cells
    worker lines 267-275 / environment resolution.  Per-loop (all a loop's cells
    share it), so callers resolve once per loop."""
    from hecbench import demangle, demangled_to_filter
    kernel_filter = None
    baseline_ms = baseline_cache.get(bench, {}).get("total_ms", 0.0)
    if len(kernel_parents) == 1:
        kf = demangled_to_filter(demangle(kernel_parents[0]))
        per_kern = baseline_cache.get(bench, {}).get("per_kernel_ms", {}).get(kf)
        if per_kern is not None:
            kernel_filter, baseline_ms = kf, per_kern
    return kernel_filter, baseline_ms


def stale_baselines(bench_paths: dict, baseline_cache: dict, arch: str) -> dict:
    """{bench: reason} for baselines this run must not score against.  The reward is
    a ratio against the baseline, so both sides must be built by the same compiler
    and run with the same command: a Makefile run target edited after the baseline
    (a cut repeat count) or a rebuilt clang would change the denominator silently.
    Entries without n_ok_runs predate the per-run nsys fix, i.e. each is one run."""
    from hecbench import _get_run_command
    live = toolchain_stamp(arch)
    out = {}
    for bench, path in sorted(bench_paths.items()):
        bl = baseline_cache.get(bench)
        if bl is None:
            continue
        if "n_ok_runs" not in bl:
            out[bench] = "measured before the per-run nsys fix (a single run)"
        elif bl.get("toolchain") != live:
            out[bench] = f"compiler differs: baseline {bl.get('toolchain')} vs now {live}"
        elif bl.get("run_cmd") != _get_run_command(Path(path), arch):
            out[bench] = (f"run command differs: baseline {bl.get('run_cmd')!r} vs now "
                          f"{_get_run_command(Path(path), arch)!r}")
    return out


def resolve_profiler(choice: str, is_hip: bool) -> str:
    """--profiler -> the timer: auto is ncu at the locked base clock on NVIDIA and the
    legacy rocprof timer on AMD (ncu has no AMD counterpart, so the lock is a no-op
    there); nsys is rocprof on AMD too.  ncu on AMD is refused."""
    if choice == "ncu" and is_hip:
        sys.exit("--profiler ncu is NVIDIA only")
    if is_hip:
        return "rocprof"
    return "ncu" if choice == "auto" else choice


def track_measure_attempts(msg: dict, pending: dict, measure_failed: dict):
    """Attempt bookkeeping (MAX_MEASURE_ATTEMPTS) for one terminal cell result.  A
    launch mismatch counts at once; a cell without profiler output (or, under ncu,
    with an unlocked clock) counts only once its GPU has measured a cell after it, so
    a failing GPU (Xid, lost device, full tmp) cannot use up the attempts of every
    cell it reaches (*pending*: gpu -> [(key, status)] since its last measurement).
    Node errors and kills never count.
    Returns the GPU's id once it has GPU_DEAD_CELLS such cells in a row over
    GPU_DEAD_BENCHES benchmarks, else None."""
    def count(key, status):
        mf = measure_failed.setdefault(key, {"attempts": 0})
        mf.update(attempts=mf["attempts"] + 1, status=status)
    side = msg.get("sidecar") or {}
    if side.get("run_ms"):
        for k, status in pending.pop(side.get("gpu"), []):
            count(k, status)
    if msg["type"] != "measure_failed":
        measure_failed.pop(msg["key"], None)
    elif msg.get("status") == "launch_mismatch":
        count(msg["key"], "launch_mismatch")
    elif msg.get("status") in ("no_output", "clock_unlocked"):
        q = pending.setdefault(side.get("gpu"), [])
        q.append((msg["key"], msg["status"]))
        if len(q) >= GPU_DEAD_CELLS and len({k.split("|")[0] for k, _ in q}) >= GPU_DEAD_BENCHES:
            return side.get("gpu")
    return None


def exec_command(is_hip: bool, run_cmd: str, out: str) -> str:
    """ONE profiled run of a cell — the same command hecbench uses for a single
    measurement (_profile_once_rocprof / _profile_once_nsys), so the pipeline changes
    the scheduling, never the measurement.  *out* is an ABSOLUTE per-run path (the
    profiler resolves it against cwd=bundle): on AMD the directory rocprof writes into
    (-d) with the csv stem inside it (-o); on NVIDIA the nsys report stem."""
    if is_hip:
        return f"rocprof --stats -d {out} -o {out}/r.csv {run_cmd}"
    return (f"nsys profile --trace=cuda --sample=none --cpuctxsw=none "
            f"--output={out} --force-overwrite=true {run_cmd}")


def artifact_for(is_hip: bool, out: str) -> str:
    """The file parse reads for that run: rocprof's pre-aggregated stats csv (rocprof
    derives it from -o as <stem>.stats.csv) or the nsys report."""
    return f"{out}/r.stats.csv" if is_hip else f"{out}.nsys-rep"


def score_ncu(item: dict, deadzone: float) -> dict:
    """The parse stage of one --profiler ncu cell: its ncu_timer results per parent ->
    the terminal message.  A parent that ran short of its baseline's launches is a
    launch_mismatch; no_output / clock_unlocked are measure_failed with that status."""
    res, plan = item["ncu"], item["ncu_plan"]
    side = {**item.get("sidecar", {}), "baseline_ms": item["baseline_ms"], "run_ms": [],
            "ncu": {p: {k: v for k, v in r.items() if k not in ("exec_s", "runs")
                        and v is not None} for p, r in res.items()}}
    for p, r in res.items():
        if r.get("inst") and plan[p].get("inst"):
            side["ncu"][p]["inst_ratio"] = round(r["inst"] / plan[p]["inst"], 4)
    msg = {"key": item["key"], "benchmark": item["benchmark"], "sidecar": side}
    bad = next((r["status"] for r in res.values() if r["status"] != "ok"), None)
    if bad or len(res) < len(plan):
        return {**msg, "type": "measure_failed",
                "status": "launch_mismatch" if bad == "short" else bad or "no_output"}
    cell_ms, run_ms = ncu_timer.combine(plan, res)
    side["run_ms"] = [round(t, 6) for t in run_ms]
    return {**msg, "type": "cell",
            "reward": float(compute_reward(item["baseline_ms"], cell_ms, deadzone))}


# ---------------------------------------------------------------------------
# Stage workers  (each: plain-data queues only; GPU visibility set first)
# ---------------------------------------------------------------------------

def build_chains(work_items: list, validate_frac: float) -> list:
    """Group cells into compile chains: one per (benchmark, loop, unmerge), factors
    ascending, in first-seen order (so the queue's fastest-first / store-hits-first
    order carries over).  A chain is compiled in factor order and STOPS at its first
    failed compile (compile_worker): in the 8,100-cell run_sweep_3 cache compile
    failures were monotone in the factor in 352/353 rows and timeouts in 33/33 — the
    unmerge clones sum_{i<factor} numPaths^i bodies and the unroller's size is linear
    in the factor.  A deterministic `validate_frac` of chains (hash of the chain id)
    also compiles the factor after the stop, to measure the rule during the run."""
    from collections import OrderedDict
    import hashlib
    groups: "OrderedDict[tuple, dict]" = OrderedDict()
    for w in work_items:
        cid = (w["benchmark_name"], w["loop_idx"], w["unmerge"])
        if cid not in groups:
            groups[cid] = {k: v for k, v in w.items() if k != "factor"}
            groups[cid]["factors"] = []
        groups[cid]["factors"].append(w["factor"])
    chains = list(groups.values())
    for c in chains:
        c["factors"].sort()
        h = hashlib.md5(f"{c['benchmark_name']}|{c['loop_idx']}|{c['unmerge']}".encode())
        c["validate"] = int(h.hexdigest()[:8], 16) / 0xFFFFFFFF < validate_frac
    return chains


def penalty_type(status: str) -> str:
    """Terminal result type (= reward class) for a compile status: a plain failure
    keeps the failure penalty; a timeout, a budget refusal (the pass's fast stand-in
    for a timeout) and an allocation failure under the memory cap take the timeout
    penalty."""
    return "compile_failed" if status == "compile_failed" else "compile_timeout"


def compile_worker(rank, work_q, q12, result_q, cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""            # CPU-only stage: never grab a GPU
    # Per-compile address-space cap (hecbench._make): a compile that outgrows it fails
    # its own allocation (compile_oom, deterministic for the cell) long before the
    # node's OOM killer picks a victim by load.
    if cfg.get("compile_mem_gb"):
        os.environ["UU_COMPILE_MEM_GB"] = str(cfg["compile_mem_gb"])
    else:
        os.environ.pop("UU_COMPILE_MEM_GB", None)
    from hecbench import compile_single_loop_ex, _get_run_command, CompileResourceExceeded
    from store_common import classify_compile_error
    lg = logging.getLogger(f"compile.{rank}")
    shm = Path(cfg["shm_root"])
    prebuilt = cfg.get("prebuilt") or set()           # cellkeys with a verified store binary
    store = cfg.get("store")
    pen = {"compile_failed": cfg["compile_failure_penalty"],
           "compile_timeout": cfg["compile_timeout_penalty"]}

    def build(item, key, f, bundle):
        """-> (status, signature, compile_s).  status: staged/ok/compile_failed/
        compile_timeout/compile_budget/compile_oom/compile_killed/compile_env.  A
        kill is retried once."""
        t0 = time.time()
        # FAST PATH: drop the pre-built binary (no make) if this cell is in the store;
        # stage_from_store unlinks any existing target first (never writes through a
        # symlink), and returns False → we fall back to building.
        if key in prebuilt and store and stage_from_store(store, key, bundle):
            return "staged", "", round(time.time() - t0, 2)
        for attempt in (0, 1):
            try:
                ok, err = compile_single_loop_ex(
                    bundle, loop_idx=item["loop_idx"], unmerge=item["unmerge"], factor=f,
                    filename=item["filename"], triple=item["triple"], arch=cfg["arch"],
                    timeout=cfg["compile_timeout"])
                status = "ok" if ok else classify_compile_error(err)   # failed / env
            except CompileResourceExceeded as e:            # budget / allocation / kill
                status, err = e.status, e.signature
            except subprocess.TimeoutExpired:
                return "compile_timeout", "", round(time.time() - t0, 2)
            if status != "compile_killed" or attempt:
                return status, err, round(time.time() - t0, 2)
            lg.warning("compile of %s killed (%s) — retrying once", key, err)

    while True:
        chain = work_q.get()
        if chain is _STOP:
            break
        bench, li, u = chain["benchmark_name"], chain["loop_idx"], chain["unmerge"]
        stop = None             # status of the chain's first failed compile
        validated = False       # the one post-stop validation compile is spent
        for f in chain["factors"]:
            key = cellkey(bench, li, u, f)
            validation = False
            if stop is not None and cfg["infer_after_failure"] and key not in prebuilt:
                # (a cell the pre-compiled store holds did compile: stage it, never infer)
                if chain["validate"] and not validated:
                    validated = validation = True      # compile this one for real
                else:
                    result_q.put({"type": penalty_type(stop), "key": key, "benchmark": bench,
                                  "reward": pen[penalty_type(stop)], "status": stop,
                                  "inferred": True,
                                  "error": f"inferred: {stop} at a lower factor"})
                    continue
            # <shm>/bundles/<key>/<bench>: the benchmark's own name innermost, so _make's
            # -hip sibling-header symlink fires (see store_common.bundle_paths).
            bundle_root, bundle = bundle_paths(shm / "bundles", sanitize(key), bench)
            try:
                shutil.rmtree(bundle_root, ignore_errors=True)
                structural_copy(Path(chain["benchmark_path"]), bundle, cfg["copy_threshold"])
                status, err, compile_s = build(chain, key, f, bundle)
                if status in ("compile_env", "compile_killed"):
                    # The node failed (disk, fork, missing tool or header; a SIGKILL
                    # twice: the OOM killer, whose victim depends on what else runs),
                    # not the cell: retried on the next run, and the row goes on.
                    shutil.rmtree(bundle_root, ignore_errors=True)
                    lg.warning("compile of %s: %s (%s) — not cached", key, status, err)
                    result_q.put({"type": "measure_failed", "key": key, "benchmark": bench,
                                  "status": status, "error": err, "compile_s": compile_s})
                    continue
                if status not in ("ok", "staged"):
                    shutil.rmtree(bundle_root, ignore_errors=True)
                    result_q.put({"type": penalty_type(status), "key": key, "benchmark": bench,
                                  "reward": pen[penalty_type(status)], "status": status,
                                  "error": err, "compile_s": compile_s,
                                  "validation": validation})
                    if stop is None:
                        stop = status
                    continue
                if validation:
                    # The stop did not hold for this row: compile the rest for real.
                    lg.warning("chain %s|%d|%d: factor %d compiled after a %s at a lower "
                               "factor — inference contradicted, resuming", bench, li, u,
                               f, stop)
                    stop = None
                run_cmd = _get_run_command(bundle, cfg["arch"])
                dev_hash = device_code_hash(bundle / run_target_rel(run_cmd))
                if (cfg["skip_identical"] and dev_hash
                        and dev_hash == chain.get("baseline_hash")):
                    # Device code byte-identical to the baseline's: the reward is 0
                    # by construction, no GPU run (~16% of compiled cells in a
                    # 483-cell HeCBench sample).
                    shutil.rmtree(bundle_root, ignore_errors=True)
                    result_q.put({"type": "cell", "key": key, "benchmark": bench,
                                  "reward": 0.0, "status": "identical",
                                  "compile_s": compile_s, "validation": validation,
                                  "sidecar": {"device_hash": dev_hash}})
                    continue
                # bundle now lives in shm until exec deletes it (immutable in between).
                q12.put({"key": key, "benchmark": bench, "loop_idx": li,
                         "unmerge": u, "factor": f, "bundle": str(bundle),
                         "bundle_root": str(bundle_root),
                         "run_cmd": run_cmd, "kernel_filter": chain["kernel_filter"],
                         "baseline_ms": chain["baseline_ms"],
                         "baseline_launches": chain.get("baseline_launches"),
                         "ncu_plan": chain.get("ncu_plan"),
                         "status": status, "compile_s": compile_s,
                         "validation": validation, "device_hash": dev_hash})
            except Exception as e:
                shutil.rmtree(bundle_root, ignore_errors=True)
                lg.warning("compile crashed for %s: %s — treated as measure_failed", key, e)
                result_q.put({"type": "measure_failed", "key": key, "benchmark": bench,
                              "status": "compile_crashed", "error": str(e)[:200]})
    result_q.put({"type": "compile_done", "rank": rank})


def exec_worker(rank, gpu_id, q12, q23, result_q, cfg):
    # One profiled run at a time per GPU.  Pin BOTH runtimes before anything else: a HIP
    # binary ignores CUDA_VISIBLE_DEVICES, so without the HIP variable every AMD exec
    # worker lands on GCD 0 and the GCDs contend (MIRROR: hecbench.measure_kernel_time,
    # collect_cells._collect_worker, train._baseline_worker).
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["HIP_VISIBLE_DEVICES"] = str(gpu_id)
    is_hip = cfg["is_hip"]
    # Absolute: the profiler's output path (-d/-o or --output) is resolved against
    # cwd=bundle, while parse reads the artifact from here.
    tmp = Path(cfg["tmp_root"]).resolve()
    my_tmpdir = tmp / f"nsys_tmp_r{rank}"             # per-WORKER TMPDIR (profiler scratch
    #                                                  isolation; per-rank, not per-gpu, so
    #                                                  >1 exec per GPU still get separate temp)
    my_tmpdir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(my_tmpdir)
    env = {**os.environ, "ARCH": cfg["arch"], "CUDA_VISIBLE_DEVICES": str(gpu_id),
           "HIP_VISIBLE_DEVICES": str(gpu_id), "TMPDIR": str(my_tmpdir)}
    lg = logging.getLogger(f"exec.g{gpu_id}")
    while True:
        item = q12.get()
        if item is _STOP:
            break
        key, bundle = item["key"], Path(item["bundle"])
        rep_dir = tmp / "reports" / sanitize(key)
        rep_dir.mkdir(parents=True, exist_ok=True)
        reports, exec_s, in_a_row, ncu_res = [], [], 0, None
        if item.get("ncu_plan"):
            # ncu: each parent kernel alone, as its baseline was (ncu_timer.measure);
            # read here, since the spread decides whether to run once more.
            ncu_res = {}
            for j, (p, pl) in enumerate(item["ncu_plan"].items()):
                r = ncu_timer.measure(item["run_cmd"], bundle, env, cfg["nsys_timeout"], p,
                                      pl["skip"], pl["count"], str(rep_dir / f"p{j}"),
                                      cfg["n_runs"], n_base=pl["n_base"], ref_mhz=pl["mhz"])
                ncu_res[p] = r
                exec_s += r["exec_s"]
                if r["status"] != "ok":
                    break
        for k in range(cfg["n_runs"]):
            if ncu_res is not None or (in_a_row == 2 and not reports):
                break          # ncu: done above; nsys: the first two runs failed
                #                (MIRROR: _measure_one_baseline)
            out = rep_dir / f"run{k}"
            if is_hip:
                out.mkdir(parents=True, exist_ok=True)  # rocprof -d dir: results/json/db + stats
            # HARD timeout that kills the profiler + the benchmark (process group), so a
            # slow run never orphans a ./main that would then contend with the next run
            # on this GPU and corrupt its timing.  ok=False → no artifact → run skipped.
            t0 = time.time()
            ok, _ = run_hard_timeout(
                exec_command(is_hip, item["run_cmd"], str(out)),
                cwd=bundle, timeout=cfg["nsys_timeout"], env=env, scope_dir=str(bundle))
            exec_s.append(round(time.time() - t0, 2))
            in_a_row += 1
            if not ok:
                continue                               # timed out → killed → skip run
            rep = Path(artifact_for(is_hip, str(out)))
            if not rep.exists() and is_hip:
                # MIRROR: _profile_once_rocprof's fallback for a version that names the
                # stats file differently.
                cands = sorted(out.rglob("*.stats.csv"))
                if cands:
                    rep = cands[0]
            if rep.exists():
                reports.append(str(rep))
                in_a_row = 0
        shutil.rmtree(item["bundle_root"], ignore_errors=True)   # free shm (last reader)
        q23.put({"key": key, "benchmark": item["benchmark"], "reports": reports,
                 "report_dir": str(rep_dir), "kernel_filter": item["kernel_filter"],
                 "baseline_ms": item["baseline_ms"],
                 "baseline_launches": item.get("baseline_launches"),
                 "ncu": ncu_res, "ncu_plan": item.get("ncu_plan"),
                 "sidecar": {"status": item.get("status"), "compile_s": item.get("compile_s"),
                             "validation": item.get("validation", False),
                             "device_hash": item.get("device_hash"),
                             "exec_s": exec_s, "gpu": gpu_id}})
    result_q.put({"type": "exec_done", "rank": rank})


def parse_worker(rank, q23, result_q, cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""            # CPU-only
    from hecbench import (_parse_nsys_kernel_times, _parse_rocprof_kernel_times,
                          _sum_kernel_times, _parse_nsys_kernel_instances,
                          _sum_kernel_instances)
    is_hip = cfg["is_hip"]
    env = {**os.environ, "ARCH": cfg["arch"]}
    lg = logging.getLogger(f"parse.{rank}")
    while True:
        item = q23.get()
        if item is _STOP:
            break
        key, kf = item["key"], item["kernel_filter"]
        if item.get("ncu") is not None:
            shutil.rmtree(item["report_dir"], ignore_errors=True)
            result_q.put(score_ncu(item, cfg["reward_deadzone"]))
            continue
        times, launches = [], []
        for rep in item["reports"]:
            if is_hip:
                # rocprof already aggregated per kernel at exec time: a file read.
                # MIRROR: hecbench._profile_once_rocprof → _parse_rocprof_kernel_times.
                kt = _parse_rocprof_kernel_times(rep)
            else:
                # same process-group-kill guard: a hung `nsys stats` would otherwise orphan.
                # MIRROR: hecbench._profile_once_nsys (per-run report, --force-export).
                ok, out = run_hard_timeout(
                    f"nsys stats --report=cuda_gpu_kern_sum --format=csv --force-export=true "
                    f"{rep}", cwd=".", timeout=30, env=env, capture=True)
                if not ok:
                    continue
                kt = _parse_nsys_kernel_times(out)
                if kt:
                    launches.append(_sum_kernel_instances(_parse_nsys_kernel_instances(out), kf))
            t = _sum_kernel_times(kt, kf)
            if t is not None:
                times.append(t)
        shutil.rmtree(item["report_dir"], ignore_errors=True)   # free tmp (last reader)
        side = {**item.get("sidecar", {}), "kernel_filter": kf,
                "baseline_ms": item["baseline_ms"], "run_ms": [round(t, 6) for t in times],
                "launches": launches, "baseline_launches": item.get("baseline_launches")}
        if not times:
            result_q.put({"type": "measure_failed", "key": key, "status": "no_output",
                          "benchmark": item["benchmark"], "sidecar": side})
            continue
        # Launch-count guard: a run that crashed or exited early launches fewer
        # kernels than the baseline and would score as a speedup.  Not cached; the
        # baseline count is only given when it was the same in every baseline run.
        bl = item.get("baseline_launches")
        if bl and launches and statistics.median_low(launches) != bl:
            result_q.put({"type": "measure_failed", "key": key, "status": "launch_mismatch",
                          "benchmark": item["benchmark"],
                          "sidecar": {**side, "launch_mismatch": True}})
            continue
        reward = compute_reward(item["baseline_ms"], statistics.median(times),
                                cfg["reward_deadzone"])
        result_q.put({"type": "cell", "key": key, "benchmark": item["benchmark"],
                      "reward": float(reward), "sidecar": side})
    result_q.put({"type": "parse_done", "rank": rank})


def postfeat_worker(rank, post_q, result_q, cfg):
    """CPU-only post-unmerge feature extraction (compile unmerge=1 + LoopCount, no
    nsys/GPU).  Mirrors collect_cells worker lines 338-361."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from environment import GpuLoopEnv, LoopRecord
    from agent import _IDX_TRIP_COUNT_KNOWN, _IDX_TRIP_COUNT
    from hecbench import FeatureNormalizer
    lg = logging.getLogger(f"postfeat.{rank}")
    normalizer = FeatureNormalizer.from_state_dict(cfg["normalizer_state"])
    env = GpuLoopEnv(arch=cfg["arch"], n_runs=cfg["n_runs"],
                     nsys_timeout=cfg["nsys_timeout"], tmp_dir=Path(cfg["tmp_root"]),
                     gpu_id=0, normalizer=normalizer, baseline_cache={})
    shm = Path(cfg["shm_root"])
    while True:
        item = post_q.get()
        if item is _STOP:
            break
        bench, li = item["benchmark_name"], item["loop_idx"]
        bundle_root, bundle = bundle_paths(shm / "postfeat", f"{bench}__{li}", bench)
        try:
            shutil.rmtree(bundle_root, ignore_errors=True)
            structural_copy(Path(item["benchmark_path"]), bundle, cfg["copy_threshold"])
            env._benchmark_dir = bundle
            raw = item["pre_features_raw"]
            pre = normalizer.normalize(torch.tensor(raw, dtype=torch.float32))
            rec = LoopRecord(loop_idx=li, filename=item["filename"],
                             triple=item["triple"], pre_features=pre,
                             kernel_parents=item["kernel_parents"],
                             trip_count_known=raw[_IDX_TRIP_COUNT_KNOWN] > 0.5,
                             trip_count=int(raw[_IDX_TRIP_COUNT]))
            post = env.get_post_unmerge_features(rec)
            if not torch.equal(post.cpu(), pre.cpu()):      # only store REAL extractions
                result_q.put({"type": "postfeat", "benchmark": bench, "loop_idx": li,
                              "features": post.detach().cpu().tolist()})
        except Exception as e:
            lg.warning("postfeat failed for %s|%d: %s", bench, li, e)
        finally:
            shutil.rmtree(bundle_root, ignore_errors=True)
    result_q.put({"type": "postfeat_done", "rank": rank})


# ---------------------------------------------------------------------------
# Feeder + memory monitor (threads in main)
# ---------------------------------------------------------------------------

def _inject_stops(q, n, consumer_procs):
    """Put n STOP sentinels, but never block main forever: if the consumers have all
    died, a full queue would never drain and a naive put() would hang. Give up in that
    case — the drain loop's liveness check then breaks cleanly."""
    for _ in range(n):
        while True:
            try:
                q.put(_STOP, timeout=5)
                break
            except _queue.Full:
                if not any(p.is_alive() for p in consumer_procs):
                    return


def feed_queue(items, q, n_stops):
    """Feed all items then n_stops sentinels onto q, in its OWN thread. Blocks on a
    bounded q (backpressure) — which is why work and post get SEPARATE feeders: a full
    post_q must never stall the work feeder (the GPU-critical path), and vice versa."""
    for it in items:
        q.put(it)
    for _ in range(n_stops):
        q.put(_STOP)


def mem_monitor(stop_evt, path, queues, cfg, counters, interval):
    shm, tmp = Path(cfg["shm_root"]), Path(cfg["tmp_root"])
    with open(path, "w") as fh:
        while not stop_evt.wait(interval):
            avail, total = _sys_mem_gb()
            def _qsz(q):
                try:
                    return q.qsize()
                except Exception:
                    return -1                # qsize unsupported on this platform
            rec = {
                "ts": round(time.time(), 1),
                "work_q": _qsz(queues["work"]), "q12": _qsz(queues["q12"]),
                "q23": _qsz(queues["q23"]), "post_q": _qsz(queues["post"]),
                "result_q": _qsz(queues["result"]),
                "shm_mb": round(dir_bytes(shm) / 1e6, 1),
                "tmp_mb": round(dir_bytes(tmp) / 1e6, 1),
                "sys_mem_avail_gb": avail, "sys_mem_total_gb": total,
                "cells_done": counters["cells_done"], "cells_total": counters["n_missing"],
            }
            fh.write(json.dumps(rec) + "\n")
            fh.flush()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _selftest() -> int:
    import tempfile
    ok = True
    def chk(name, cond):
        nonlocal ok
        ok = ok and cond
        print(f"  {'PASS' if cond else 'FAIL'}  {name}")

    chk("cellkey", cellkey("b", 3, 1, 4) == "b|3|1|4")
    # --limit round-robins across benchmarks and is identity-safe at the no-cap edges
    _items = [{"benchmark_name": b} for b in ["a", "a", "a", "b", "b", "c"]]
    chk("limit spreads across benchmarks",
        [x["benchmark_name"] for x in _limit_by_benchmark(_items, 3)] == ["a", "b", "c"])
    chk("limit 4 round-robin wraps",
        [x["benchmark_name"] for x in _limit_by_benchmark(_items, 4)] == ["a", "b", "c", "a"])
    chk("limit n>=len returns same list", _limit_by_benchmark(_items, 99) is _items)
    chk("limit 0 returns same list", _limit_by_benchmark(_items, 0) is _items)
    chk("limit never exceeds n", len(_limit_by_benchmark(_items, 5)) == 5)
    # run_hard_timeout: clean exit is True; on timeout the GRANDCHILD is killed too (the
    # orphaned-nsys bug).  A surviving grandchild would touch `marker` after 2s; timeout
    # fires at 1s and killpg must reap the whole group so the marker never appears.
    import time as _time
    chk("hard_timeout clean exit", run_hard_timeout("exit 0", cwd=".", timeout=10,
                                                    env=dict(os.environ))[0] is True)
    with tempfile.TemporaryDirectory() as d:
        marker = Path(d) / "grandchild_ran"
        t0 = _time.time()
        ok_to, _ = run_hard_timeout(f"sh -c 'sleep 2; touch {marker}'",
                                    cwd=d, timeout=1, env=dict(os.environ))
        chk("hard_timeout returns False on timeout", ok_to is False)
        chk("hard_timeout kills promptly", _time.time() - t0 < 5)
        _time.sleep(2.5)                    # past when a SURVIVING grandchild would touch it
        chk("hard_timeout killed the grandchild (whole process group)", not marker.exists())
    # tree-kill: a child that escapes to its OWN process group (as nsys launches ./main)
    # would survive killpg — the ppid tree-walk must still reap it.  Linux-only (/proc).
    if sys.platform.startswith("linux"):
        with tempfile.TemporaryDirectory() as d:
            marker = Path(d) / "regrouped_ran"
            script = Path(d) / "mk.py"
            script.write_text(
                "import os,time\n"
                "if os.fork()==0:\n"
                "    os.setpgrp()                 # NEW process group → killpg(parent) misses it\n"
                "    time.sleep(2)\n"
                f"    open(r'{marker}','w').close()\n"
                "    os._exit(0)\n"
                "time.sleep(30)\n"                # parent lingers so the tree is alive at timeout
            )
            ok_tk, _ = run_hard_timeout(f"{sys.executable} {script}", cwd=d, timeout=1,
                                        env=dict(os.environ))
            chk("hard_timeout returns False (tree-kill case)", ok_tk is False)
            _time.sleep(2.5)
            chk("hard_timeout tree-kill reaps a child in its OWN process group",
                not marker.exists())
        # cwd sweep in ISOLATION: a target that escapes BOTH the group and the ppid tree
        # (setsid → new session + reparented to init) but runs in scope_dir.  killpg and
        # tree-kill both miss it; only _pids_with_cwd(scope_dir) reaps it.  The outer shell
        # sleeps so the timeout still fires.
        with tempfile.TemporaryDirectory() as d:
            scope = Path(d) / "bundle"
            scope.mkdir()
            marker = scope / "escaped_ran"
            cmd = (f"setsid sh -c 'cd {scope}; sleep 2; touch {marker}' ; sleep 30")
            ok_cw, _ = run_hard_timeout(cmd, cwd=d, timeout=1, env=dict(os.environ),
                                        scope_dir=str(scope))
            chk("hard_timeout returns False (cwd-sweep case)", ok_cw is False)
            _time.sleep(2.5)
            chk("hard_timeout cwd-sweep reaps a process reparented out of the tree",
                not marker.exists())
    # per-run profiler command/artifact: byte-for-byte what hecbench's single-cell
    # measurement runs (_profile_once_nsys / _profile_once_rocprof)
    chk("exec cmd nvidia", exec_command(False, "./main 1", "/t/run0") ==
        "nsys profile --trace=cuda --sample=none --cpuctxsw=none --output=/t/run0 "
        "--force-overwrite=true ./main 1")
    chk("exec cmd amd", exec_command(True, "./main 1", "/t/run0") ==
        "rocprof --stats -d /t/run0 -o /t/run0/r.csv ./main 1")
    chk("artifact nvidia", artifact_for(False, "/t/run0") == "/t/run0.nsys-rep")
    chk("artifact amd", artifact_for(True, "/t/run0") == "/t/run0/r.stats.csv")
    # bundle layout: benchmark name innermost (the -hip sibling-symlink guard in _make
    # keys on it), private root per cell, structural_copy lands inside it
    with tempfile.TemporaryDirectory() as d:
        root, b = bundle_paths(Path(d) / "bundles", "x-hip__3__1__4", "x-hip")
        chk("bundle: benchmark name innermost", b.name == "x-hip" and b.parent == root)
        chk("bundle: root is per-cell", root.name == "x-hip__3__1__4")
        src = Path(d) / "src"
        src.mkdir()
        (src / "Makefile").write_text("all:")
        structural_copy(src, b, threshold=1024)
        chk("bundle: structural_copy creates the nested dir", (b / "Makefile").is_file())
        chk("bundle: sibling symlink slot is inside the root",
            (root / "x-cuda").parent == root and not (root / "x-cuda").exists())
    # compile chains: one per (bench, loop, unmerge), factors ascending, first-seen order
    _w = [{"benchmark_name": b, "loop_idx": li, "unmerge": u, "factor": f, "x": 1}
          for (b, li, u, f) in [("a", 0, 1, 3), ("a", 0, 0, 2), ("a", 0, 1, 1), ("b", 2, 1, 2),
                                ("a", 0, 1, 2)]]
    _c = build_chains(_w, 0.0)
    chk("chains: grouped per row in first-seen order",
        [(c["benchmark_name"], c["unmerge"]) for c in _c] == [("a", 1), ("a", 0), ("b", 1)])
    chk("chains: factors ascending", _c[0]["factors"] == [1, 2, 3])
    chk("chains: per-cell fields kept, factor dropped", _c[0]["x"] == 1 and "factor" not in _c[0])
    chk("chains: validate 0 → none", not any(c["validate"] for c in _c))
    chk("chains: validate 1 → all", all(c["validate"] for c in build_chains(_w, 1.0)))
    _vid = lambda cs: {(c["benchmark_name"], c["loop_idx"], c["unmerge"]): c["validate"]
                       for c in cs}
    chk("chains: validation choice depends only on the row",
        _vid(build_chains(_w, 0.5)) == _vid(build_chains(list(reversed(_w)), 0.5)))
    from store_common import classify_compile_error, failure_cause
    chk("classify: pass budget", classify_compile_error(
        "fatal error: error in backend: UU budget: loop 3 in 'k' projected unmerge size 9")
        == "compile_budget")
    chk("classify: OOM", classify_compile_error("LLVM ERROR: out of memory") == "compile_oom")
    chk("classify: SIGKILL", [classify_compile_error(s) for s in (
        "clang++: error: unable to execute command: Killed",
        "make: *** [Makefile:50: main.o] Killed", "make: *** [Makefile:50: main.o] Error 137")]
        == ["compile_killed"] * 3)
    chk("classify: 'Killed' inside an identifier is not a kill",
        classify_compile_error("main.cu:3:1: error: use of undeclared identifier 'numKilled'")
        == "compile_failed")
    chk("classify: node / bundle errors are not the cell's", [classify_compile_error(s) for s in (
        "fatal error: error writing to /dev/shm/x.o: No space left on device",
        "clang++: error: unable to execute command: posix_spawn failed: No such file or "
        "directory", "main.cu:3:10: fatal error: '../include/x.h' file not found")]
        == ["compile_env"] * 3)
    chk("classify: assertion is a plain failure", classify_compile_error(
        "clang++: .../LoopUnroll.cpp:500: Assertion `x' failed.") == "compile_failed")
    # ptxas under the address-space cap (aop-cuda|3|1|4 at 12 GB, verbatim): the first
    # error line is clang's generic one, the cause is the ptxas line
    _ptx = ("clang++: warning: CUDA version is newer than the latest partially supported "
            "version 12.8 [-Wunknown-cuda-version]\n"
            "ptxas fatal   : Memory allocation failure\n"
            "ptxas fatal   : Ptx assembly aborted due to errors\n"
            "clang++: error: ptxas command failed with exit code 255 (use -v to see "
            "invocation)\nmake: *** [Makefile:50: main.o] Error 255\n")
    chk("cause: ptxas OOM", failure_cause(_ptx) == "ptxas fatal   : Memory allocation failure"
        and classify_compile_error(failure_cause(_ptx)) == "compile_oom")
    chk("cause: a source excerpt is not a tool line", failure_cause(
        "main.cu:9:5: warning: format specifies type 'int' [-Wformat]\n"
        "out of memory: %d\n    ^~\nmain.cu:12:1: error: expected ';'\n") == "")
    chk("cause: budget outranks the kill it led to", classify_compile_error(failure_cause(
        "fatal error: error in backend: UU budget: unmerge of loop 'l' cloned 3e7\n"
        "make: *** [Makefile:5: main.o] Killed\n")) == "compile_budget")
    chk("penalty class", [penalty_type(s) for s in ("compile_failed", "compile_timeout",
                                                    "compile_budget", "compile_oom")]
        == ["compile_failed", "compile_timeout", "compile_timeout", "compile_timeout"])
    # measure attempts: a GPU's no-output cells count once it measures again
    _mf, _pend = {}, {}
    _nf = lambda k, g: {"type": "measure_failed", "key": k, "status": "no_output",
                        "sidecar": {"gpu": g}}
    track_measure_attempts(_nf("a|0|1|1", 0), _pend, _mf)
    chk("attempts: no-output deferred", _mf == {} and _pend == {0: [("a|0|1|1", "no_output")]})
    track_measure_attempts({"type": "cell", "key": "a|0|1|2", "sidecar": {"gpu": 1,
                                                                          "run_ms": [1.0]}},
                           _pend, _mf)
    chk("attempts: another GPU's measurement does not count it", _mf == {})
    track_measure_attempts({"type": "measure_failed", "key": "a|0|1|3",
                            "status": "launch_mismatch",
                            "sidecar": {"gpu": 0, "run_ms": [1.0]}}, _pend, _mf)
    chk("attempts: counted once the same GPU measured; mismatch counts at once",
        _mf == {"a|0|1|1": {"attempts": 1, "status": "no_output"},
                "a|0|1|3": {"attempts": 1, "status": "launch_mismatch"}} and _pend == {})
    track_measure_attempts({"type": "cell", "key": "a|0|1|3", "sidecar": {}}, _pend, _mf)
    chk("attempts: a later reward clears the record", "a|0|1|3" not in _mf)
    track_measure_attempts({**_nf("a|0|1|5", 4), "status": "clock_unlocked"}, _pend, _mf)
    track_measure_attempts({"type": "cell", "key": "a|0|1|6",
                            "sidecar": {"gpu": 4, "run_ms": [1.0]}}, _pend, _mf)
    chk("attempts: an unlocked clock is deferred like no-output, keeps its status",
        _mf.get("a|0|1|5") == {"attempts": 1, "status": "clock_unlocked"})
    track_measure_attempts({"type": "measure_failed", "key": "a|0|1|4",
                            "status": "compile_killed"}, _pend, _mf)
    chk("attempts: node errors never count", "a|0|1|4" not in _mf)
    _dead = [track_measure_attempts(_nf(f"{'bcd'[i % 3]}|0|1|{i}", 2), _pend, _mf)
             for i in range(GPU_DEAD_CELLS)]
    chk("attempts: a GPU failing across benchmarks is reported",
        _dead[-1] == 2 and not any(x is not None for x in _dead[:-1]))
    _p3 = {}
    chk("attempts: ... and one benchmark's hanging cells are not",
        all(track_measure_attempts(_nf(f"e|0|1|{i}", 3), _p3, {}) is None
            for i in range(2 * GPU_DEAD_CELLS)) and len(_p3[3]) == 2 * GPU_DEAD_CELLS)
    # ncu scoring: launch-weighted mean per-launch time; short runs are mismatches
    _plan = {"A": {"w": 0.75, "inst": 100.0}, "B": {"w": 0.25, "inst": None}}
    _ok = lambda ms, runs: {"status": "ok", "ms": ms, "runs": runs, "inst": 150.0,
                            "exec_s": [3.0]}
    _it = lambda res: {"key": "a|0|1|2", "benchmark": "a", "baseline_ms": 2.0,
                       "ncu": res, "ncu_plan": _plan, "sidecar": {"gpu": 0}}
    _m = score_ncu(_it({"A": _ok(1.0, [1.0, 1.0]), "B": _ok(3.0, [3.0])}), 0.0)
    chk("ncu score: reward over the weighted parents, run_ms paired",
        _m["type"] == "cell" and abs(_m["reward"] - 0.25) < 1e-12
        and _m["sidecar"]["run_ms"] == [1.5] and _m["sidecar"]["ncu"]["A"]["inst_ratio"] == 1.5)
    _m = score_ncu(_it({"A": _ok(1.0, [1.0]), "B": {"status": "short", "exec_s": []}}), 0.0)
    chk("ncu score: a short parent is a launch mismatch, not cached",
        (_m["type"], _m["status"], _m["sidecar"]["run_ms"])
        == ("measure_failed", "launch_mismatch", []))
    _m = score_ncu(_it({"A": {"status": "clock_unlocked", "exec_s": []}}), 0.0)
    chk("ncu score: unlocked clock / a parent never reached",
        (_m["type"], _m["status"]) == ("measure_failed", "clock_unlocked")
        and score_ncu(_it({"A": _ok(1.0, [1.0])}), 0.0)["status"] == "no_output")
    chk("profiler default: ncu (locked base clock) on NVIDIA, rocprof (no lock) on AMD",
        [resolve_profiler(c, h) for c, h in (("auto", False), ("auto", True),
                                             ("nsys", False), ("nsys", True), ("ncu", False))]
        == ["ncu", "rocprof", "nsys", "rocprof", "ncu"])
    try:
        resolve_profiler("ncu", True)
        chk("profiler: ncu on AMD refused", False)
    except SystemExit:
        chk("profiler: ncu on AMD refused", True)
    chk("ncu command locks the base clock", "--clock-control base" in
        ncu_timer.command("./main", "k", "/t/l", 2, 20))
    # reward math matches collect_cells exactly
    chk("reward speedup", abs(compute_reward(100.0, 80.0, 0.0) - 0.2) < 1e-9)
    chk("reward clip -1", compute_reward(100.0, 500.0, 0.0) == -1.0)
    chk("reward deadzone", compute_reward(100.0, 99.5, 0.01) == 0.0)
    chk("reward baseline~0 safe", compute_reward(0.0, 0.0, 0.0) == 0.0)
    # structural_copy: small copied, large symlinked, data readable through symlink
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d) / "src", Path(d) / "dst"
        (src / "sub").mkdir(parents=True)
        (src / "main.cu").write_text("int main(){}")          # build input → copy
        (src / "Makefile").write_text("all:")                 # build input → copy
        big = src / "data.bin"
        big.write_bytes(b"x" * 4096)                          # > threshold → symlink
        (src / "sub" / "kernel.cuh").write_text("//h")        # build input → copy
        structural_copy(src, dst, threshold=1024)
        chk("copy source file", (dst / "main.cu").is_file()
            and not (dst / "main.cu").is_symlink())
        chk("copy Makefile", (dst / "Makefile").is_file()
            and not (dst / "Makefile").is_symlink())
        chk("symlink large data", (dst / "data.bin").is_symlink())
        chk("data readable via symlink", (dst / "data.bin").read_bytes() == b"x" * 4096)
        chk("recurse subdir header copied", (dst / "sub" / "kernel.cuh").is_file()
            and not (dst / "sub" / "kernel.cuh").is_symlink())

    # --- pre-compiled store: round-trip + the C1 write-through-symlink guard ---
    from store_common import write_cell, run_target_rel
    chk("run_target_rel", run_target_rel("./main a b") == "main"
        and run_target_rel("./sub/foo") == "sub/foo")
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        bench = d / "bench"
        (bench).mkdir()
        (bench / "Makefile").write_text("all:")
        (bench / "main").write_bytes(b"OLD" * 1000)          # 3000B > threshold → symlinked
        orig = (bench / "main").read_bytes()
        store = str(d / "store")
        prebuilt = d / "prebuilt_bin"
        prebuilt.write_bytes(b"NEWBINARY")
        key = cellkey("bench", 0, 1, 2)
        write_cell(store, key, prebuilt, "main", "bench", "fp123")
        chk("store: cell marked done", is_cell_done(store, key))
        bundle = d / "bundle"
        structural_copy(bench, bundle, threshold=1024)
        chk("store: committed binary symlinked in bundle", (bundle / "main").is_symlink())
        staged = stage_from_store(store, key, bundle)
        chk("store: stage_from_store ok", staged)
        chk("store: staged binary is a REAL file (not symlink)",
            (bundle / "main").is_file() and not (bundle / "main").is_symlink())
        chk("store: staged bytes are the stored binary",
            (bundle / "main").read_bytes() == b"NEWBINARY")
        chk("store C1: original benchmark binary UNCHANGED (no write-through-symlink)",
            (bench / "main").read_bytes() == orig)
        fp1 = benchmark_fingerprint(bench)
        chk("store: fingerprint stable", benchmark_fingerprint(bench) == fp1)
        (bench / "kern.cu").write_text("__global__ void k(){}")
        chk("store: fingerprint changes on source edit",
            benchmark_fingerprint(bench) != fp1)

    # --- manifest round-trip + toolchain stamp (the fail-safe verify components) ---
    from store_common import write_manifest
    with tempfile.TemporaryDirectory() as d:
        st = str(d)
        write_manifest(st, {"arch": "sm_x", "clang": "v1"}, {"b": "fp"}, ["b|0|1|2"])
        m = load_manifest(st)
        chk("manifest round-trip", m is not None and m["stamps"]["arch"] == "sm_x"
            and m["benchmarks"]["b"] == "fp" and "b|0|1|2" in m["cells"])
        chk("load_manifest(missing) → None", load_manifest(str(Path(d) / "nope")) is None)
    chk("toolchain_stamp carries arch", toolchain_stamp("sm_z")["arch"] == "sm_z")
    # set_post_features: a persist must never wipe on-disk post-feature vectors just
    # because the in-memory dict is empty after a normalizer-signature mismatch.
    from store_common import set_post_features
    pl = {"normalizer_sig": "old", "post_features": {"b|0": [1.0]}, "rewards": {}}
    set_post_features(pl, {}, "new")
    chk("postf: empty in-memory keeps the file's vectors", pl["post_features"] == {"b|0": [1.0]})
    chk("postf: ... and their own signature", pl["normalizer_sig"] == "old")
    set_post_features(pl, {"b|0": [2.0]}, "new")
    chk("postf: non-empty replaces under the new signature",
        pl["post_features"] == {"b|0": [2.0]} and pl["normalizer_sig"] == "new")
    pl2 = {"rewards": {}}
    set_post_features(pl2, {}, "new")
    chk("postf: empty + nothing on disk writes {} under the signature",
        pl2["post_features"] == {} and pl2["normalizer_sig"] == "new")
    pl3 = {"post_features": {"b|0": [1.0]}}
    set_post_features(pl3, {}, "new")
    chk("postf: unsigned vectors on disk stay unsigned", "normalizer_sig" not in pl3 and pl3["post_features"])

    print(f"\n  {'ALL PASS' if ok else 'FAILURES ABOVE'}")
    return 0 if ok else 1


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint_dir", type=Path, nargs="?")
    p.add_argument("--split", default="test", help="comma list: test,val,train")
    p.add_argument("--n-runs", type=int, help="REQUIRED (unless --selftest); match the run "
                                              "(--profiler ncu: 2, i.e. 2 x 20 launches; "
                                              "baselines take at least 2)")
    p.add_argument("--exec", dest="n_exec_per_gpu", type=int, default=1,
                   help="exec processes PER GPU (default 1; one nsys profile/GPU)")
    p.add_argument("--compile", dest="n_compile_per_gpu", type=int, default=2,
                   help="compile processes per GPU (default 2)")
    p.add_argument("--parse", dest="n_parse_per_gpu", type=int, default=3,
                   help="parse processes per GPU (default 3)")
    p.add_argument("--postfeat", dest="n_postfeat", type=int, default=2,
                   help="TOTAL post-feature processes (default 2)")
    p.add_argument("--gpus", type=int, default=1, help="#GPUs on this node")
    p.add_argument("--gpu-ids", default=None,
                   help="comma list of the GPUs to use, as CUDA_VISIBLE_DEVICES numbers "
                        "them (set CUDA_DEVICE_ORDER=PCI_BUS_ID for nvidia-smi's order); "
                        "overrides --gpus.  Default 0..--gpus-1")
    p.add_argument("--precompiled-store", default=None,
                   help="dir from precompile.py: STAGE its binaries instead of building "
                        "(fail-safe — verified by toolchain stamp + per-benchmark source "
                        "fingerprint; any mismatch falls back to building)")
    p.add_argument("--shm-root", default="/dev/shm/uu_pipeline")
    p.add_argument("--tmp-root", default=None, help="reports/nsys temp (default CKPT/pipe_tmp)")
    p.add_argument("--copy-threshold", type=int, default=1_000_000,
                   help="files > this many bytes are symlinked, not copied (default 1MB)")
    p.add_argument("--q12-max", type=int, default=0, help="compile→exec queue cap (0=auto)")
    p.add_argument("--q23-max", type=int, default=0, help="exec→parse queue cap (0=auto)")
    p.add_argument("--mem-log-interval", type=float, default=15.0)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--no-post-features", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--baselines-only", action="store_true",
                   help="measure the baseline of EVERY benchmark in --split that lacks one "
                        "(not only those with missing cells), write baseline_cache.json, "
                        "and exit without collecting cells — the first step of a collection")
    p.add_argument("--baselines", choices=("never", "auto"), default="never",
                   help="cell runs: 'never' (default) measures no baseline and refuses to "
                        "start while a benchmark with cells to collect has neither a "
                        "baseline nor a recorded failure (run --baselines-only first); "
                        "'auto' measures the missing ones first")
    p.add_argument("--retry-failed", action="store_true",
                   help=f"retry benchmarks whose baseline failed {MAX_MEASURE_ATTEMPTS} times "
                        "(baseline_cache.json 'failed'; with --baselines-only / auto) and "
                        f"cells whose runs failed {MAX_MEASURE_ATTEMPTS} times "
                        "(migration.measure_failed); both are skipped by default")
    p.add_argument("--read-only", action="store_true",
                   help="VALIDATION: run the real compile->nsys->parse pipeline but NEVER "
                        "write reward_cache.json or baseline_cache.json (baselines are not "
                        "measured either — cells whose benchmark has no baseline are skipped). "
                        "All scratch + the memlog go OUTSIDE the checkpoint dir. Pair with "
                        "--limit for a quick end-to-end check.")
    p.add_argument("--limit", type=int, default=0,
                   help="cap the run to N (loop, unmerge) rows, round-robin across "
                        "benchmarks, each with all its factors (so row stops and their "
                        "validation are exercised) — for a fast validation sample; 0 = no cap")
    p.add_argument("--selftest", action="store_true")
    # must match the training run / existing cache
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--compile-failure-penalty", type=float)
    p.add_argument("--reward-deadzone", type=float)
    p.add_argument("--compile-timeout-penalty", type=float, default=-1.0)
    p.add_argument("--compile-mem-gb", type=float, default=24,
                   help="address-space cap per compile in GB (UU_COMPILE_MEM_GB; default "
                        "24, 0 = none): an allocation failure under it is the cell's own "
                        "size (timeout penalty, stops its row); a SIGKILL is the node's "
                        "OOM killer and is not cached")
    p.add_argument("--compile-timeout", type=int, default=300,
                   help="per-cell compile budget in seconds (default 300); a hit is the "
                        "compile_timeout penalty.  MIRROR: collect_cells --compile-timeout")
    p.add_argument("--no-infer", dest="infer_after_failure", action="store_false",
                   help="compile EVERY cell; by default a (loop, unmerge) row stops at its "
                        "first failed compile and the higher factors get the same penalty, "
                        "recorded in migration.inferred_keys (see build_chains)")
    p.add_argument("--no-skip-identical", dest="skip_identical", action="store_false",
                   help="run every compiled cell on the GPU; by default a cell whose device "
                        "SASS hashes equal to its baseline's gets reward 0.0 without a run "
                        "(status 'identical' in cells.jsonl; needs the baseline's "
                        "device_hash, recorded by measure_baselines)")
    p.add_argument("--validate-inferred", type=float, default=0.10, metavar="FRAC",
                   help="fraction of rows that also compile the factor after their stop, "
                        "to check the inference during the run (default 0.10)")
    p.add_argument("--arch", default=None, help="default: TARGET_ARCH (must agree with it: "
                                                "hecbench picks the compile driver and the "
                                                "profiler from TARGET_ARCH at import)")
    p.add_argument("--nsys-timeout", type=int, default=300,
                   help="per profiled run (nsys, rocprof or ncu) hard timeout, seconds")
    p.add_argument("--profiler", choices=("auto", "nsys", "ncu"), default="auto",
                   help="timer: auto (default) = ncu on NVIDIA (the loop's parent kernels "
                        "at the locked base clock; see the module doc) and rocprof on AMD "
                        "(no clock lock there); nsys = the previous NVIDIA timer (all "
                        "launches at dynamic clocks).  Stamped into both caches, which must "
                        "agree")
    p.add_argument("--ncu-launches", type=int, default=ncu_timer.COUNT,
                   help=f"ncu: launches profiled per run after the first {ncu_timer.SKIP} "
                        f"(default {ncu_timer.COUNT}; 0 = all, the application is not "
                        "killed)")
    p.add_argument("--ncu-clock-mhz", type=float, default=0.0,
                   help="ncu: optional sanity bound, e.g. 765 (the A100-PCIE-40GB base "
                        "clock): exit if a baseline kernel was locked at another clock "
                        "(±1%%).  Without it the clock is only checked for agreement: within "
                        "a run, and a cell with its baseline")
    p.add_argument("--hecbench-src", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.selftest:
        return _selftest()
    for req in ("checkpoint_dir", "n_runs", "compile_failure_penalty", "reward_deadzone"):
        if getattr(args, req) is None:
            sys.exit(f"--{req.replace('_','-')} is required (see --help)")

    import hashlib
    import torch.multiprocessing as mp
    from collect_cells import valid_cells, missing_cells, save_cache
    from hecbench import ARCH, HECBENCH_SRC, IS_HIP, discover_benchmarks, IR2VEC_VOCAB
    from train import (precheck_benchmarks, split_benchmarks, build_loop_assignments,
                       measure_baselines, baseline_failure_current)
    if args.arch is None:
        args.arch = ARCH
    # hecbench derives IS_HIP (compile driver, profiler, -hip/-cuda suffix) from
    # TARGET_ARCH at import; an --arch of the other family would compile one way and
    # profile the other.  Refuse rather than mix.
    if str(args.arch).startswith("gfx") != IS_HIP:
        sys.exit(f"--arch {args.arch} disagrees with TARGET_ARCH={ARCH!r} "
                 f"(IS_HIP={IS_HIP}); export TARGET_ARCH for the intended target instead")
    args.profiler = resolve_profiler(args.profiler, IS_HIP)
    if args.profiler == "ncu" and args.n_exec_per_gpu > 1:
        sys.exit("--profiler ncu needs --exec 1: each exec has its own TMPDIR, hence its "
                 "own ncu lock, so two on one GPU would profile it at the same time")
    # What the rewards and baselines are timed with; None = nsys / rocprof, as before.
    timer = (ncu_timer.timer_stamp("ncu", args.ncu_launches)
             if args.profiler == "ncu" else None)

    if not args.no_post_features and (not IR2VEC_VOCAB or not Path(IR2VEC_VOCAB).exists()):
        sys.exit(f"IR2VEC_VOCAB unset/missing ({IR2VEC_VOCAB!r}) but post-features on; "
                 f"set it or pass --no-post-features")

    ro = args.read_only
    ckpt = args.checkpoint_dir
    elig = ckpt / "eligible_benchmarks.json"
    if not elig.exists():
        sys.exit(f"missing {elig}")
    # Per-checkpoint scratch so two jobs on one node (different run dirs) can't clobber
    # each other's shm bundles when either does its startup wipe.  Read-only validation
    # gets its OWN "__val" namespace so it can never wipe a real run's live shm bundles.
    shm_root = Path(args.shm_root) / (ckpt.name + ("__val" if ro else ""))
    src = Path(args.hecbench_src) if args.hecbench_src else HECBENCH_SRC

    all_b, _lc, loop_records_map, normalizer = precheck_benchmarks(
        discover_benchmarks(src), elig, skip=True, strict=True)
    train_b, val_b, test_b = split_benchmarks(all_b, args.val_ratio, args.test_ratio,
                                              args.split_seed)
    pools = {"train": train_b, "val": val_b, "test": test_b}
    wanted = [s.strip() for s in args.split.split(",") if s.strip()]
    if any(s not in pools for s in wanted):
        sys.exit(f"unknown split(s) in {wanted}")
    benches = [b for s in wanted for b in pools[s]]
    assignments = build_loop_assignments(benches, loop_records_map)

    # --- worker counts (per-GPU × #GPUs); gate postfeat work on having procs ---
    # Physical GPU ids (CUDA_VISIBLE_DEVICES values) for rank 0..G-1.
    gpu_ids = ([int(x) for x in args.gpu_ids.split(",")] if args.gpu_ids
               else list(range(max(args.gpus, 1))))
    G = len(gpu_ids)
    n_exec = args.n_exec_per_gpu * G
    n_compile = args.n_compile_per_gpu * G
    n_parse = args.n_parse_per_gpu * G
    n_postfeat = 0 if args.no_post_features else max(args.n_postfeat, 0)
    if (not args.no_post_features) and n_postfeat == 0:
        log.warning("--postfeat 0 with post-features enabled → post-features NOT extracted")
    if args.n_exec_per_gpu > 1:
        log.warning("--exec %d (>1 per GPU): multiple concurrent profiled runs on ONE GPU "
                    "CORRUPT the timing measurements (GPU contention). Use --exec 1 unless "
                    "the GPUs are genuinely distinct/partitioned.", args.n_exec_per_gpu)

    # --- existing cache (resume truth) ---
    rc_file = ckpt / "reward_cache.json"
    rewards, postf, measure_failed = {}, {}, {}
    norm_sig = hashlib.md5(json.dumps(normalizer.state_dict(),
                                      sort_keys=True).encode()).hexdigest()[:12]
    if rc_file.exists():
        data = json.loads(rc_file.read_text())
        if data.get("rewards") and data.get("timer") != timer:
            sys.exit(f"{rc_file} holds rewards timed with {data.get('timer') or 'nsys'}, "
                     f"this run times with {timer or 'nsys'}: they must not mix (use "
                     f"another checkpoint dir, or the cache's --profiler settings)")
        rewards = {k: float(v) for k, v in data.get("rewards", {}).items()}
        if data.get("normalizer_sig") == norm_sig:
            postf = data.get("post_features", {})
        measure_failed = (data.get("migration") or {}).get("measure_failed", {})
    todo = missing_cells(assignments, rewards,
                         None if args.no_post_features else postf)
    n_missing = sum(len(v) for v in todo.values())

    # --- baselines FIRST (a cell's baseline_ms must exist before its reward) ---
    # Baselines are a separate step (--baselines-only) unless --baselines auto: a
    # multi-job collection otherwise re-measures, at the start of every job, each
    # benchmark whose baseline failed in the previous one.
    bl_file = ckpt / "baseline_cache.json"
    baseline_cache, bl_failed = {}, {}
    if bl_file.exists():
        _bl = json.loads(bl_file.read_text())
        if _bl.get("baselines") and ((_bl.get("arch"), _bl.get("n_runs"), _bl.get("timer"))
                                     != (args.arch, args.n_runs, timer)):
            sys.exit(f"{bl_file} was measured with arch={_bl.get('arch')} n_runs="
                     f"{_bl.get('n_runs')} timer={_bl.get('timer') or 'nsys'}; this run has "
                     f"arch={args.arch} n_runs={args.n_runs} timer={timer or 'nsys'}")
        baseline_cache, bl_failed = _bl.get("baselines", {}), _bl.get("failed", {})

    def _check_clock():
        """--ncu-clock-mhz: exit if a baseline kernel of this split was locked at
        another clock (also on --dry-run, for the baselines already on disk)."""
        if timer and args.ncu_clock_mhz:
            off = ncu_timer.off_clock({b.name: baseline_cache[b.name] for b in benches
                                       if b.name in baseline_cache}, args.ncu_clock_mhz)
            if off:
                sys.exit(f"{len(off)} baseline kernel(s) locked at a clock other than "
                         f"--ncu-clock-mhz {args.ncu_clock_mhz:g} (±1%): "
                         + ", ".join(off[:20]))
    _check_clock()
    todo_benches = {a["benchmark_name"] for a in assignments
                    if f"{a['benchmark_name']}|{a['loop_idx']}" in todo}
    if args.baselines_only:
        need_bl = [b for b in benches if b.name not in baseline_cache]
    else:
        need_bl = [b for b in benches if b.name in todo_benches and b.name not in baseline_cache]
    # READ-ONLY: never measure baselines (that path WRITES baseline_cache.json). Cells whose
    # benchmark has no baseline are dropped below (base_ms<=0) rather than measured.
    if ro and need_bl:
        log.warning("READ-ONLY: %d benchmark(s) have no baseline — their cells are SKIPPED "
                    "(baselines are never measured or written in read-only mode)", len(need_bl))
    if ro:
        need_bl = []
    elif args.baselines == "never" and not args.baselines_only and need_bl:
        live = toolchain_stamp(args.arch)
        unknown = [b.name for b in need_bl
                   if not baseline_failure_current(bl_failed.get(b.name), b, args.arch, live)]
        if unknown:
            msg = (f"{len(unknown)} benchmark(s) with cells to collect have no baseline in "
                   f"{bl_file}: run --baselines-only first (or pass --baselines auto): "
                   + ", ".join(unknown))
            if not args.dry_run:
                sys.exit(msg)
            log.warning(msg)
        need_bl = []

    # In read-only mode keep ALL writes out of the checkpoint dir: scratch (reports/nsys temp)
    # goes to a system-temp validation dir unless the user points --tmp-root at fast local disk.
    if args.tmp_root:
        tmp_root = Path(args.tmp_root)
    elif ro:
        import tempfile as _tf
        tmp_root = Path(_tf.gettempdir()) / f"uu_pipe_val__{ckpt.name}"
    else:
        tmp_root = ckpt / "pipe_tmp"
    if ro:
        log.warning("=== READ-ONLY VALIDATION ===  reward_cache.json / baseline_cache.json "
                    "will NOT be written.  scratch=%s  shm=%s", tmp_root, shm_root)

    # --- build the per-cell work list + per-loop postfeat list ---
    # Scope (kernel_filter, baseline_ms) is attached LATER, after baselines are measured.
    work_items, post_items = [], []
    for a in assignments:
        bench, li = a["benchmark_name"], a["loop_idx"]
        key = f"{bench}|{li}"
        if key not in todo:
            continue
        for (u, f) in todo[key]:
            work_items.append({"benchmark_name": bench, "benchmark_path": a["benchmark_path"],
                               "loop_idx": li, "filename": a["filename"],
                               "triple": a["triple"], "unmerge": u, "factor": f})
        has_unmerge = any(u == 1 for u, _ in valid_cells(a["pre_features_raw"]))
        if n_postfeat > 0 and has_unmerge and key not in postf:
            post_items.append({"benchmark_name": bench, "benchmark_path": a["benchmark_path"],
                               "loop_idx": li, "filename": a["filename"], "triple": a["triple"],
                               "pre_features_raw": a["pre_features_raw"],
                               "kernel_parents": a.get("kernel_parents", [])})

    log.info("cells: %d missing of %d valid | postfeat loops: %d | baselines to measure: %d",
             n_missing, sum(len(valid_cells(a["pre_features_raw"])) for a in assignments),
             len(post_items), len(need_bl))
    log.info("workers: exec=%d compile=%d parse=%d postfeat=%d  (per-GPU %d/%d/%d × %d GPUs)"
             "  target=%s profiler=%s",
             n_exec, n_compile, n_parse, n_postfeat,
             args.n_exec_per_gpu, args.n_compile_per_gpu, args.n_parse_per_gpu, G,
             args.arch, args.profiler)

    if args.dry_run:
        log.info("dry run — nothing measured.")
        return 0
    if not n_missing and not post_items and not args.baselines_only:
        log.info("nothing to do.")
        return 0

    if need_bl:
        log.info("measuring %d missing baselines first...", len(need_bl))
        # MIRROR: collect_cells — the shared baseline path is itself a dynamic queue
        # over the node's GPUs (one worker per GPU, rank == gpu_id).
        baseline_cache = measure_baselines(
            need_bl, loop_records_map=loop_records_map, arch=args.arch,
            n_runs=args.n_runs, nsys_timeout=args.nsys_timeout,
            tmp_dir=tmp_root, gpu_id=gpu_ids[0], gpu_ids=gpu_ids, cache_file=bl_file,
            num_workers=G,
            skip_failed_after=0 if args.retry_failed else MAX_MEASURE_ATTEMPTS, timer=timer)
        bl_failed = json.loads(bl_file.read_text()).get("failed", {})
    _check_clock()                         # the baselines just measured, if any
    if args.baselines_only:
        have = sum(1 for b in benches if b.name in baseline_cache)
        log.info("--baselines-only: %d/%d benchmarks of %s have a baseline in %s, %d "
                 "recorded as failed — done", have, len(benches), args.split, bl_file,
                 sum(1 for b in benches if b.name in bl_failed and b.name not in baseline_cache))
        return 0

    # Attach scope per LOOP (resolved once, now that baselines exist) and DROP cells
    # whose benchmark still has no baseline — they cannot be scored, so leave them
    # missing to retry on a later run rather than divide by a bogus 0 baseline.
    # ncu: the scope is the loop's launched parents (plans[k]), or why there is none.
    scope, plans = {}, {}
    for a in assignments:
        k = (a["benchmark_name"], a["loop_idx"])
        if k in scope:
            continue
        if timer:
            plan, base = ncu_timer.scope(a.get("kernel_parents", []),
                                         baseline_cache.get(a["benchmark_name"]) or {})
            plans[k], scope[k] = (plan, (None, base)) if plan else (base, (None, 0.0))
        else:
            scope[k] = resolve_kernel_scope(a.get("kernel_parents", []),
                                            baseline_cache, a["benchmark_name"])
    stale = stale_baselines({w["benchmark_name"]: w["benchmark_path"] for w in work_items},
                            baseline_cache, args.arch)
    if stale:
        sys.exit("%d baseline(s) cannot be scored against — re-measure them (move "
                 "baseline_cache.json aside, or delete these entries and re-run):\n  %s"
                 % (len(stale), "\n  ".join(f"{b}: {r}" for b, r in stale.items())))
    # A loop whose kernel parents all have 0 launches in the baseline (exact, by
    # mangled name) can only measure noise: left missing, like a missing baseline.
    parents = {(a["benchmark_name"], a["loop_idx"]): a.get("kernel_parents") or []
               for a in assignments}
    kept, dropped, unlaunched, untimed, give_up, skipped = [], set(), set(), {}, 0, []
    for w in work_items:
        b, k = w["benchmark_name"], (w["benchmark_name"], w["loop_idx"])
        kf, base_ms = scope[k]
        key = cellkey(b, w["loop_idx"], w["unmerge"], w["factor"])
        if b not in baseline_cache or (timer is None and base_ms <= 0.0):
            dropped.add(b)
            continue
        bl = baseline_cache.get(b, {}).get("launches") or {}
        if parents[k] and all(bl.get("parents", {}).get(m) == 0 for m in parents[k]):
            unlaunched.add(k)
            skipped.append((key, "never_launched"))
            continue
        if isinstance(plans.get(k), str):         # ncu cannot time this loop's kernel
            untimed[k] = plans[k]
            skipped.append((key, plans[k]))
            continue
        if (measure_failed.get(key, {}).get("attempts", 0) >= MAX_MEASURE_ATTEMPTS
                and not args.retry_failed):
            give_up += 1
            continue
        w["kernel_filter"], w["baseline_ms"] = kf, base_ms
        w["baseline_launches"] = (None if not (bl.get("stable") or {}).get(kf or "") else
                                  bl.get("per_kernel", {}).get(kf) if kf else bl.get("total"))
        w["baseline_hash"] = baseline_cache.get(b, {}).get("device_hash")
        w["ncu_plan"] = plans.get(k)
        kept.append(w)
    n_dropped = len(work_items) - len(kept)
    work_items = kept
    if dropped:
        log.warning("cells of %d benchmark(s) skipped — no baseline: %s", len(dropped),
                    ", ".join(f"{b} ({bl_failed[b]['reason']})" if b in bl_failed else b
                              for b in sorted(dropped)))
    if unlaunched:
        log.info("%d loop(s) skipped — their kernel never launches in the baseline run: %s",
                 len(unlaunched), ", ".join(f"{b}|{li}" for b, li in sorted(unlaunched)))
    if untimed:
        log.warning("%d loop(s) skipped — ncu cannot time their kernel: %s", len(untimed),
                    ", ".join(f"{b}|{li} ({why})" for (b, li), why in sorted(untimed.items())))
    if give_up:
        log.info("%d cell(s) skipped — their runs failed %d times (--retry-failed retries "
                 "them)", give_up, MAX_MEASURE_ATTEMPTS)
    if n_dropped:
        log.info("%d of %d missing cells skipped in total", n_dropped, n_dropped + len(kept))
    # FASTEST-FIRST (MIRROR: collect_cells' queue order): cells of the quickest-running
    # benchmarks go through the pipeline first, so if the window runs out the slow-kernel
    # tail is what is left unmeasured.  Stable sort: loop/cell order within a benchmark
    # is unchanged.  The store-hits-first sort below is also stable, so hits keep this
    # order among themselves.
    work_items.sort(key=lambda w: (baseline_cache.get(w["benchmark_name"]) or {})
                    .get("total_ms", float("inf")))
    n_missing = len(work_items)          # completion target: one terminal result per item

    # --- optional --limit: validate a small, benchmark-spread sample of WHOLE rows ---
    if args.limit and args.limit > 0:
        rows = {(c["benchmark_name"], c["loop_idx"], c["unmerge"])
                for c in _limit_by_benchmark(build_chains(work_items, 0.0), args.limit)}
        work_items = [w for w in work_items
                      if (w["benchmark_name"], w["loop_idx"], w["unmerge"]) in rows]
        post_items = _limit_by_benchmark(post_items, args.limit)
        n_missing = len(work_items)
        log.info("--limit %d → %d rows, %d cells across %d benchmarks + %d postfeat loops",
                 args.limit, len(rows), len(work_items),
                 len({w["benchmark_name"] for w in work_items}), len(post_items))

    # --- pre-compiled store (optional): verify fail-safe, then mark stage-able cells ---
    prebuilt, store = set(), None
    if args.precompiled_store:
        store = str(args.precompiled_store)
        man = load_manifest(store)
        if man is None:
            log.warning("--precompiled-store %s: no manifest — ignoring (build all)", store)
            store = None
        else:
            live = toolchain_stamp(args.arch)
            ms = man.get("stamps", {})
            if any(ms.get(k) != live[k] for k in ("arch", "clang", "clang_build")):
                log.warning("store stamp MISMATCH (arch %s vs %s / clang or its build "
                            "differs) — IGNORING store, building instead",
                            ms.get("arch"), live["arch"])
                store = None
            else:
                in_store = set(man.get("cells", []))
                man_fps, fp_live = man.get("benchmarks", {}), {}
                for w in work_items:
                    k = cellkey(w["benchmark_name"], w["loop_idx"], w["unmerge"], w["factor"])
                    if k not in in_store:
                        continue
                    b = w["benchmark_name"]
                    if b not in fp_live:
                        fp_live[b] = benchmark_fingerprint(Path(w["benchmark_path"]))
                    if man_fps.get(b) == fp_live[b] and is_cell_done(store, k):
                        prebuilt.add(k)
                log.info("pre-compiled store: %d of %d cells usable (staged, not built)",
                         len(prebuilt), len(work_items))
                # W1: feed store-hits FIRST so exec warms immediately, builds overlap behind.
                work_items.sort(key=lambda w: 0 if cellkey(
                    w["benchmark_name"], w["loop_idx"], w["unmerge"], w["factor"]) in prebuilt
                    else 1)

    cfg = {
        "arch": args.arch, "n_runs": args.n_runs, "nsys_timeout": args.nsys_timeout,
        "reward_deadzone": args.reward_deadzone,
        "compile_failure_penalty": args.compile_failure_penalty,
        "compile_timeout_penalty": args.compile_timeout_penalty,
        "compile_timeout": args.compile_timeout, "is_hip": IS_HIP,
        "shm_root": str(shm_root), "tmp_root": str(tmp_root),
        "copy_threshold": args.copy_threshold,
        "normalizer_state": normalizer.state_dict(),
        "prebuilt": prebuilt, "store": store,
        "infer_after_failure": args.infer_after_failure,
        "skip_identical": args.skip_identical,
        "compile_mem_gb": args.compile_mem_gb,
    }
    chains = build_chains(work_items, args.validate_inferred if args.infer_after_failure else 0.0)
    log.info("compile chains: %d (loop, unmerge) rows for %d cells; stop-at-first-failure %s, "
             "validating %d rows", len(chains), len(work_items),
             "ON" if args.infer_after_failure else "OFF", sum(c["validate"] for c in chains))
    # fresh scratch (crash orphans from a prior run are garbage — resume is the cache)
    for r in (shm_root, tmp_root):
        shutil.rmtree(r, ignore_errors=True)
        r.mkdir(parents=True, exist_ok=True)

    q12_max = args.q12_max or max(2 * n_exec, 8)
    q23_max = args.q23_max or max(2 * n_parse, 8)
    work_q = mp.Queue(maxsize=max(4 * n_compile, 16))
    q12 = mp.Queue(maxsize=q12_max)
    q23 = mp.Queue(maxsize=q23_max)
    post_q = mp.Queue(maxsize=max(2 * n_postfeat, 8)) if n_postfeat else mp.Queue()
    result_q = mp.Queue()                       # UNBOUNDED — no producer ever blocks here

    compile_procs = [mp.Process(target=compile_worker, args=(r, work_q, q12, result_q, cfg),
                                daemon=True) for r in range(n_compile)]
    exec_procs = [mp.Process(target=exec_worker,
                             args=(r, gpu_ids[r % G], q12, q23, result_q, cfg),
                             daemon=True) for r in range(n_exec)]
    parse_procs = [mp.Process(target=parse_worker, args=(r, q23, result_q, cfg),
                              daemon=True) for r in range(n_parse)]
    postfeat_procs = [mp.Process(target=postfeat_worker, args=(r, post_q, result_q, cfg),
                                 daemon=True) for r in range(n_postfeat)]
    all_procs = compile_procs + exec_procs + parse_procs + postfeat_procs
    for p in all_procs:
        p.start()

    # Separate feeders so a full post_q can't stall the work feeder (GPU-critical path).
    threading.Thread(target=feed_queue, args=(chains, work_q, n_compile),
                     daemon=True).start()
    threading.Thread(target=feed_queue, args=(post_items, post_q, n_postfeat),
                     daemon=True).start()

    counters = {"cells_done": 0, "n_missing": n_missing}
    stop_mem = threading.Event()
    # read-only keeps the memlog out of the checkpoint dir too (scratch, not a result).
    memlog = (tmp_root if ro else ckpt) / "pipeline_memlog.jsonl"
    mem_t = threading.Thread(target=mem_monitor,
                             args=(stop_mem, memlog,
                                   {"work": work_q, "q12": q12, "q23": q23,
                                    "post": post_q, "result": result_q},
                                   cfg, counters, args.mem_log_interval), daemon=True)
    mem_t.start()

    failure_keys, timeout_keys, inferred_keys = set(), set(), set()
    exec_left, parse_left, postfeat_left = n_exec, n_parse, n_postfeat
    q12_stops_sent = q23_stops_sent = False
    by_status = {}
    t0 = time.time()
    GET_TIMEOUT = 120          # periodic liveness check when idle; benign if a cell > this

    def _persist():
        if ro:
            return                         # read-only: NEVER write reward_cache.json
        save_cache(rc_file, rewards, postf, norm_sig, failure_keys,
                   args.compile_failure_penalty, args.reward_deadzone,
                   timeout_keys=timeout_keys, inferred_keys=inferred_keys,
                   timeout_penalty=args.compile_timeout_penalty,
                   measure_failed=measure_failed, timer=timer)

    # One JSON line per terminal cell (status, error signature, compile/exec seconds,
    # per-run kernel ms and launch counts, validation flag): the data for sizing the
    # pools, the run-to-run spread, compile-time monotonicity and the inference check.
    sidecar = open((tmp_root if ro else ckpt) / "cells.jsonl", "a")
    for k, why in skipped:                  # never-launched / untimed loops: listed only
        sidecar.write(json.dumps({"key": k, "type": "skipped", "status": why,
                                  "ts": round(time.time(), 1)}) + "\n")
    validation = {"compiled": 0, "failed": 0}
    no_output = {}                          # gpu -> its no-output cells since it last measured
    gpu_dead = None

    # Drain results. Terminal per cell: cell/compile_failed/compile_timeout/measure_failed.
    # When every cell is terminal the pipeline is drained (Q12/Q23 empty) → inject STOPs.
    while parse_left > 0 or exec_left > 0 or postfeat_left > 0 or not q12_stops_sent:
        try:
            msg = result_q.get(timeout=GET_TIMEOUT)
        except _queue.Empty:
            # Liveness/stall detection so a partial death cannot hang the run forever.
            if postfeat_left > 0 and not any(p.is_alive() for p in postfeat_procs):
                log.warning("postfeat workers gone — %d loop(s) left unextracted", postfeat_left)
                postfeat_left = 0
            cell_alive = any(p.is_alive() for p in compile_procs + exec_procs + parse_procs)
            if counters["cells_done"] < n_missing and not cell_alive:
                log.error("cell stages all dead at %d/%d cells — stopping (re-run to resume)",
                          counters["cells_done"], n_missing)
                break
            if not any(p.is_alive() for p in all_procs):
                log.error("all workers dead; stopping")
                break
            continue
        t = msg.get("type")
        if t in ("cell", "compile_failed", "compile_timeout", "measure_failed"):
            if ro:
                # Validation has no cache write, so the per-cell value is only
                # visible here; print it so it can be compared against the
                # real run's reward_cache.json by key.
                log.info("READ-ONLY %-16s %-40s reward=%s", t, msg.get("key"),
                         "n/a" if t == "measure_failed" else f"{msg['reward']:+.4f}")
            side = msg.get("sidecar") or {}
            gpu_dead = track_measure_attempts(msg, no_output, measure_failed)
            if t != "measure_failed":                 # measure_failed is NOT cached
                rewards[msg["key"]] = msg["reward"]
                if t in ("compile_failed", "compile_timeout"):
                    # failure_keys keeps BOTH penalties (train --offline tells them
                    # apart by value); timeout_keys / inferred_keys say which is which.
                    failure_keys.add(msg["key"])
                    if t == "compile_timeout":
                        timeout_keys.add(msg["key"])
                    if msg.get("inferred"):
                        inferred_keys.add(msg["key"])
            if msg.get("validation") or side.get("validation"):
                validation["failed" if t in ("compile_failed", "compile_timeout")
                           else "compiled"] += 1
            sidecar.write(json.dumps({
                "key": msg["key"], "type": t, "reward": msg.get("reward"),
                "status": msg.get("status") or side.get("status"),
                "inferred": bool(msg.get("inferred")),
                "validation": bool(msg.get("validation") or side.get("validation")),
                "error": msg.get("error"), "compile_s": msg.get("compile_s"),
                **{k: v for k, v in side.items() if k not in ("status", "validation")},
                "ts": round(time.time(), 1)}) + "\n")
            sidecar.flush()
            counters["cells_done"] += 1
            st = ("inferred_" if msg.get("inferred") else "") + (msg.get("status") or t)
            by_status[st] = by_status.get(st, 0) + 1
            if gpu_dead is not None:
                log.error("GPU %s: %d cells in a row without profiler output, over %d "
                          "benchmarks — the GPU or the profiler is failing, not the cells. "
                          "Stopping; their attempts are not counted (re-run on a healthy "
                          "node).", gpu_dead, len(no_output[gpu_dead]),
                          len({k.split("|")[0] for k, _ in no_output[gpu_dead]}))
                break
            if counters["cells_done"] % args.save_every == 0:
                _persist()
                rate = counters["cells_done"] / max(time.time() - t0, 1e-9) * 3600
                log.info("progress %d/%d (%.0f/h, eta %.1fh) %s",
                         counters["cells_done"], n_missing, rate,
                         (n_missing - counters["cells_done"]) / max(rate, 1e-9), by_status)
        elif t == "postfeat":
            postf[f"{msg['benchmark']}|{msg['loop_idx']}"] = msg["features"]
        elif t == "compile_done":
            pass                                       # compile pool drains via work_q STOPs
        elif t == "exec_done":
            exec_left -= 1
        elif t == "parse_done":
            parse_left -= 1
        elif t == "postfeat_done":
            postfeat_left -= 1

        # All cells terminal → queues are empty → wake exec/parse to exit.  Use the
        # timeout-aware injector so a dead-consumer + full-queue can't hang main.
        if not q12_stops_sent and counters["cells_done"] >= n_missing:
            _inject_stops(q12, n_exec, exec_procs)
            q12_stops_sent = True
        if q12_stops_sent and not q23_stops_sent and exec_left == 0:
            _inject_stops(q23, n_parse, parse_procs)
            q23_stops_sent = True

    stop_mem.set()
    _persist()
    sidecar.close()
    if validation["compiled"] or validation["failed"]:
        log.info("inference check: %d validation compile(s) after a stop — %d failed as "
                 "inferred, %d COMPILED (inference contradicted; those rows were compiled "
                 "for real)", sum(validation.values()), validation["failed"],
                 validation["compiled"])
    if by_status.get("launch_mismatch"):
        log.warning("%d cell(s) launched a different number of kernels than their baseline "
                    "(crash / early exit?) — not cached; launches in cells.jsonl",
                    by_status["launch_mismatch"])
    if gpu_dead is not None:
        # Stopped early, nothing left to drain.  SIGINT, not terminate(): the
        # KeyboardInterrupt lets run_hard_timeout / _make kill the profiler + benchmark
        # (or compile) tree they started; a SIGTERM'd worker would orphan it on the GPU.
        for p in all_procs:
            if p.is_alive():
                os.kill(p.pid, signal.SIGINT)
    for p in all_procs:
        p.join(timeout=30)
        if gpu_dead is not None and p.is_alive():
            p.terminate()
    log.info("DONE  cells_done=%d/%d  status=%s  postfeat=%d  wall=%.1fh  memlog=%s",
             counters["cells_done"], n_missing, by_status, len(postf),
             (time.time() - t0) / 3600, memlog)
    left = missing_cells(assignments, rewards,
                         None if args.no_post_features else postf)
    n_left = sum(len(v) for v in left.values())
    if n_left and args.limit and args.limit > 0:
        log.info("%d cells not attempted (--limit %d) — expected for a validation sample",
                 n_left, args.limit)
    elif n_left:
        log.warning("%d cells still missing (measure_failed / crashed / skipped above) — "
                    "re-run to retry the retriable ones", n_left)
    return 3 if gpu_dead is not None else 0


if __name__ == "__main__":
    sys.exit(main())
