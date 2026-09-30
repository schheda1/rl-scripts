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

INVARIANTS (must match collect_cells / the existing cache): 20 cold `nsys profile` runs;
`nsys profile --trace=cuda --sample=none --cpuctxsw=none`; reward =
clip_-1((baseline_ms - median(sum_kernel_times))/max(baseline_ms,1e-9)) then deadzone;
compile_failed/timeout → cached penalty; measure_failed → NOT cached (retried).  --n-runs
REQUIRED and must match the run.  Pin the nsys version across the whole collection.

Usage:
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
from store_common import (cellkey, structural_copy, dir_bytes, stage_from_store,
                          is_cell_done, load_manifest, toolchain_stamp,
                          benchmark_fingerprint)

_STOP = None                       # queue sentinel (picklable, and no work item is None)


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


# ---------------------------------------------------------------------------
# Stage workers  (each: plain-data queues only; CUDA_VISIBLE_DEVICES set first)
# ---------------------------------------------------------------------------

def compile_worker(rank, work_q, q12, result_q, cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""            # CPU-only stage: never grab a GPU
    from hecbench import compile_single_loop_ex, _get_run_command
    lg = logging.getLogger(f"compile.{rank}")
    shm = Path(cfg["shm_root"])
    prebuilt = cfg.get("prebuilt") or set()           # cellkeys with a verified store binary
    store = cfg.get("store")
    while True:
        item = work_q.get()
        if item is _STOP:
            break
        bench, li = item["benchmark_name"], item["loop_idx"]
        u, f = item["unmerge"], item["factor"]
        key = cellkey(bench, li, u, f)
        bundle = shm / "bundles" / key.replace("|", "__")
        try:
            shutil.rmtree(bundle, ignore_errors=True)
            structural_copy(Path(item["benchmark_path"]), bundle, cfg["copy_threshold"])
            # FAST PATH: drop the pre-built binary (no make) if this cell is in the store;
            # stage_from_store unlinks any existing target first (never writes through a
            # symlink), and returns False → we fall back to building.
            staged = key in prebuilt and store and stage_from_store(store, key, bundle)
            if not staged:
                try:
                    ok, err = compile_single_loop_ex(
                        bundle, loop_idx=li, unmerge=u, factor=f,
                        filename=item["filename"], triple=item["triple"], arch=cfg["arch"])
                except subprocess.TimeoutExpired:
                    shutil.rmtree(bundle, ignore_errors=True)
                    result_q.put({"type": "compile_timeout", "key": key, "benchmark": bench,
                                  "reward": cfg["compile_timeout_penalty"]})
                    continue
                if not ok:
                    shutil.rmtree(bundle, ignore_errors=True)
                    result_q.put({"type": "compile_failed", "key": key, "benchmark": bench,
                                  "reward": cfg["compile_failure_penalty"], "error": err})
                    continue
            run_cmd = _get_run_command(bundle, cfg["arch"])
            # bundle now lives in shm until exec deletes it (immutable in between).
            q12.put({"key": key, "benchmark": bench, "loop_idx": li,
                     "unmerge": u, "factor": f, "bundle": str(bundle),
                     "run_cmd": run_cmd, "kernel_filter": item["kernel_filter"],
                     "baseline_ms": item["baseline_ms"]})
        except Exception as e:
            shutil.rmtree(bundle, ignore_errors=True)
            lg.warning("compile crashed for %s: %s — treated as measure_failed", key, e)
            result_q.put({"type": "measure_failed", "key": key, "benchmark": bench})
    result_q.put({"type": "compile_done", "rank": rank})


def exec_worker(rank, gpu_id, q12, q23, result_q, cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)   # one nsys profile at a time / GPU
    tmp = Path(cfg["tmp_root"])
    my_tmpdir = tmp / f"nsys_tmp_r{rank}"             # per-WORKER TMPDIR (nsys isolation;
    #                                                  per-rank, not per-gpu, so >1 exec
    #                                                  per GPU still get separate temp)
    my_tmpdir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(my_tmpdir)
    env = {**os.environ, "ARCH": cfg["arch"], "CUDA_VISIBLE_DEVICES": str(gpu_id),
           "TMPDIR": str(my_tmpdir)}
    lg = logging.getLogger(f"exec.g{gpu_id}")
    while True:
        item = q12.get()
        if item is _STOP:
            break
        key, bundle = item["key"], Path(item["bundle"])
        rep_dir = tmp / "reports" / key.replace("|", "__")
        rep_dir.mkdir(parents=True, exist_ok=True)
        reports = []
        for k in range(cfg["n_runs"]):
            out = rep_dir / f"run{k}"
            try:
                subprocess.run(
                    f"nsys profile --trace=cuda --sample=none --cpuctxsw=none "
                    f"--output={out} --force-overwrite=true {item['run_cmd']}",
                    cwd=bundle, shell=True, capture_output=True, text=True,
                    timeout=cfg["nsys_timeout"], env=env)
            except subprocess.TimeoutExpired:
                continue                               # failed run → skipped (as today)
            rep = Path(f"{out}.nsys-rep")
            if rep.exists():
                reports.append(str(rep))
        shutil.rmtree(bundle, ignore_errors=True)      # free shm immediately (last reader)
        q23.put({"key": key, "benchmark": item["benchmark"], "reports": reports,
                 "report_dir": str(rep_dir), "kernel_filter": item["kernel_filter"],
                 "baseline_ms": item["baseline_ms"]})
    result_q.put({"type": "exec_done", "rank": rank})


def parse_worker(rank, q23, result_q, cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""            # CPU-only
    from hecbench import _parse_nsys_kernel_times, _sum_kernel_times
    env = {**os.environ, "ARCH": cfg["arch"]}
    lg = logging.getLogger(f"parse.{rank}")
    while True:
        item = q23.get()
        if item is _STOP:
            break
        key, kf = item["key"], item["kernel_filter"]
        times = []
        for rep in item["reports"]:
            try:
                r = subprocess.run(
                    f"nsys stats --report=cuda_gpu_kern_sum --format=csv {rep}",
                    shell=True, capture_output=True, text=True, timeout=30, env=env)
            except subprocess.TimeoutExpired:
                continue
            t = _sum_kernel_times(_parse_nsys_kernel_times(r.stdout + r.stderr), kf)
            if t is not None:
                times.append(t)
        shutil.rmtree(item["report_dir"], ignore_errors=True)   # free tmp (last reader)
        if not times:
            result_q.put({"type": "measure_failed", "key": key,
                          "benchmark": item["benchmark"]})
            continue
        reward = compute_reward(item["baseline_ms"], statistics.median(times),
                                cfg["reward_deadzone"])
        result_q.put({"type": "cell", "key": key, "benchmark": item["benchmark"],
                      "reward": float(reward)})
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
        bundle = shm / "postfeat" / f"{bench}__{li}"
        try:
            shutil.rmtree(bundle, ignore_errors=True)
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
            shutil.rmtree(bundle, ignore_errors=True)
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

    print(f"\n  {'ALL PASS' if ok else 'FAILURES ABOVE'}")
    return 0 if ok else 1


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint_dir", type=Path, nargs="?")
    p.add_argument("--split", default="test", help="comma list: test,val,train")
    p.add_argument("--n-runs", type=int, help="REQUIRED (unless --selftest); match the run")
    p.add_argument("--exec", dest="n_exec_per_gpu", type=int, default=1,
                   help="exec processes PER GPU (default 1; one nsys profile/GPU)")
    p.add_argument("--compile", dest="n_compile_per_gpu", type=int, default=2,
                   help="compile processes per GPU (default 2)")
    p.add_argument("--parse", dest="n_parse_per_gpu", type=int, default=3,
                   help="parse processes per GPU (default 3)")
    p.add_argument("--postfeat", dest="n_postfeat", type=int, default=2,
                   help="TOTAL post-feature processes (default 2)")
    p.add_argument("--gpus", type=int, default=1, help="#GPUs on this node")
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
    p.add_argument("--read-only", action="store_true",
                   help="VALIDATION: run the real compile->nsys->parse pipeline but NEVER "
                        "write reward_cache.json or baseline_cache.json (baselines are not "
                        "measured either — cells whose benchmark has no baseline are skipped). "
                        "All scratch + the memlog go OUTSIDE the checkpoint dir. Pair with "
                        "--limit for a quick end-to-end check.")
    p.add_argument("--limit", type=int, default=0,
                   help="cap the run to N cells (round-robin across benchmarks) — for a fast "
                        "validation sample; 0 = no cap")
    p.add_argument("--selftest", action="store_true")
    # must match the training run / existing cache
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--compile-failure-penalty", type=float)
    p.add_argument("--reward-deadzone", type=float)
    p.add_argument("--compile-timeout-penalty", type=float, default=-1.0)
    p.add_argument("--arch", default=None, help="default: auto-detected TARGET_ARCH")
    p.add_argument("--nsys-timeout", type=int, default=300)
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
    from hecbench import ARCH, HECBENCH_SRC, discover_benchmarks, IR2VEC_VOCAB
    from train import precheck_benchmarks, split_benchmarks, build_loop_assignments, measure_baselines
    if args.arch is None:
        args.arch = ARCH

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
    G = max(args.gpus, 1)
    n_exec = args.n_exec_per_gpu * G
    n_compile = args.n_compile_per_gpu * G
    n_parse = args.n_parse_per_gpu * G
    n_postfeat = 0 if args.no_post_features else max(args.n_postfeat, 0)
    if (not args.no_post_features) and n_postfeat == 0:
        log.warning("--postfeat 0 with post-features enabled → post-features NOT extracted")
    if args.n_exec_per_gpu > 1:
        log.warning("--exec %d (>1 per GPU): multiple concurrent `nsys profile` on ONE GPU "
                    "CORRUPT the timing measurements (GPU contention). Use --exec 1 unless "
                    "the GPUs are genuinely distinct/partitioned.", args.n_exec_per_gpu)

    # --- existing cache (resume truth) ---
    rc_file = ckpt / "reward_cache.json"
    rewards, postf = {}, {}
    norm_sig = hashlib.md5(json.dumps(normalizer.state_dict(),
                                      sort_keys=True).encode()).hexdigest()[:12]
    if rc_file.exists():
        data = json.loads(rc_file.read_text())
        rewards = {k: float(v) for k, v in data.get("rewards", {}).items()}
        if data.get("normalizer_sig") == norm_sig:
            postf = data.get("post_features", {})
    todo = missing_cells(assignments, rewards,
                         None if args.no_post_features else postf)
    n_missing = sum(len(v) for v in todo.values())

    # --- baselines FIRST (a cell's baseline_ms must exist before its reward) ---
    bl_file = ckpt / "baseline_cache.json"
    baseline_cache = {}
    if bl_file.exists():
        baseline_cache = json.loads(bl_file.read_text()).get("baselines", {})
    todo_benches = {a["benchmark_name"] for a in assignments
                    if f"{a['benchmark_name']}|{a['loop_idx']}" in todo}
    need_bl = [b for b in benches if b.name in todo_benches and b.name not in baseline_cache]
    # READ-ONLY: never measure baselines (that path WRITES baseline_cache.json). Cells whose
    # benchmark has no baseline are dropped below (base_ms<=0) rather than measured.
    if ro and need_bl:
        log.warning("READ-ONLY: %d benchmark(s) have no baseline — their cells are SKIPPED "
                    "(baselines are never measured or written in read-only mode)", len(need_bl))
    if ro:
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
    log.info("workers: exec=%d compile=%d parse=%d postfeat=%d  (per-GPU %d/%d/%d × %d GPUs)",
             n_exec, n_compile, n_parse, n_postfeat,
             args.n_exec_per_gpu, args.n_compile_per_gpu, args.n_parse_per_gpu, G)

    if args.dry_run:
        log.info("dry run — nothing measured.")
        return 0
    if not n_missing and not post_items:
        log.info("nothing to do.")
        return 0

    if need_bl:
        log.info("measuring %d missing baselines first...", len(need_bl))
        baseline_cache = measure_baselines(
            need_bl, loop_records_map=loop_records_map, arch=args.arch,
            n_runs=args.n_runs, nsys_timeout=args.nsys_timeout,
            tmp_dir=tmp_root, gpu_id=0, cache_file=bl_file)

    # Attach scope per LOOP (resolved once, now that baselines exist) and DROP cells
    # whose benchmark still has no baseline — they cannot be scored, so leave them
    # missing to retry on a later run rather than divide by a bogus 0 baseline.
    scope = {}
    for a in assignments:
        k = (a["benchmark_name"], a["loop_idx"])
        if k not in scope:
            scope[k] = resolve_kernel_scope(a.get("kernel_parents", []),
                                            baseline_cache, a["benchmark_name"])
    kept, dropped = [], 0
    for w in work_items:
        kf, base_ms = scope[(w["benchmark_name"], w["loop_idx"])]
        if base_ms <= 0.0:
            dropped += 1
            continue
        w["kernel_filter"], w["baseline_ms"] = kf, base_ms
        kept.append(w)
    work_items = kept
    if dropped:
        log.warning("%d cells skipped — benchmark has no baseline (measure it, re-run)", dropped)
    n_missing = len(work_items)          # completion target: one terminal result per item

    # --- optional --limit: validate a small, benchmark-spread sample (compile/run/parse) ---
    if args.limit and args.limit > 0:
        work_items = _limit_by_benchmark(work_items, args.limit)
        post_items = _limit_by_benchmark(post_items, args.limit)
        n_missing = len(work_items)
        log.info("--limit %d → %d cells across %d benchmarks + %d postfeat loops",
                 args.limit, len(work_items),
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
            if ms.get("arch") != live["arch"] or ms.get("clang") != live["clang"]:
                log.warning("store stamp MISMATCH (arch %s vs %s / clang differs) — "
                            "IGNORING store, building instead", ms.get("arch"), live["arch"])
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
        "shm_root": str(shm_root), "tmp_root": str(tmp_root),
        "copy_threshold": args.copy_threshold,
        "normalizer_state": normalizer.state_dict(),
        "prebuilt": prebuilt, "store": store,
    }
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
    exec_procs = [mp.Process(target=exec_worker, args=(r, r % G, q12, q23, result_q, cfg),
                             daemon=True) for r in range(n_exec)]
    parse_procs = [mp.Process(target=parse_worker, args=(r, q23, result_q, cfg),
                              daemon=True) for r in range(n_parse)]
    postfeat_procs = [mp.Process(target=postfeat_worker, args=(r, post_q, result_q, cfg),
                                 daemon=True) for r in range(n_postfeat)]
    all_procs = compile_procs + exec_procs + parse_procs + postfeat_procs
    for p in all_procs:
        p.start()

    # Separate feeders so a full post_q can't stall the work feeder (GPU-critical path).
    threading.Thread(target=feed_queue, args=(work_items, work_q, n_compile),
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

    failure_keys = set()
    exec_left, parse_left, postfeat_left = n_exec, n_parse, n_postfeat
    q12_stops_sent = q23_stops_sent = False
    by_status = {}
    t0 = time.time()
    GET_TIMEOUT = 120          # periodic liveness check when idle; benign if a cell > this

    def _persist():
        if ro:
            return                         # read-only: NEVER write reward_cache.json
        save_cache(rc_file, rewards, postf, norm_sig, failure_keys,
                   args.compile_failure_penalty, args.reward_deadzone)

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
            if t != "measure_failed":                 # measure_failed is NOT cached
                rewards[msg["key"]] = msg["reward"]
                if t in ("compile_failed", "compile_timeout"):
                    failure_keys.add(msg["key"])
            counters["cells_done"] += 1
            by_status[t] = by_status.get(t, 0) + 1
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
    for p in all_procs:
        p.join(timeout=30)
    log.info("DONE  cells_done=%d/%d  status=%s  postfeat=%d  wall=%.1fh  memlog=%s",
             counters["cells_done"], n_missing, by_status, len(postf),
             (time.time() - t0) / 3600, memlog)
    left = missing_cells(assignments, rewards,
                         None if args.no_post_features else postf)
    n_left = sum(len(v) for v in left.values())
    if n_left:
        log.warning("%d cells still missing (measure_failed / crashed) — re-run to retry",
                    n_left)
    return 0


if __name__ == "__main__":
    sys.exit(main())
