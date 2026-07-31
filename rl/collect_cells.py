"""
Exhaustive ground-truth collector for the reward table.

Measures EVERY valid (loop, unmerge, factor) cell for one or more splits, so
the oracle stops being "max over whatever the policy happened to sample" and
becomes exact.  With a full table, every downstream experiment — oracle audit,
offline ranking probes, few-shot adaptation, capture-ratio comparisons between
PPO and the bandit — becomes a table lookup with no GPU and no collection
confound.

WHY THIS EXISTS SEPARATELY FROM train.py
----------------------------------------
The training worker asks a policy which single action to take per loop.  This
asks for all of them.  The measurement protocol, however, must be BIT-IDENTICAL
or the collected cells are not comparable with the ones training produced.
Every protocol-bearing block below is mirrored from train._worker_fn and is
marked `MIRROR:`; if that function changes, these must change with it.

TEST BASELINES ARE NOT PRE-MEASURED BY TRAINING
-----------------------------------------------
train.main calls measure_baselines(train + val) only — test baselines are
resolved lazily inside GpuLoopEnv.reset() during the final evaluation and are
never persisted.  Collecting test cells against an empty baseline entry would
silently produce total_ms=0.0 and garbage rewards, so this script measures any
missing baselines FIRST and refuses to proceed for benchmarks that still lack
one.

RESUMABILITY
------------
Cells already present in the cache are never re-measured, and the cache is
written incrementally, so an interrupted or time-limited job can simply be
relaunched with the same arguments and will continue where it stopped.

Usage:
  # what would it cost?  (no GPU, no compiles)
  python3 collect_cells.py CKPT_DIR --split test --dry-run

  # fill the test split
  python3 collect_cells.py CKPT_DIR --split test --num-workers 4 \\
      --compile-failure-penalty -0.16 --reward-deadzone 0.005

  # cap the spend
  python3 collect_cells.py CKPT_DIR --split test,val --max-cells 2000

The measurement flags MUST match the training run that produced the cache, or
the table mixes two reward encodings.
"""

import argparse
import hashlib
import json
import logging
import os
import queue
import shutil
import sys
import time
from itertools import groupby
from pathlib import Path

import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).parent))

from agent import FACTOR_VALUES, build_factor_mask, _IDX_TRIP_COUNT_KNOWN, _IDX_TRIP_COUNT
from hecbench import ARCH, HECBENCH_SRC, discover_benchmarks
from train import (
    build_loop_assignments, measure_baselines, precheck_benchmarks,
    split_benchmarks,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("collect")

# no-op (unmerge=0, factor=1) is free and exactly 0.0 by definition; training
# never stores it, so neither does this.  MIRROR: train._worker_fn.
NOOP = (0, 1)


# ---------------------------------------------------------------------------
# Cell enumeration
# ---------------------------------------------------------------------------

def valid_cells(pre_features_raw: list) -> list[tuple[int, int]]:
    """
    All (unmerge, factor) cells that are legal for this loop.

    Trip-count masking is applied with the RAW feature values, exactly as the
    agent does at selection time — a z-scored tripCount cannot be inverted, and
    using it would enumerate factors LLVM will silently cap.
    MIRROR: train._worker_fn trip_known/trip_count derivation + build_factor_mask.
    """
    trip_known = pre_features_raw[_IDX_TRIP_COUNT_KNOWN] > 0.5
    trip_count = int(pre_features_raw[_IDX_TRIP_COUNT])
    mask = build_factor_mask(trip_known, trip_count).tolist()
    out = []
    for unmerge in (0, 1):
        for i, ok in enumerate(mask):
            if not ok:
                continue
            cell = (unmerge, FACTOR_VALUES[i])
            if cell != NOOP:
                out.append(cell)
    return out


def missing_cells(assignments: list[dict], rewards: dict,
                  postf: "dict | None" = None) -> dict:
    """
    {loop_key: [(unmerge, factor), ...]} for cells not already measured.

    A loop whose cells are ALL measured but whose post-unmerge feature vector
    is missing is included with an empty list, so a re-run still extracts it.
    Without that, a first run where extraction failed would skip the loop
    forever and the model would be fitted on real post-features but evaluated
    on the pre-feature fallback — a silent distribution shift.  Pass
    postf=None to disable (used for pure coverage accounting).
    """
    out: dict = {}
    for a in assignments:
        bench, li = a["benchmark_name"], a["loop_idx"]
        cells = valid_cells(a["pre_features_raw"])
        todo = [c for c in cells
                if f"{bench}|{li}|{c[0]}|{c[1]}" not in rewards]
        key = f"{bench}|{li}"
        if todo:
            out[key] = todo
        elif (postf is not None and key not in postf
              and any(u == 1 for u, _ in cells)):
            out[key] = []
    return out


def pack_by_cells(assignments: list[dict], todo: dict,
                  n_workers: int) -> list[list[dict]]:
    """
    Bin-pack whole BENCHMARKS across workers, weighted by cell count.

    Benchmark granularity (not loop, as training uses) because each worker
    copies a benchmark tree once and reuses it for every loop and cell inside
    it; splitting one benchmark across workers would duplicate that copy.
    Weighted by cells rather than loops because a 19-cell loop is 19x the work
    of a 1-cell loop, and loop counts per benchmark are heavily skewed.
    """
    by_bench: dict = {}
    for a in assignments:
        key = f"{a['benchmark_name']}|{a['loop_idx']}"
        if key in todo:
            by_bench.setdefault(a["benchmark_name"], []).append(a)

    weights = [
        (sum(len(todo[f"{a['benchmark_name']}|{a['loop_idx']}"]) for a in loops),
         name, loops)
        for name, loops in by_bench.items()
    ]
    weights.sort(key=lambda t: -t[0])          # heaviest first

    buckets: list[list[dict]] = [[] for _ in range(n_workers)]
    loads = [0] * n_workers
    for w, _name, loops in weights:
        i = loads.index(min(loads))
        buckets[i].extend(loops)
        loads[i] += w
    for b in buckets:
        b.sort(key=lambda a: (a["benchmark_name"], a["loop_idx"]))
    return buckets


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _collect_worker(rank: int, gpu_id: int, assignments: list[dict],
                    todo: dict, hp: dict, result_q) -> None:
    """
    Measure every assigned cell.  Sends only plain Python over the queue.

    NEVER put torch tensors on this queue: torch.multiprocessing passes them by
    shared memory and frees the backing store when the producer exits, so a
    fast worker that finishes while main is still draining silently corrupts
    its own results (this cost 15k samples in the 2026-07 training run).
    """
    import subprocess
    import torch
    from pathlib import Path as _Path

    _here = _Path(__file__).parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))

    from environment import GpuLoopEnv, LoopRecord
    from hecbench import (
        FeatureNormalizer, compile_single_loop_ex, demangle,
        demangled_to_filter, measure_kernel_time,
    )

    _log = logging.getLogger(f"collect.w{rank}")
    logging.basicConfig(level=logging.INFO,
                        format=f"%(asctime)s %(levelname)s [W{rank}] %(message)s",
                        datefmt="%H:%M:%S")

    try:
        # MIRROR: train._worker_fn — CUDA_VISIBLE_DEVICES before any CUDA call.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        normalizer = FeatureNormalizer.from_state_dict(hp["normalizer_state"])
        baseline_cache: dict = hp["baseline_cache"]
        tmp_dir = _Path(hp["tmp_dir"]) / f"collect_{rank}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        working_set_dir = tmp_dir / "working_set"
        working_set_dir.mkdir(parents=True, exist_ok=True)

        env = GpuLoopEnv(
            arch=hp["arch"], n_runs=hp["n_runs"], nsys_timeout=hp["nsys_timeout"],
            tmp_dir=tmp_dir,
            compile_timeout_penalty=hp["compile_timeout_penalty"],
            compile_failure_penalty=hp["compile_failure_penalty"],
            reward_deadzone=hp["reward_deadzone"],
            gpu_id=gpu_id, normalizer=normalizer, baseline_cache=baseline_cache,
        )

        for bench_name, bench_iter in groupby(
                assignments, key=lambda x: x["benchmark_name"]):
            bench_loops = list(bench_iter)

            # MIRROR: train._worker_fn — benchmark must have a baseline entry.
            if baseline_cache.get(bench_name) is None:
                _log.warning("No baseline for %s — skipping %d loops",
                             bench_name, len(bench_loops))
                continue

            copy_dir = working_set_dir / bench_name
            try:
                if copy_dir.exists():
                    shutil.rmtree(copy_dir)
                shutil.copytree(_Path(bench_loops[0]["benchmark_path"]), copy_dir)
            except Exception as e:
                _log.warning("Failed to copy %s: %s — skipping", bench_name, e)
                continue
            env._benchmark_dir = copy_dir

            for a in bench_loops:
                loop_idx = a["loop_idx"]
                filename, triple = a["filename"], a["triple"]
                loop_key = f"{bench_name}|{loop_idx}"
                cells = todo.get(loop_key, [])
                raw = a["pre_features_raw"]
                kernel_parents = a.get("kernel_parents", [])

                # A loop can be here for cells, for a missing post-unmerge
                # vector, or both.  `has_unmerge` comes from the loop's VALID
                # cells, not the missing ones: a loop whose unmerge cells were
                # all measured on an earlier run still needs its vector.
                has_unmerge = any(u == 1 for u, _ in valid_cells(raw))
                need_post = (hp["collect_post_features"] and has_unmerge
                             and loop_key not in hp["have_postf"])
                if not cells and not need_post:
                    continue

                # MIRROR: train._worker_fn — kernel filter and baseline resolved
                # as a COUPLED pair so both sides are measured at the same scope.
                # `is not None` (not `or`) because a per-kernel time of 0.0 is
                # falsy but valid; on a per-kernel MISS the filter is forced back
                # to None so the modified run also measures total.
                kernel_filter = None
                baseline_ms = baseline_cache.get(bench_name, {}).get("total_ms", 0.0)
                if len(kernel_parents) == 1:
                    _kf = demangled_to_filter(demangle(kernel_parents[0]))
                    _per_kern = (baseline_cache.get(bench_name, {})
                                 .get("per_kernel_ms", {}).get(_kf))
                    if _per_kern is not None:
                        kernel_filter = _kf
                        baseline_ms = _per_kern

                _log.info("%s loop=%d: %d cells", bench_name, loop_idx, len(cells))

                for unmerge, factor in cells:
                    status, err_sig, reward = "ok", "", None
                    try:
                        ok, err_sig = compile_single_loop_ex(
                            copy_dir, loop_idx=loop_idx, unmerge=unmerge,
                            factor=factor, filename=filename, triple=triple,
                            arch=hp["arch"],
                        )
                    except subprocess.TimeoutExpired:
                        # MIRROR: compile timeout → fixed penalty, cached.
                        reward, status, err_sig = (
                            hp["compile_timeout_penalty"], "compile_timeout",
                            "compile timeout")
                        ok = False

                    if status == "ok" and not ok:
                        # MIRROR: compile failure → penalty, cached.
                        reward, status = hp["compile_failure_penalty"], "compile_failed"
                    elif status == "ok":
                        try:
                            modified_ms = measure_kernel_time(
                                copy_dir, arch=hp["arch"], n_runs=hp["n_runs"],
                                nsys_timeout=hp["nsys_timeout"], tmp_dir=tmp_dir,
                                gpu_id=gpu_id, kernel_filter=kernel_filter,
                            )
                        except RuntimeError:
                            # MIRROR: measurement failure is INFRASTRUCTURE, not
                            # a property of the action — never cached, so the
                            # cell stays missing and a later run retries it.
                            _log.warning("%s loop=%d u=%d f=%d — MEASURE FAILED, "
                                         "not cached", bench_name, loop_idx,
                                         unmerge, factor)
                            result_q.put({"type": "measure_failed",
                                          "benchmark": bench_name,
                                          "loop_idx": loop_idx, "rank": rank})
                            continue
                        # MIRROR: clip at -1.0, then symmetric deadzone.  Order
                        # matters: the deadzone must never touch a penalty.
                        reward_raw = (baseline_ms - modified_ms) / max(baseline_ms, 1e-9)
                        reward = max(reward_raw, -1.0)
                        dz = hp["reward_deadzone"]
                        if dz > 0 and abs(reward) < dz:
                            reward = 0.0

                    result_q.put({
                        "type": "cell", "benchmark": bench_name,
                        "loop_idx": loop_idx, "unmerge": unmerge, "factor": factor,
                        "reward": float(reward), "status": status,
                        "error": err_sig, "filename": filename, "triple": triple,
                        "rank": rank,
                    })

                # Post-unmerge features: one extra compile per loop, only when
                # the loop has unmerge cells and no vector is cached yet.  The
                # policy conditions its factor decision on these for unmerge=1,
                # and training persisted them only for train-split loops — so
                # without this, a model trained on real post-features would be
                # evaluated on the pre-feature fallback, a silent distribution
                # shift between fit and test.
                if need_post:
                    try:
                        pre = normalizer.normalize(
                            torch.tensor(raw, dtype=torch.float32))
                        rec = LoopRecord(
                            loop_idx=loop_idx, filename=filename, triple=triple,
                            pre_features=pre, kernel_parents=kernel_parents,
                            trip_count_known=raw[_IDX_TRIP_COUNT_KNOWN] > 0.5,
                            trip_count=int(raw[_IDX_TRIP_COUNT]),
                        )
                        post = env.get_post_unmerge_features(rec)
                        # get_post_unmerge_features returns pre_features
                        # unchanged when extraction fails; storing that would
                        # cache the fallback as if it were a real measurement.
                        if not torch.equal(post.cpu(), pre.cpu()):
                            result_q.put({
                                "type": "postfeat", "benchmark": bench_name,
                                "loop_idx": loop_idx,
                                "features": post.detach().cpu().tolist(),
                                "rank": rank,
                            })
                    except Exception as e:
                        _log.warning("post-feature extraction failed for %s "
                                     "loop=%d: %s", bench_name, loop_idx, e)

            shutil.rmtree(copy_dir, ignore_errors=True)

    except Exception as e:
        import traceback as _tb
        _log.error("Worker %d CRASHED — its remaining cells are unmeasured:\n%s",
                   rank, _tb.format_exc())
        result_q.put({"type": "worker_crashed", "rank": rank,
                      "error": f"{type(e).__name__}: {e}"})
    finally:
        result_q.put({"type": "worker_done", "rank": rank})


# ---------------------------------------------------------------------------
# Cache persistence
# ---------------------------------------------------------------------------

def save_cache(path: Path, rewards: dict, postf: dict, norm_sig: str,
               failure_keys: set, penalty: float, deadzone: float) -> None:
    """
    Write the cache, preserving every key this script does not own and adding
    the failure cells it discovered.

    The `migration` block records WHICH cells hold a compile-failure penalty
    rather than a measurement.  Without it a cell at -0.16 is indistinguishable
    from a genuine -0.16 measurement, so the penalty can never be re-tuned —
    and a partial list is worse than none, because a re-tune would move
    training's failure cells while silently leaving the collector's behind at
    the old value.
    """
    payload: dict = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text())
        except Exception:
            payload = {}
    payload["normalizer_sig"] = norm_sig
    payload["rewards"] = rewards
    payload["post_features"] = postf
    if failure_keys or payload.get("migration"):
        mig = payload.get("migration") or {}
        mig["failure_keys"] = sorted(
            set(mig.get("failure_keys", [])) | failure_keys)
        mig.setdefault("failure_penalty", penalty)
        mig.setdefault("deadzone", deadzone)
        mig.setdefault("deadzoned_keys", [])
        mig.setdefault("history", [])
        payload["migration"] = mig
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(path)          # atomic: a kill mid-write cannot truncate the cache


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint_dir", type=Path)
    p.add_argument("--split", default="test",
                   help="Comma-separated: test, val, train (default: test)")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-cells", type=int, default=0,
                   help="Stop after roughly this many NEW cells (0 = no cap). "
                        "Approximate: whole benchmarks are assigned up front.")
    p.add_argument("--dry-run", action="store_true",
                   help="Report the work and exit — no compiles, no GPU.")
    p.add_argument("--no-post-features", action="store_true",
                   help="Skip post-unmerge feature extraction (1 extra compile "
                        "per loop with unmerge cells).")
    # --- Must match the training run ---
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--compile-failure-penalty", type=float, required=True,
                   help="REQUIRED: must match the training run that built this "
                        "cache, or the table mixes two reward encodings.")
    p.add_argument("--reward-deadzone", type=float, required=True,
                   help="REQUIRED: must match the training run (see above).")
    p.add_argument("--compile-timeout-penalty", type=float, default=-1.0)
    # --- Measurement ---
    p.add_argument("--arch", default=ARCH)
    p.add_argument("--n-runs", type=int, default=2)
    p.add_argument("--nsys-timeout", type=int, default=300)
    p.add_argument("--tmp-dir", default=None)
    p.add_argument("--hecbench-src", default=None)
    p.add_argument("--save-every", type=int, default=25,
                   help="Flush the cache to disk every N cells (default: 25)")
    args = p.parse_args()

    ckpt = args.checkpoint_dir
    elig_file = ckpt / "eligible_benchmarks.json"
    if not elig_file.exists():
        sys.exit(f"missing {elig_file} — point this at a real run directory")

    # MIRROR: train.main — same discovery order feeds split_benchmarks, and the
    # split is only reproducible if that order matches.
    src = Path(args.hecbench_src) if args.hecbench_src else HECBENCH_SRC
    all_benchmarks, _, loop_records_map, normalizer = precheck_benchmarks(
        discover_benchmarks(src), elig_file, skip=True,
    )
    train_b, val_b, test_b = split_benchmarks(
        all_benchmarks, args.val_ratio, args.test_ratio, args.split_seed)
    pools = {"train": train_b, "val": val_b, "test": test_b}

    wanted = [s.strip() for s in args.split.split(",") if s.strip()]
    bad = [s for s in wanted if s not in pools]
    if bad:
        sys.exit(f"unknown split(s): {bad}")
    benches = [b for s in wanted for b in pools[s]]
    log.info("Splits %s → %d benchmarks", wanted, len(benches))

    assignments = build_loop_assignments(benches, loop_records_map)

    # --- Existing cache ---
    rc_file = ckpt / "reward_cache.json"
    rewards, postf = {}, {}
    norm_sig = hashlib.md5(
        json.dumps(normalizer.state_dict(), sort_keys=True).encode()
    ).hexdigest()[:12]
    if rc_file.exists():
        data = json.loads(rc_file.read_text())
        rewards = {k: float(v) for k, v in data.get("rewards", {}).items()}
        # MIRROR: train.main — post_features are NORMALISED vectors, valid only
        # under the normalizer that produced them.  Keeping them across a
        # refit and then stamping the new signature would silently relabel
        # stale vectors as current.  Rewards are raw measurements and survive.
        if data.get("normalizer_sig") == norm_sig:
            postf = data.get("post_features", {})
        elif data.get("post_features"):
            log.warning("Normalizer signature changed (%s != %s) — discarding "
                        "%d cached post-unmerge vectors; they will be "
                        "re-extracted", data.get("normalizer_sig"), norm_sig,
                        len(data["post_features"]))

    todo = missing_cells(assignments, rewards,
                         None if args.no_post_features else postf)
    n_missing = sum(len(v) for v in todo.values())
    n_total = sum(len(valid_cells(a["pre_features_raw"])) for a in assignments)
    log.info("Cells: %d valid, %d already measured, %d missing (%.1f%% filled)",
             n_total, n_total - n_missing, n_missing,
             100 * (n_total - n_missing) / max(n_total, 1))

    if args.max_cells and n_missing > args.max_cells:
        kept, running = {}, 0
        for k, v in todo.items():
            if running >= args.max_cells:
                break
            kept[k] = v
            running += len(v)
        todo = kept
        n_missing = running
        log.info("Capped to %d cells by --max-cells", n_missing)

    if not n_missing:
        log.info("Nothing to measure — the requested splits are complete.")
        return

    est_h = n_missing * 1.5 / 60 / max(args.num_workers, 1)
    log.info("Estimated wall time: ~%.1f h on %d workers (~1.5 min/cell)",
             est_h, args.num_workers)

    # --- Baselines: test/val baselines may never have been persisted ---
    # The file is {"arch": ..., "baselines": {name: {...}}} — the flat map the
    # workers expect lives under "baselines", not at the top level.
    bl_file = ckpt / "baseline_cache.json"
    baseline_cache: dict = {}
    if bl_file.exists():
        try:
            baseline_cache = json.loads(bl_file.read_text()).get("baselines", {})
        except Exception as e:
            log.warning("Could not read %s: %s", bl_file, e)
    _todo_benches = {a["benchmark_name"] for a in assignments
                     if f"{a['benchmark_name']}|{a['loop_idx']}" in todo}
    need_bl = [b for b in benches
               if b.name in _todo_benches and b.name not in baseline_cache]
    if need_bl:
        log.warning("%d benchmark(s) have no cached baseline — measuring them "
                    "first (train.main only pre-measures train+val)", len(need_bl))

    if args.dry_run:
        log.info("Dry run: %d cells across %d loops, %d benchmarks needing "
                 "baselines. Nothing measured.",
                 n_missing, len(todo), len(need_bl))
        return

    tmp_dir = Path(args.tmp_dir) if args.tmp_dir else ckpt / "collect_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if need_bl:
        baseline_cache = measure_baselines(
            need_bl, loop_records_map=loop_records_map, arch=args.arch,
            n_runs=args.n_runs, nsys_timeout=args.nsys_timeout,
            tmp_dir=tmp_dir, gpu_id=0, cache_file=bl_file,
        )
        still = [b.name for b in need_bl if b.name not in baseline_cache]
        if still:
            log.warning("No baseline could be measured for %d benchmark(s); "
                        "their cells are SKIPPED: %s", len(still), sorted(still))

    hp = {
        "arch": args.arch, "n_runs": args.n_runs,
        "nsys_timeout": args.nsys_timeout, "tmp_dir": str(tmp_dir),
        "compile_timeout_penalty": args.compile_timeout_penalty,
        "compile_failure_penalty": args.compile_failure_penalty,
        "reward_deadzone": args.reward_deadzone,
        "normalizer_state": normalizer.state_dict(),
        "baseline_cache": baseline_cache,
        "collect_post_features": not args.no_post_features,
        "have_postf": set(postf.keys()),
    }

    buckets = pack_by_cells(assignments, todo, args.num_workers)
    result_q: mp.Queue = mp.Queue()
    procs = []
    for rank in range(args.num_workers):
        if not buckets[rank]:
            continue
        pr = mp.Process(target=_collect_worker,
                        args=(rank, rank, buckets[rank], todo, hp, result_q),
                        daemon=True)
        pr.start()
        procs.append(pr)
    log.info("Launched %d workers", len(procs))

    n_alive = len(procs)
    done = measured = failed = crashed = 0
    by_status: dict = {}
    # Cells whose value is a penalty, not a measurement — recorded so the
    # penalty stays re-tunable (see save_cache).
    failure_keys: set = set()
    t0 = time.time()
    msg_timeout = args.n_runs * args.nsys_timeout + 600

    while done < n_alive:
        try:
            msg = result_q.get(timeout=msg_timeout)
        except queue.Empty:
            if not any(pr.is_alive() for pr in procs):
                log.error("All workers died")
                break
            continue
        except Exception as e:
            log.error("result_q message DROPPED (%s: %s) — cell lost",
                      type(e).__name__, e)
            if not any(pr.is_alive() for pr in procs):
                break
            continue

        t = msg["type"]
        if t == "cell":
            key = (f"{msg['benchmark']}|{msg['loop_idx']}"
                   f"|{msg['unmerge']}|{msg['factor']}")
            rewards[key] = msg["reward"]
            measured += 1
            by_status[msg["status"]] = by_status.get(msg["status"], 0) + 1
            if msg["status"] == "compile_failed":
                failure_keys.add(key)
            if measured % args.save_every == 0:
                save_cache(rc_file, rewards, postf, norm_sig, failure_keys,
                           args.compile_failure_penalty, args.reward_deadzone)
                rate = measured / max(time.time() - t0, 1e-9) * 3600
                log.info("progress %d/%d cells (%.0f/h, eta %.1f h)",
                         measured, n_missing, rate,
                         (n_missing - measured) / max(rate, 1e-9))
        elif t == "postfeat":
            postf[f"{msg['benchmark']}|{msg['loop_idx']}"] = msg["features"]
        elif t == "measure_failed":
            failed += 1
        elif t == "worker_crashed":
            crashed += 1
            log.error("Worker %d crashed (%s)", msg["rank"], msg.get("error"))
        elif t == "worker_done":
            done += 1

    for pr in procs:
        pr.join(timeout=30)
        if pr.exitcode not in (0, None):
            log.error("Worker exited with code %s — see its traceback above",
                      pr.exitcode)
    save_cache(rc_file, rewards, postf, norm_sig, failure_keys,
               args.compile_failure_penalty, args.reward_deadzone)

    todo_after = missing_cells(assignments, rewards)
    n_left = sum(len(v) for v in todo_after.values())
    log.info("")
    log.info("=== done in %.1f h ===", (time.time() - t0) / 3600)
    log.info("  cells measured   : %d", measured)
    log.info("  by status        : %s", dict(sorted(by_status.items())))
    log.info("  measure failures : %d (not cached — rerun to retry)", failed)
    log.info("  workers crashed  : %d", crashed)
    log.info("  post-feature vecs: %d", len(postf))
    log.info("  coverage now     : %d/%d (%.1f%%), %d cells still missing",
             n_total - n_left, n_total,
             100 * (n_total - n_left) / max(n_total, 1), n_left)
    log.info("  cache            : %s", rc_file)


if __name__ == "__main__":
    # spawn is required for CUDA safety: forking after CUDA init is unsupported.
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
