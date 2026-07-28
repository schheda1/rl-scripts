"""
Outer training loop for the per-loop UU RL pipeline.

Usage:
  python train.py [--epochs N] [--buffer-size N] [--n-runs N]
                  [--arch sm_80] [--checkpoint-dir checkpoints/]
                  [--checkpoint-every N]
                  [--resume checkpoints/latest.pt]
                  [--val-ratio 0.15] [--test-ratio 0.15] [--split-seed 42]
                  [--skip-precheck]

Startup pre-flight check:
  Before splitting, every discovered benchmark is compiled with LoopCount to
  confirm it has at least one eligible loop.  Results are cached in
  {checkpoint_dir}/eligible_benchmarks.json.  Use --skip-precheck to load
  from cache (or skip entirely if no cache exists).

Dynamic benchmark removal:
  If reset() fails for a benchmark during training or validation it is removed
  from its list for all future epochs and logged at WARNING level.
"""

import argparse
import csv
import getpass
import hashlib
import heapq
import json
import logging
import queue
import random
import sys
from datetime import datetime
from pathlib import Path
import statistics 

import torch.multiprocessing as mp

import matplotlib
matplotlib.use("Agg")   # non-interactive — safe on headless servers
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).parent))

from agent import (
    Agent, BanditAgent, RolloutBuffer, RolloutEntry, FACTOR_VALUES, N_FEATURES,
)
from environment import GpuLoopEnv
from hecbench import (
    ARCH, FEATURE_COLUMNS, STUDY_A_NUMPATHS_MIN, STUDY_A_NUMPATHS_MAX,
    discover_benchmarks,
)

# Bump when the feature schema changes so a --skip-precheck run cannot load a
# cache whose pre_features_raw / normalizer are the wrong dimensionality.
# v1 = 18 structural features; v2 = 18 structural + 75 IR2Vec embeddings.
FEATURES_VERSION = 2

# Eligibility marker: changing which loops are eligible (e.g. Study A's
# numPaths>1 gate) does NOT change feature dims, so FEATURES_VERSION won't
# catch a stale precheck cache reused via --skip-precheck.  Derived from the
# actual bounds so a changed STUDY_A_NUMPATHS_MAX env var invalidates the cache.
ELIGIBILITY_VERSION = (
    f"studyA:numPaths>{STUDY_A_NUMPATHS_MIN},<={STUDY_A_NUMPATHS_MAX}"
)

# Placeholder factor log-prob for no-op (unmerge==0) rollout entries — never
# used in the PPO update (factor_active=False masks it out), but RolloutEntry
# needs a tensor.
_ZERO_LOGP = torch.zeros(())


class _EpochFilter(logging.Filter):
    """
    Injects %(epoch_tag)s into every log record on the main process.
    Set to "" before training starts so pre-epoch messages are unaffected.
    During training, tag is "[epoch/total] " (with trailing space).
    """
    def __init__(self) -> None:
        super().__init__()
        self.tag: str = ""

    def set(self, epoch: int, total: int) -> None:
        self.tag = f"[{epoch}/{total}] "

    def clear(self) -> None:
        self.tag = ""

    def filter(self, record: logging.LogRecord) -> bool:
        record.epoch_tag = self.tag  # type: ignore[attr-defined]
        return True


_epoch_filter = _EpochFilter()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(epoch_tag)s%(message)s",
    datefmt="%H:%M:%S",
)
# Attach filter to every handler on the root logger so %(epoch_tag)s
# is always defined regardless of which logger emits the record.
for _h in logging.root.handlers:
    _h.addFilter(_epoch_filter)

log = logging.getLogger("train")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--buffer-size", type=int, default=128,
                   help="Rollout buffer capacity before a PPO update is triggered. "
                        "Larger buffers reduce overfit risk per update and improve "
                        "sample diversity — important when benchmarks vary from 1-2 "
                        "to ~40 eligible loops. (default: 128)")
    p.add_argument("--n-runs", type=int, default=20,
                   help="nsys measurement repetitions per kernel-time estimate")
    p.add_argument("--nsys-timeout", type=int, default=300,
                   help="Per-run nsys profile timeout in seconds (default: 300)")
    p.add_argument("--compile-timeout-penalty", type=float, default=-1.0,
                   help="Reward assigned when compilation times out due to SCEV/unroll "
                        "complexity. Should be negative to discourage large factors. "
                        "(default: -1.0)")
    p.add_argument("--compile-failure-penalty", type=float, default=-0.25,
                   help="Reward assigned when compilation FAILS (not a timeout). "
                        "Previously 0.0, which is indistinguishable from a genuine "
                        "no-effect transform — the agent could not learn to avoid "
                        "over-aggressive actions. Must rank below no-op (0.0); kept "
                        "well above -1.0 so the ~36%% failure mass does not swamp the "
                        "genuine slowdown signal. (default: -0.25)")
    p.add_argument("--reward-deadzone", type=float, default=0.01,
                   help="Rewards with |r| below this are treated as measurement noise "
                        "and set to exactly 0.0. Applied AFTER failure penalties, so "
                        "penalties are never zeroed. Set 0 to disable. (default: 0.01)")
    p.add_argument("--test-checkpoints", type=str, default="best,last",
                   help="Which checkpoints to run the final test evaluation on: "
                        "comma-separated from {best,last} or explicit .pt paths. "
                        "'best' guards against best-val being picked early; 'last' "
                        "shows the converged policy. (default: best,last)")
    # --- Run health gates: fail fast instead of producing empty epochs ---
    p.add_argument("--tolerate-worker-crashes", type=int, default=0,
                   help="Abort the run if more than N workers crash in an epoch. "
                        "A crash is a code bug — every expected failure is handled "
                        "in-worker — and it silently forfeits that worker's whole "
                        "shard. (default: 0, i.e. abort on the first crash)")
    p.add_argument("--min-sample-frac", type=float, default=0.5,
                   help="An epoch collecting fewer than this fraction of the "
                        "BEST epoch's sample count so far is 'starved' (zero "
                        "samples always counts). Relative to the best epoch, "
                        "not to the assigned count, so loops that are "
                        "legitimately skipped every epoch never trip it. "
                        "(default: 0.5)")
    p.add_argument("--max-starved-epochs", type=int, default=2,
                   help="Abort after this many CONSECUTIVE starved epochs — "
                        "training is not seeing its data. (default: 2)")
    p.add_argument("--arch", type=str, default=ARCH)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--checkpoint-every", type=int, default=1,
                   help="Save a checkpoint every N epochs (default: every epoch)")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--agent", choices=("ppo", "bandit"), default="ppo",
                   help="Solution method for the (identical) contextual-bandit "
                        "formulation: 'ppo' = on-policy clipped policy gradient; "
                        "'bandit' = value-based Q-regression with epsilon-greedy "
                        "collection, warm-started from the reward cache. Same "
                        "nets, buffer, workers, caches, and eval paths; greedy "
                        "val/test selection is identical code for both. "
                        "(default: ppo)")
    p.add_argument("--bandit-epsilon", type=float, default=0.3,
                   help="Bandit collection epsilon at epoch 1; decays linearly "
                        "to --bandit-epsilon-final over the run. Ignored for "
                        "--agent ppo. (default: 0.3)")
    p.add_argument("--bandit-epsilon-final", type=float, default=0.05,
                   help="Bandit epsilon at the last epoch. Set equal to "
                        "--bandit-epsilon to disable the decay. (default: 0.05)")
    p.add_argument("--bandit-warm-epochs", type=int, default=10,
                   help="Warm-start passes over the cached (state, action, "
                        "reward) cells before epoch 1 (each pass = K update "
                        "epochs). 0 disables. Ignored for --agent ppo. "
                        "(default: 10)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01,
                   help="AdamW decoupled weight decay on the networks' weight "
                        "matrices (biases and LayerNorm params exempt). Bounds "
                        "policy-logit growth — the mechanism of entropy "
                        "collapse. 0 disables. (default: 0.01)")
    p.add_argument("--max-grad-norm", type=float, default=0.5,
                   help="Global gradient-norm clip across actor+critic per "
                        "update. 0 disables. (default: 0.5)")
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--K", type=int, default=2, dest="K",
                   help="PPO epochs per rollout update. K=2 with buffer=128 gives "
                        "~32 gradient steps per 128 samples (0.25 updates/sample), "
                        "reducing overfit risk vs the previous K=4/buffer=32 ratio. "
                        "Increase if rewards plateau; decrease if loss oscillates.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--value-loss-coef", type=float, default=0.5)
    p.add_argument("--entropy-coef", type=float, default=0.01,
                   help="Entropy bonus coefficient (encourages exploration). "
                        "Increase if no-op rate stays >80%% early in training.")
    p.add_argument("--entropy-coef-final", type=float, default=0.001,
                   help="Entropy coefficient at the LAST epoch: the coefficient "
                        "decays linearly from --entropy-coef to this value over "
                        "the run, so the policy explores early and commits late. "
                        "Set equal to --entropy-coef to disable the decay.")
    p.add_argument("--val-ratio", type=float, default=0.15,
                   help="Fraction of benchmarks held out for validation")
    p.add_argument("--test-ratio", type=float, default=0.15,
                   help="Fraction of benchmarks held out for test")
    p.add_argument("--split-seed", type=int, default=42,
                   help="RNG seed for train/val/test split (ensures reproducibility)")
    p.add_argument("--tmp-dir", type=str,
                   default=f"/tmp/rl_pipeline_{getpass.getuser()}",
                   help="Directory for nsys reports and other pipeline temp files. "
                        "Created automatically if it does not exist.")
    p.add_argument("--skip-precheck", action="store_true",
                   help="Skip the pre-flight LoopCount check. If a cached "
                        "eligible_benchmarks.json exists in the checkpoint dir "
                        "it will be used; otherwise all discovered benchmarks "
                        "are passed to the split without verification.")
    p.add_argument("--hecbench-src", type=str, default=None,
                   help="Override path to HeCBench/src")
    p.add_argument("--benchmarks", type=str, nargs="+", default=None,
                   metavar="NAME",
                   help="Restrict to these benchmark names before splitting. "
                        "Must match directory names under HeCBench/src.")
    p.add_argument("--num-workers", type=int, default=1,
                   help="Number of parallel GPU workers. Worker k uses GPU k. "
                        "Default 1 = sequential path (unchanged behaviour). "
                        "Requires at least --num-workers GPUs to be visible.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Pre-flight eligibility check
# ---------------------------------------------------------------------------

def _dedup_loop_records(
    loop_records_map: dict[str, list[dict]],
) -> tuple[dict[str, list[dict]], int]:
    """
    Drop within-benchmark loops whose raw feature vector duplicates an earlier
    loop's — overwhelmingly sibling template instantiations of the same source
    loop (observed: sortKV-cuda with 182 eligible loops, mostly identical
    features, consuming 67% of epoch-1 wall clock).  Feature-identical loops
    are the same sample to the policy; keeping one representative per unique
    vector loses nothing statistically and removes contradictory duplicate
    signals within a benchmark.

    Deliberately does NOT dedup across benchmarks: identical features from
    different applications may carry genuinely different rewards (known
    feature-aliasing limitation) and are kept as real environment signal.

    Returns (deduped_map, n_dropped).  Keeps the first (lowest loop_idx)
    representative.
    """
    deduped: dict[str, list[dict]] = {}
    dropped = 0
    for bname, records in loop_records_map.items():
        seen: set[tuple] = set()
        keep: list[dict] = []
        for rec in records:
            key = tuple(rec.get("pre_features_raw", []))
            if key in seen:
                dropped += 1
                continue
            seen.add(key)
            keep.append(rec)
        deduped[bname] = keep
    return deduped, dropped


def precheck_benchmarks(
    benchmarks: list[Path],
    cache_file: Path,
    skip: bool,
) -> tuple[list[Path], dict[str, int], dict[str, list[dict]], "FeatureNormalizer"]:
    """
    Return (eligible_benchmarks, loop_counts, loop_records_map, normalizer).

    loop_counts      — benchmark name → number of eligible loops (for logging)
    loop_records_map — benchmark name → list of per-loop dicts:
                         {loop_idx, filename, triple, pre_features_raw: list[float]}
                       pre_features_raw stores un-normalised raw values; workers
                       apply the normalizer at runtime so the cache stays valid
                       if the normalizer is ever re-fitted.
    normalizer       — FeatureNormalizer fitted on all eligible loop rows

    If *skip* is True and a valid cache exists, load from cache.
    Otherwise run LoopCount on each benchmark and save results to cache.
    """
    from hecbench import FeatureNormalizer, _row_to_tensor, get_loop_features

    # --- Try to load from cache ---
    if skip and cache_file.exists():
        try:
            data = json.loads(cache_file.read_text())
            # Feature-schema version guard: a cache written before the IR2Vec
            # feature change carries 18-dim pre_features_raw and an 18-dim
            # normalizer.  Loading it would silently train on the wrong feature
            # space, so discard it and fall through to a fresh precheck.
            if data.get("features_version") != FEATURES_VERSION:
                log.warning(
                    "Precheck cache features_version=%s != %d — feature schema "
                    "changed; discarding cache and re-running the pre-flight check.",
                    data.get("features_version"), FEATURES_VERSION,
                )
                raise ValueError("stale features_version")
            # Eligibility guard: changing the eligible loop SET (Study A's
            # numPaths>1 gate) does not change feature dims, so it slips past the
            # features_version check.  A mismatch means the cache holds a
            # different loop population — discard and re-run.
            if data.get("eligibility_version") != ELIGIBILITY_VERSION:
                log.warning(
                    "Precheck cache eligibility_version=%s != %s — eligible loop "
                    "set changed; discarding cache and re-running the pre-flight check.",
                    data.get("eligibility_version"), ELIGIBILITY_VERSION,
                )
                raise ValueError("stale eligibility_version")
            eligible_names = set(data["eligible"])
            loop_counts: dict[str, int] = data.get("loop_counts", {})
            loop_records_map: dict[str, list[dict]] = data.get("loop_records", {})
            normalizer = FeatureNormalizer.from_state_dict(data.get("normalizer", {}))
            result = [b for b in benchmarks if b.name in eligible_names]
            log.info(
                "Pre-flight check skipped — loaded %d eligible benchmarks "
                "from cache (%s)%s",
                len(result), cache_file,
                " [normalizer loaded]" if normalizer._fitted else " [no normalizer in cache — will be identity]",
            )
            # Dedup applies at load time too, so pre-dedup caches keep working.
            loop_records_map, dropped = _dedup_loop_records(loop_records_map)
            loop_counts = {k: len(v) for k, v in loop_records_map.items()}
            if dropped:
                log.info(
                    "Feature dedup: dropped %d duplicate loops, %d remain",
                    dropped, sum(loop_counts.values()),
                )
            return result, loop_counts, loop_records_map, normalizer
        except Exception as e:
            log.warning("Could not read precheck cache (%s): %s — running check", cache_file, e)

    if skip:
        log.info("--skip-precheck set but no cache found — running pre-flight check anyway")

    log.info("Pre-flight check: testing %d benchmarks for eligible loops...", len(benchmarks))

    eligible: list[Path] = []
    loop_counts: dict[str, int] = {}
    loop_records_map: dict[str, list[dict]] = {}
    excluded: list[tuple[str, str]] = []
    all_feature_tensors = []

    for b in benchmarks:
        try:
            file_map, _, triple = get_loop_features(b)
            n = sum(len(df) for df in file_map.values())
            if n > 0:
                eligible.append(b)
                loop_counts[b.name] = n
                # Collect raw (un-normalised) feature tensors for fitting
                # and store per-loop records for loop-level worker distribution.
                records: list[dict] = []
                for filename, df in file_map.items():
                    for _, row in df.iterrows():
                        raw = _row_to_tensor(row)
                        all_feature_tensors.append(raw)
                        # kernelParents is a '|'-separated string of mangled names;
                        # split into a list (empty string → empty list).
                        kp_raw = str(row.get("kernelParents", "")).strip()
                        kernel_parents = [p for p in kp_raw.split("|") if p]
                        records.append({
                            "loop_idx":           int(row["loopIdx"]),
                            "filename":           filename,
                            "triple":             triple,
                            "pre_features_raw":   raw.tolist(),
                            "is_kernel_function": bool(int(row.get("isKernelFunction", 0))),
                            "kernel_parents":     kernel_parents,
                        })
                loop_records_map[b.name] = records
                log.info("  PASS  %-35s  eligible_loops=%d", b.name, n)
            else:
                reason = "0 eligible loops after filtering"
                excluded.append((b.name, reason))
                log.warning("  SKIP  %-35s  %s", b.name, reason)
        except Exception as e:
            reason = str(e)
            excluded.append((b.name, reason))
            log.warning("  SKIP  %-35s  %s", b.name, reason)

    log.info(
        "Pre-flight complete: %d eligible, %d excluded",
        len(eligible), len(excluded),
    )
    if excluded:
        log.info("Excluded benchmarks:")
        for name, reason in excluded:
            log.info("  %-35s  %s", name, reason)

    # Dedup BEFORE fitting the normalizer so a benchmark with 180 identical
    # template-instantiation loops doesn't skew the feature statistics.
    loop_records_map, dropped = _dedup_loop_records(loop_records_map)
    loop_counts = {k: len(v) for k, v in loop_records_map.items()}
    if dropped:
        log.info(
            "Feature dedup: dropped %d duplicate loops, %d remain",
            dropped, sum(loop_counts.values()),
        )
    all_feature_tensors = [
        torch.tensor(rec["pre_features_raw"], dtype=torch.float32)
        for records in loop_records_map.values()
        for rec in records
    ]

    # Fit normalizer on the deduped loop feature vectors
    normalizer = FeatureNormalizer()
    normalizer.fit(all_feature_tensors)
    log.info(
        "Normalizer fitted on %d loop feature vectors",
        len(all_feature_tensors),
    )
    if normalizer._fitted:
        log.info("  mean: %s", [round(v, 4) for v in normalizer.mean.tolist()])
        log.info("  std:  %s", [round(v, 4) for v in normalizer.std.tolist()])

    # --- Save cache ---
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps({
            "checked_at": datetime.now().isoformat(),
            "features_version": FEATURES_VERSION,
            "eligibility_version": ELIGIBILITY_VERSION,
            "eligible": [b.name for b in eligible],
            "loop_counts": loop_counts,
            "loop_records": loop_records_map,
            "normalizer": normalizer.state_dict(),
            "excluded": [{"name": n, "reason": r} for n, r in excluded],
        }, indent=2))
        log.info("Pre-flight cache saved: %s", cache_file)
    except Exception as e:
        log.warning("Could not save precheck cache: %s", e)

    return eligible, loop_counts, loop_records_map, normalizer


# ---------------------------------------------------------------------------
# One-shot baseline measurement
# ---------------------------------------------------------------------------

def measure_baselines(
    benchmarks: list[Path],
    loop_records_map: dict[str, list[dict]],
    arch: str,
    n_runs: int,
    nsys_timeout: int,
    tmp_dir: Path,
    gpu_id: int = 0,
    cache_file: "Path | None" = None,
) -> dict[str, dict]:
    """
    Compile and measure baseline kernel times for each benchmark once.

    Returns a cache dict keyed by benchmark name:
        {
          "total_ms":      float,           # sum of all kernels (B2 / fallback)
          "per_kernel_ms": {                # demangled parent kernel → ms
              "mandel(int *, ...)": 5995.4,
          }
        }

    per_kernel_ms is built by collecting all unique kernelParents values from
    the benchmark's loop records, demangling each, and filtering the nsys output
    to isolate that kernel's time.  Cases A and B1 (single kernel parent) use
    per_kernel_ms; Case B2 (multiple parents) falls back to total_ms.

    A benchmark is skipped if compilation or nsys measurement fails; workers
    fall back to on-demand measurement via GpuLoopEnv.reset() on a cache miss.

    If *cache_file* is given, previously measured baselines are loaded from it
    (skipping re-measurement — ~3h for the full HeCBench set) and the merged
    result is saved back.  The file is arch-tagged; a mismatched arch ignores
    the cache.  Baseline stability across restarts also keeps the cross-epoch
    reward cache consistent: cached rewards were computed against these
    exact baseline values.
    """
    from hecbench import compile_baseline, demangle, demangled_to_filter, measure_kernel_time, _parse_nsys_kernel_times, _sum_kernel_times, _get_run_command
    import tempfile as _tempfile

    cache: dict[str, dict] = {}
    if cache_file is not None and Path(cache_file).exists():
        try:
            saved = json.loads(Path(cache_file).read_text())
            if saved.get("arch") == arch:
                cache.update(saved.get("baselines", {}))
                log.info(
                    "Loaded %d baselines from cache (%s) — measuring only the rest",
                    len(cache), cache_file,
                )
            else:
                log.warning(
                    "Baseline cache arch mismatch (%s != %s) — re-measuring all",
                    saved.get("arch"), arch,
                )
        except Exception as e:
            log.warning("Could not read baseline cache (%s): %s", cache_file, e)

    benchmarks = [b for b in benchmarks if b.name not in cache]
    log.info("Measuring baselines for %d benchmarks (once per run)...", len(benchmarks))

    tmp_dir.mkdir(parents=True, exist_ok=True)
    env_base = {**__import__("os").environ, "ARCH": arch, "CUDA_VISIBLE_DEVICES": str(gpu_id)}

    for b in benchmarks:
        if not compile_baseline(b, arch=arch):
            log.warning("  SKIP  %-35s  baseline compile failed", b.name)
            continue

        # Collect unique mangled kernel parent names for this benchmark's loops
        unique_parents: set[str] = set()
        for rec in loop_records_map.get(b.name, []):
            for p in rec.get("kernel_parents", []):
                if p:
                    unique_parents.add(p)

        # Run nsys once, parse the full kernel-time dict.
        # A per-run TimeoutExpired must not propagate: one slow benchmark
        # would otherwise abort the entire baseline pass (and the job) —
        # skip the run, and skip the benchmark if no run succeeds.  Workers
        # already skip benchmarks that have no baseline cache entry.
        _subprocess = __import__("subprocess")
        run_cmd = _get_run_command(b, arch)
        report_path = _tempfile.mktemp(prefix="nsys_bl_", dir=str(tmp_dir))
        run_times_raw: list[dict] = []
        timed_out = 0
        for _ in range(n_runs):
            try:
                _subprocess.run(
                    f"nsys profile --trace=cuda --sample=none --cpuctxsw=none "
                    f"--output={report_path} --force-overwrite=true {run_cmd}",
                    cwd=b, shell=True, capture_output=True, text=True,
                    timeout=nsys_timeout, env=env_base,
                )
                stats = _subprocess.run(
                    f"nsys stats --report=cuda_gpu_kern_sum --format=csv {report_path}.nsys-rep",
                    shell=True, capture_output=True, text=True, timeout=30, env=env_base,
                )
            except _subprocess.TimeoutExpired:
                timed_out += 1
                continue
            kt = _parse_nsys_kernel_times(stats.stdout + stats.stderr)
            if kt:
                run_times_raw.append(kt)

        if timed_out:
            log.warning(
                "  WARN  %-35s  %d/%d baseline nsys runs timed out (>%ds) — "
                "raise --nsys-timeout to include this benchmark reliably",
                b.name, timed_out, n_runs, nsys_timeout,
            )

        if not run_times_raw:
            log.warning("  SKIP  %-35s  nsys produced no output", b.name)
            continue

        # Median total time across runs — more robust than mean against
        # scheduling outliers and nsys warm-up effects.
        total_ms = statistics.median(
            sum(kt.values()) for kt in run_times_raw
        )

        # Per-kernel medians: for each unique parent, take the median of the
        # filtered time across runs.  Median is more robust than mean here.
        # Key in per_kernel_ms is "funcname(" (via demangled_to_filter) so it
        # matches regardless of how c++filt vs nsys format pointer/const tokens.
        per_kernel_ms: dict[str, float] = {}
        for mangled in unique_parents:
            nsys_filter = demangled_to_filter(demangle(mangled))
            run_vals = [
                _sum_kernel_times(kt, nsys_filter)
                for kt in run_times_raw
            ]
            valid = [v for v in run_vals if v is not None]
            if valid:
                per_kernel_ms[nsys_filter] = statistics.median(valid)
                log.info(
                    "  DONE  %-35s  kernel=%-50s  %.3f ms",
                    b.name, nsys_filter, per_kernel_ms[nsys_filter],
                )
            else:
                log.warning(
                    "  WARN  %-35s  kernel filter %r not found in nsys output "
                    "(mangled: %r) — B2 fallback will apply",
                    b.name, nsys_filter, mangled,
                )

        cache[b.name] = {"total_ms": total_ms, "per_kernel_ms": per_kernel_ms}
        log.info(
            "  DONE  %-35s  total=%.3f ms  kernels_cached=%d",
            b.name, total_ms, len(per_kernel_ms),
        )

    # Accounting: len(benchmarks) is what was ATTEMPTED this call (the ones not
    # already cached), not what succeeded.  Report both — a benchmark without a
    # baseline is silently skipped by every worker in every epoch, so a failure
    # here permanently removes its loops from training.
    attempted = len(benchmarks)
    succeeded = sum(1 for b in benchmarks if b.name in cache)
    failed    = attempted - succeeded
    log.info(
        "Baseline cache: %d benchmarks total (attempted %d, succeeded %d, failed %d)",
        len(cache), attempted, succeeded, failed,
    )
    if failed:
        _lost = [b.name for b in benchmarks if b.name not in cache]
        log.warning(
            "%d benchmarks have NO baseline and will be skipped in EVERY epoch "
            "(their loops never become training samples): %s",
            failed, _lost,
        )

    if cache_file is not None:
        try:
            Path(cache_file).write_text(json.dumps(
                {"arch": arch, "n_runs": n_runs, "baselines": cache}, indent=2,
            ))
            log.info("Baseline cache saved: %s", cache_file)
        except Exception as e:
            log.warning("Could not save baseline cache: %s", e)

    return cache


# ---------------------------------------------------------------------------
# Benchmark split
# ---------------------------------------------------------------------------

def split_benchmarks(
    benchmarks: list,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list, list, list]:
    """Randomly split benchmarks into (train, val, test) by application."""
    rng = random.Random(seed)
    shuffled = list(benchmarks)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test = max(1, round(n * test_ratio))
    n_val = max(1, round(n * val_ratio))
    n_train = max(1, n - n_val - n_test)
    n_val = max(0, n - n_train - n_test)
    train = shuffled[:n_train]
    val   = shuffled[n_train:n_train + n_val]
    test  = shuffled[n_train + n_val:]
    return train, val, test


# ---------------------------------------------------------------------------
# Evaluation (validation / test) — no gradient updates
# ---------------------------------------------------------------------------

def evaluate(
    agent: Agent,
    env: GpuLoopEnv,
    benchmarks: list[Path],
    device: torch.device,
    label: str = "val",
    greedy: bool = True,
) -> tuple[dict, list[Path]]:
    """
    Run the current policy over *benchmarks* without any gradient updates.

    greedy=True (default) reports the deployment-mode argmax policy; greedy=False
    samples from the policy distribution (matches the noisier training-time draw).

    Returns:
      (metrics_dict, failed_benchmarks)

    failed_benchmarks contains any benchmark whose reset() failed — the
    caller should remove these from future evaluation passes.
    """
    all_rewards: list[float] = []
    all_advantages: list[float] = []
    per_benchmark: list[dict] = []
    failed: list[Path] = []
    failures: list[dict] = []   # compile failures, reported separately
    samples = 0
    missed = 0

    for benchmark_dir in benchmarks:
        bmark_rewards: list[float] = []
        # Per-benchmark outcome tally.  reward==0.0 is ambiguous on its own —
        # it means "no-op", "compile failed", or "measurement failed" — so read
        # env.last_status to separate a genuine neutral from a no-signal loop.
        bmark_status: dict[str, int] = {}

        try:
            first_features = env.reset(benchmark_dir)
        except Exception as e:
            log.warning(
                "[%s] reset failed for %s — removing from future %s passes: %s",
                label, benchmark_dir.name, label, e,
            )
            failed.append(benchmark_dir)
            continue

        if first_features is None:
            log.info("[%s] %s — no eligible loops, skipping", label, benchmark_dir.name)
            continue

        for loop_record in env.eligible_loops:
            pre_features = loop_record.pre_features.to(device)

            # Greedy (argmax) by default: reports the deployment-mode decision,
            # not a sample from the exploration distribution.
            unmerge, _ = agent.select_unmerge(pre_features, greedy=greedy)

            # Study A action space is {no-op, unroll-only, unmerge(+unroll)}:
            #   unmerge==1            → post-unmerge features → factor → full UU
            #   unmerge==0, factor>1  → unroll-only on the un-unmerged loop
            #   unmerge==0, factor==1 → pure no-op (env.step fast path)
            # The FactorActor is consulted on both branches; only its input
            # state differs (post-unmerge vs. pre-unmerge features).  Trip count
            # is invariant under unmerge, so the pre-features mask is valid here.
            if unmerge == 1:
                try:
                    step2_features = env.get_post_unmerge_features(loop_record).to(device)
                except Exception:
                    step2_features = pre_features
            else:
                step2_features = pre_features
            factor_idx, _, _ = agent.select_factor(
                step2_features,
                trip_known=loop_record.trip_count_known,
                trip_count=loop_record.trip_count,
                loop_idx=loop_record.loop_idx,
                greedy=greedy,
            )

            try:
                _, reward, done = env.step(loop_record, unmerge, factor_idx)
            except Exception as e:
                log.warning(
                    "[%s] step failed for loop_idx=%d: %s",
                    label, loop_record.loop_idx, e,
                )
                missed += 1
                continue

            status = getattr(env, "last_status", "ok")
            bmark_status[status] = bmark_status.get(status, 0) + 1

            # Excluded from the REPORTED statistics (training still saw the
            # penalty — see the methodology note):
            #   measure_failed  — infrastructure, not a result
            #   compile_failed  — a transform-coverage limitation, studied
            #                     separately via the CPU-only remark analysis
            if status in ("measure_failed", "compile_failed"):
                if status == "compile_failed":
                    failures.append({
                        "benchmark": benchmark_dir.name,
                        "loop_idx":  loop_record.loop_idx,
                        "unmerge":   unmerge,
                        "factor":    FACTOR_VALUES[factor_idx],
                        "filename":  loop_record.filename,
                        "triple":    loop_record.triple,
                        "error_signature": getattr(env, "last_error", ""),
                    })
                else:
                    missed += 1
                if done:
                    break
                continue

            v = agent.predict_value(pre_features)
            log.info(
                "  [%s] %s loop_idx=%d unmerge=%d factor=%d "
                "reward=%.4f V(s)=%.4f%s",
                label, benchmark_dir.name, loop_record.loop_idx,
                unmerge, FACTOR_VALUES[factor_idx], reward, v,
                "" if status in ("ok", "noop") else f" [{status.upper()}]",
            )

            all_rewards.append(reward)
            all_advantages.append(reward - v)
            bmark_rewards.append(reward)
            samples += 1

            if done:
                break

        if bmark_rewards:
            # no_signal = loops excluded from the statistics because the
            # toolchain failed to measure them.  NOTE: compile failures are NOT
            # no-signal any more — they are scored with an explicit penalty, so
            # they are a real (negative) result.  Only measurement failures,
            # which are dropped, count as no-signal.
            n_fail    = bmark_status.get("compile_failed", 0)
            n_timeout = bmark_status.get("compile_timeout", 0)
            n_measfail = bmark_status.get("measure_failed", 0)
            per_benchmark.append({
                "benchmark":   benchmark_dir.name,
                "loops":       len(bmark_rewards),
                "avg_reward":  sum(bmark_rewards) / len(bmark_rewards),
                "min_reward":  min(bmark_rewards),
                "max_reward":  max(bmark_rewards),
                # ±1% classification thresholds — below that is measurement noise
                "loops_win":        sum(1 for r in bmark_rewards if r > 0.01),
                "loops_regression": sum(1 for r in bmark_rewards if r < -0.01),
                "loops_noop":       bmark_status.get("noop", 0),
                "compile_failed":   n_fail,
                "compile_timeout":  n_timeout,
                "measure_failed":   n_measfail,
                "no_signal":        n_measfail,   # compile failures are scored, not no-signal
            })
            if n_fail or n_timeout or n_measfail:
                log.warning(
                    "[%s] %-30s %d/%d loops gave NO SIGNAL "
                    "(compile_failed=%d, compile_timeout=%d, measure_failed=%d)",
                    label, benchmark_dir.name, n_fail + n_timeout + n_measfail,
                    len(bmark_rewards), n_fail, n_timeout, n_measfail,
                )

    avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
    avg_adv    = sum(all_advantages) / len(all_advantages) if all_advantages else 0.0

    metrics = {
        f"{label}_avg_reward":    avg_reward,
        f"{label}_avg_advantage": avg_adv,
        f"{label}_samples":       samples,
        f"{label}_missed":        missed,
        f"{label}_per_benchmark": per_benchmark,
        f"{label}_compile_failures": failures,
    }
    return metrics, failed


# ---------------------------------------------------------------------------
# Metrics CSV
# ---------------------------------------------------------------------------

def append_metrics(metrics_file: str, row: dict) -> None:
    """Append one row to the metrics CSV, writing the header if the file is new."""
    p = Path(metrics_file)
    write_header = not p.exists()
    csv_row = {k: v for k, v in row.items() if not isinstance(v, list)}
    with open(p, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(csv_row)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_training_curves(metrics_file: str, output_dir: str) -> None:
    """Read metrics CSV and save training/validation curve plots."""
    try:
        import pandas as pd
        df = pd.read_csv(metrics_file)
        df = df[pd.to_numeric(df["epoch"], errors="coerce").notna()].copy()
        df["epoch"] = df["epoch"].astype(int)
    except Exception as e:
        log.warning("Could not generate plots: %s", e)
        return

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    ax = axes[0]
    ax.plot(df["epoch"], df["train_actor_loss"], label="actor_loss")
    ax.plot(df["epoch"], df["train_value_loss"], label="value_loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Training Loss"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    if "train_entropy" in df.columns:
        ax.plot(df["epoch"], df["train_entropy"], label="entropy", color="purple")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Entropy")
    ax.set_title("Policy Entropy (higher = more exploratory)")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(df["epoch"], df["train_avg_reward"], label="train")
    if "val_avg_reward" in df.columns:
        ax.plot(df["epoch"], df["val_avg_reward"], label="val", linestyle="--")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Avg Reward")
    ax.set_title("Average Reward (higher = better)"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[3]
    ax.plot(df["epoch"], df["train_avg_advantage"], label="train")
    if "val_avg_advantage" in df.columns:
        ax.plot(df["epoch"], df["val_avg_advantage"], label="val", linestyle="--")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Avg Advantage")
    ax.set_title("Average Advantage (reward − V(s))"); ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path(output_dir) / "training_curves.png"
    plt.savefig(str(out), dpi=150)
    plt.close()
    log.info("Training curves saved: %s", out)


# ---------------------------------------------------------------------------
# Parallel training helpers
# ---------------------------------------------------------------------------

def build_loop_assignments(
    benchmarks: list[Path],
    loop_records_map: dict[str, list[dict]],
) -> list[dict]:
    """
    Build a flat list of loop assignment dicts for a set of benchmarks.

    Each dict contains everything a worker needs to process one loop without
    calling env.reset() or consulting the original source tree:
        benchmark_name    — for baseline_cache lookup and working-set naming
        benchmark_path    — original source directory (str, for shutil.copytree)
        loop_idx          — index passed to compile_single_loop
        filename          — source file containing the loop
        triple            — target triple (e.g. "nvptx64-nvidia-cuda")
        pre_features_raw  — un-normalised feature vector as list[float]
    """
    assignments: list[dict] = []
    for b in benchmarks:
        for record in loop_records_map.get(b.name, []):
            assignments.append({
                "benchmark_name":    b.name,
                "benchmark_path":    str(b),
                "loop_idx":          record["loop_idx"],
                "filename":          record["filename"],
                "triple":            record["triple"],
                "pre_features_raw":  record["pre_features_raw"],
                "is_kernel_function": record.get("is_kernel_function", True),
                "kernel_parents":    record.get("kernel_parents", []),
            })
    return assignments


def assign_loops_to_workers(
    loop_assignments: list[dict],
    n_workers: int,
) -> list[list[dict]]:
    """
    Greedy min-heap bin-packing at loop granularity.

    Each loop is one unit of work.  The input list is shuffled before packing
    so different epochs get different distributions (exploration diversity).
    Within each worker's share the assignments are sorted by
    (benchmark_name, loop_idx) so each benchmark's loops are contiguous —
    the worker copies a benchmark directory once and processes all its assigned
    loops before moving on.
    """
    # heap entries: (loops_assigned, worker_idx, assignment_list)
    heap: list[tuple[int, int, list]] = [(0, i, []) for i in range(n_workers)]
    heapq.heapify(heap)

    for loop in loop_assignments:
        total, idx, lst = heapq.heappop(heap)
        lst.append(loop)
        heapq.heappush(heap, (total + 1, idx, lst))

    # Sort each worker's share so benchmark groups are contiguous
    per_worker: list[list[dict]] = [[] for _ in range(n_workers)]
    for total, idx, lst in heap:
        lst.sort(key=lambda x: (x["benchmark_name"], x["loop_idx"]))
        per_worker[idx] = lst
    return per_worker


def _get_weights(agent: Agent) -> dict:
    """Snapshot all network weights as CPU state_dicts (picklable)."""
    return {
        "unmerge_actor": {k: v.cpu().clone() for k, v in agent.unmerge_actor.state_dict().items()},
        "factor_actor":  {k: v.cpu().clone() for k, v in agent.factor_actor.state_dict().items()},
        "critic":        {k: v.cpu().clone() for k, v in agent.critic.state_dict().items()},
    }


def _load_weights(agent: Agent, weights: dict) -> None:
    """Load a weight snapshot broadcast from main into a worker's agent."""
    agent.unmerge_actor.load_state_dict(weights["unmerge_actor"])
    agent.factor_actor.load_state_dict(weights["factor_actor"])
    agent.critic.load_state_dict(weights["critic"])


# ---------------------------------------------------------------------------
# Worker process (module-level so it is picklable by multiprocessing)
# ---------------------------------------------------------------------------

def _worker_fn(
    rank: int,
    gpu_id: int,
    loop_assignments: list,      # list[dict] — flat, sorted by (benchmark_name, loop_idx)
    initial_weights: dict,
    hparams: dict,
    result_q,                    # mp.Queue: worker → main
    weight_q,                    # mp.Queue: main → worker
    mode: str,                   # "train" or "eval"
) -> None:
    """
    Worker process: iterates over *loop_assignments* and streams result dicts
    to *result_q*.

    Each benchmark is copied once to an isolated working directory:
        tmp_dir / "working_set" / benchmark_name
    so compilations for different workers targeting the same benchmark never
    conflict and the original HeCBench source tree is never modified.

    env.reset() and env.step() are NOT called here.  The worker drives
    compile_single_loop and measure_kernel_time directly, using pre-measured
    baseline times from hparams["baseline_cache"] and pre-extracted loop
    features from the assignment dicts.  GpuLoopEnv is instantiated only to
    provide get_post_unmerge_features() for the unmerge=1 path.

    Message types sent to result_q:
      {"type": "entry",       "entry": RolloutEntry, "benchmark": str,
       "loop_idx": int, "unmerge": int, "factor": int,
       "reward": float, "value": float, "timeout": bool}  — training sample
      {"type": "eval_result", "benchmark": str, "loop_idx": int,
       "reward": float, "value": float, "timeout": bool}  — eval sample
      {"type": "step_failed", "loop_idx": int, "rank": int}
      {"type": "worker_done", "rank": int}

    Weight updates (train mode only): main puts a weight dict into *weight_q*
    after each PPO update; the worker drains it between benchmark groups.
    """
    import logging
    import os
    import shutil
    import subprocess
    import sys
    from itertools import groupby
    from pathlib import Path as _Path

    # Re-insert scripts/rl into sys.path (spawn starts fresh)
    _here = _Path(__file__).parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))

    import torch
    from agent import (
        Agent, BanditAgent, RolloutEntry, FACTOR_VALUES,
        _IDX_TRIP_COUNT_KNOWN, _IDX_TRIP_COUNT,
    )
    from environment import GpuLoopEnv, LoopRecord
    from hecbench import FeatureNormalizer, compile_single_loop_ex, demangle, demangled_to_filter, measure_kernel_time

    _epoch = hparams.get("epoch", 0)
    _total = hparams.get("total_epochs", 0)
    _epoch_tag = f"[{_epoch}/{_total}] " if _total > 0 else ""
    _log = logging.getLogger(f"worker.{rank}")
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s %(levelname)s {_epoch_tag}[W{rank}] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Everything below runs inside the try: a setup failure (CUDA init,
    # bad normalizer state, unwritable tmp dir) would otherwise kill this
    # process with neither worker_crashed nor worker_done on the queue,
    # leaving main to discover it only via the idle timeout.
    try:
        # ------------------------------------------------------------------
        # GPU assignment: set CUDA_VISIBLE_DEVICES BEFORE any CUDA call so
        # PyTorch maps device 0 → physical GPU gpu_id in this process.
        # ------------------------------------------------------------------
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Agent type mirrors main's --agent flag; both classes expose the same
        # interface (select_*, predict_value, ppo_update) and net shapes, so
        # everything downstream — weight broadcast, buffer, queue — is agnostic.
        _common = dict(
            clip_eps=hparams["clip_eps"],
            K=hparams["K"],
            batch_size=hparams["batch_size"],
            lr=hparams["lr"],
            value_loss_coef=hparams["value_loss_coef"],
            entropy_coef=hparams["entropy_coef"],
            weight_decay=hparams["weight_decay"],
            max_grad_norm=hparams["max_grad_norm"],
            device=device,
        )
        if hparams.get("agent_type") == "bandit":
            agent = BanditAgent(epsilon=hparams.get("bandit_epsilon", 0.0),
                                **_common)
        else:
            agent = Agent(**_common)
        _load_weights(agent, initial_weights)

        worker_normalizer = FeatureNormalizer.from_state_dict(
            hparams.get("normalizer_state", {})
        )
        baseline_cache: dict = hparams.get("baseline_cache", {})
        # Cross-epoch caches (read-only snapshots; main merges new results and
        # persists them).  The environment is deterministic — same (loop, action)
        # → same binary → same reward — so a hit replaces compile + measure.
        reward_cache: dict = hparams.get("reward_cache", {})
        postf_cache: dict = hparams.get("postf_cache", {})

        # Per-worker directories
        tmp_dir = _Path(hparams["tmp_dir"]) / f"worker_{rank}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        working_set_dir = tmp_dir / "working_set"
        working_set_dir.mkdir(parents=True, exist_ok=True)

        # Lightweight env used only for get_post_unmerge_features().
        # _benchmark_dir is set per benchmark group below.
        env = GpuLoopEnv(
            arch=hparams["arch"],
            n_runs=hparams["n_runs"],
            nsys_timeout=hparams["nsys_timeout"],
            tmp_dir=tmp_dir,
            compile_timeout_penalty=hparams["compile_timeout_penalty"],
            # This env is used only for get_post_unmerge_features() — the worker
            # scores rewards inline — but keep the penalties in sync with the
            # inline path so an env.step() added here later cannot silently use
            # different reward semantics from the rest of the run.
            compile_failure_penalty=hparams["compile_failure_penalty"],
            reward_deadzone=hparams.get("reward_deadzone", 0.0),
            gpu_id=gpu_id,
            normalizer=worker_normalizer,
            baseline_cache=baseline_cache,
        )

        # loop_assignments is sorted by (benchmark_name, loop_idx) so groupby
        # yields contiguous benchmark groups — one shutil.copytree per benchmark.
        for bench_name, bench_iter in groupby(
            loop_assignments, key=lambda x: x["benchmark_name"]
        ):
            bench_loops = list(bench_iter)

            # ----------------------------------------------------------
            # Between benchmark groups: absorb weight updates from main
            # (train mode only — eval uses a frozen policy snapshot)
            # ----------------------------------------------------------
            if mode == "train":
                try:
                    while True:
                        new_weights = weight_q.get_nowait()
                        _load_weights(agent, new_weights)
                        _log.debug("Weights updated from main")
                except Exception:
                    pass  # queue.Empty — no update pending

            # ----------------------------------------------------------
            # Validate baseline and copy benchmark to isolated working dir
            # ----------------------------------------------------------
            baseline_ms = baseline_cache.get(bench_name)
            if baseline_ms is None:
                _log.warning(
                    "No baseline cached for %s — skipping %d loops",
                    bench_name, len(bench_loops),
                )
                continue

            original_path = _Path(bench_loops[0]["benchmark_path"])
            copy_dir = working_set_dir / bench_name
            try:
                if copy_dir.exists():
                    shutil.rmtree(copy_dir)
                shutil.copytree(original_path, copy_dir)
            except Exception as e:
                _log.warning(
                    "Failed to copy %s to working set: %s — skipping",
                    bench_name, e,
                )
                continue

            # Point env at the worker's copy so get_post_unmerge_features
            # compiles inside the isolated directory.
            env._benchmark_dir = copy_dir

            _log.info("Processing %s: %d loops", bench_name, len(bench_loops))

            # ----------------------------------------------------------
            # Per-loop: compile, measure, send result
            # ----------------------------------------------------------
            for loop_data in bench_loops:
                loop_idx = loop_data["loop_idx"]
                filename  = loop_data["filename"]
                triple    = loop_data["triple"]

                raw_features = torch.tensor(
                    loop_data["pre_features_raw"], dtype=torch.float32
                )
                pre_features = worker_normalizer.normalize(raw_features).to(device)
                kernel_parents = loop_data.get("kernel_parents", [])

                # RAW trip-count values for factor masking — must come from the
                # un-normalised tensor; pre_features is z-scored and the trip
                # count cannot be recovered from it.
                trip_known = raw_features[_IDX_TRIP_COUNT_KNOWN].item() > 0.5
                trip_count = int(raw_features[_IDX_TRIP_COUNT].item())

                loop_record = LoopRecord(
                    loop_idx=loop_idx,
                    filename=filename,
                    triple=triple,
                    pre_features=pre_features.cpu(),
                    kernel_parents=kernel_parents,
                    trip_count_known=trip_known,
                    trip_count=trip_count,
                )

                # Resolve kernel filter and baseline for this loop as a COUPLED
                # pair whose measurement scope is guaranteed symmetric — both
                # per-kernel, or both total.  Mirrors GpuLoopEnv._resolve_measurement.
                #
                # Cases A / B1: single parent → filter nsys to that kernel + use
                # per-kernel baseline.
                # Case B2 / no parents / per-kernel cache MISS: no filter → total
                # benchmark time on BOTH sides.
                #
                # Two traps avoided here:
                #   1. Python falsy-0.0: a per-kernel time of 0.0 must not fall
                #      through to total — hence `is not None`, not `or`.
                #   2. Cache-miss asymmetry: if the per-kernel baseline is absent,
                #      baseline would be total while modified still measured with
                #      the per-kernel filter → asymmetric (baseline=total,
                #      modified=per-kernel) comparison that corrupts the reward.
                #      Fix: on a miss force kernel_filter=None too, so the modified
                #      measurement also falls back to total.
                kernel_filter = None
                baseline_ms = baseline_cache.get(bench_name, {}).get("total_ms", 0.0)
                if len(kernel_parents) == 1:
                    _kf = demangled_to_filter(demangle(kernel_parents[0]))
                    _per_kern = (
                        baseline_cache.get(bench_name, {})
                        .get("per_kernel_ms", {})
                        .get(_kf)
                    )
                    if _per_kern is not None:
                        kernel_filter = _kf
                        baseline_ms = _per_kern
                    # else: leave (None, total_ms) — both sides total.

                # --- Agent decisions ---
                # Eval mode uses the greedy (argmax) policy: this is the
                # deployment-mode measurement, free of sampling noise.
                greedy = (mode == "eval")
                unmerge, log_p1 = agent.select_unmerge(pre_features, greedy=greedy)

                # Study A action space is {no-op, unroll-only, unmerge(+unroll)}:
                #   unmerge==1            → post-unmerge features → factor → full UU
                #   unmerge==0, factor>1  → unroll-only on the un-unmerged loop
                #   unmerge==0, factor==1 → pure no-op (free, no compile)
                # The FactorActor decides on BOTH the unmerge==1 and unmerge==0
                # branches; only its input state differs (post-unmerge vs.
                # pre-unmerge features).  Trip count is invariant under unmerge,
                # so the pre-features mask is valid for the unroll-only branch.
                if unmerge == 1:
                    # Post-unmerge features (deterministic per loop — cache hits
                    # skip the 2-compile feature extraction), then factor.
                    _pf = postf_cache.get(f"{bench_name}|{loop_idx}")
                    if _pf is not None:
                        step2_features = torch.tensor(
                            _pf, dtype=torch.float32
                        ).to(device)
                    else:
                        try:
                            step2_features = env.get_post_unmerge_features(
                                loop_record
                            ).to(device)
                        except Exception:
                            step2_features = pre_features
                else:
                    # Unroll-only / no-op: no unmerge compile, so the factor
                    # decision conditions on the loop's pre-unmerge features.
                    step2_features = pre_features

                factor_idx, log_p2, mask2 = agent.select_factor(
                    step2_features,
                    trip_known=trip_known,
                    trip_count=trip_count,
                    loop_idx=loop_idx,
                    greedy=greedy,
                )
                factor = FACTOR_VALUES[factor_idx]

                # Pure no-op (unmerge==0, factor==1): reward 0 by definition, no
                # compile.  The FactorActor is not trained on this sample
                # (factor_active=False) — there is no unroll decision to learn.
                if unmerge == 0 and factor == 1:
                    v = agent.predict_value(pre_features)
                    _send_loop_result(
                        result_q, mode, rank, bench_name, loop_idx,
                        0, 1, 0.0, v, False,
                        pre_features, step2_features, factor_idx,
                        log_p1, log_p2, mask2,
                        cached=False, factor_active=False,
                        status="noop", filename=filename, triple=triple,
                    )
                    continue

                # --- Compile + measure (unmerge, factor) ---
                is_timeout = False
                from_cache = False
                status = "ok"
                # Must be reset PER LOOP, not inside the cache-miss branch: on a
                # cache hit the compile is skipped, so a branch-local binding is
                # unbound on the first hit (UnboundLocalError kills the whole
                # worker) and stale on later ones (a previous loop's error
                # signature attached to an unrelated result).
                err_sig = ""
                _rc_key = f"{bench_name}|{loop_idx}|{unmerge}|{factor}"
                _rc_hit = reward_cache.get(_rc_key)
                if _rc_hit is not None:
                    # Cross-epoch cache hit: the action is still sampled fresh
                    # from the current policy (on-policy log-probs above);
                    # only the deterministic compile+measure is memoized.
                    # Cached failures (0.0) and timeout penalties (-1.0) are
                    # also hits — known-bad compiles are never re-paid.
                    reward = float(_rc_hit)
                    from_cache = True
                else:
                    try:
                        ok, err_sig = compile_single_loop_ex(
                            copy_dir,
                            loop_idx=loop_idx,
                            unmerge=unmerge,
                            factor=factor,
                            filename=filename,
                            triple=triple,
                            arch=hparams["arch"],
                        )
                    except subprocess.TimeoutExpired:
                        reward    = hparams["compile_timeout_penalty"]
                        is_timeout = True
                        _log.warning(
                            "%s loop_idx=%d unmerge=%d factor=%d — COMPILE TIMEOUT "
                            "(penalty=%.2f)",
                            bench_name, loop_idx, unmerge, factor, reward,
                        )
                        v = agent.predict_value(pre_features)
                        _send_loop_result(
                            result_q, mode, rank, bench_name, loop_idx,
                            unmerge, factor, reward, v, is_timeout,
                            pre_features, step2_features, factor_idx,
                            log_p1, log_p2, mask2, factor_active=True,
                            status="compile_timeout", error="compile timeout",
                            filename=filename, triple=triple,
                        )
                        reward_cache[_rc_key] = reward
                        continue

                    if not ok:
                        # Compile FAILED.  Previously scored 0.0, which is
                        # indistinguishable from a genuine no-effect transform —
                        # so the agent could never learn to avoid over-aggressive
                        # actions.  Now an explicit penalty: the action is
                        # unusable, and must rank below declining to transform.
                        status = "compile_failed"
                        reward = hparams["compile_failure_penalty"]
                        _log.warning(
                            "%s loop_idx=%d unmerge=%d factor=%d — COMPILE FAILED "
                            "(penalty=%.2f)",
                            bench_name, loop_idx, unmerge, factor, reward,
                        )
                    else:
                        try:
                            modified_ms = measure_kernel_time(
                                copy_dir,
                                arch=hparams["arch"],
                                n_runs=hparams["n_runs"],
                                nsys_timeout=hparams["nsys_timeout"],
                                tmp_dir=tmp_dir,
                                gpu_id=gpu_id,
                                kernel_filter=kernel_filter,
                            )
                        except RuntimeError:
                            # Measurement failure is INFRASTRUCTURE, not the
                            # action's fault — penalising would teach the agent
                            # to avoid a perfectly good action.  Drop the sample
                            # entirely and do NOT cache it (may be transient).
                            _log.warning(
                                "%s loop_idx=%d unmerge=%d factor=%d — MEASUREMENT "
                                "FAILED, sample dropped (not cached)",
                                bench_name, loop_idx, unmerge, factor,
                            )
                            result_q.put({"type": "step_failed",
                                          "benchmark": bench_name,
                                          "loop_idx": loop_idx, "rank": rank})
                            continue

                        # Clip at -1.0 (the timeout-penalty scale): a pathological
                        # slowdown (observed: -52 on wlcpow) would otherwise
                        # dominate the normalised advantages of its whole PPO
                        # buffer.  Upside is bounded at 1.0 by construction.
                        reward_raw = (baseline_ms - modified_ms) / max(baseline_ms, 1e-9)
                        reward = max(reward_raw, -1.0)
                        if reward_raw < -1.0:
                            _log.warning(
                                "%s loop_idx=%d reward %.2f clipped to -1.0",
                                bench_name, loop_idx, reward_raw,
                            )
                        # Deadzone: |r| below the measurement-noise floor is not
                        # signal.  Applied only to measured rewards — never to
                        # failure/timeout penalties.
                        _dz = hparams.get("reward_deadzone", 0.0)
                        if _dz > 0 and abs(reward) < _dz:
                            reward = 0.0

                    reward_cache[_rc_key] = reward

                v = agent.predict_value(pre_features)
                _send_loop_result(
                    result_q, mode, rank, bench_name, loop_idx,
                    unmerge, factor, reward, v, is_timeout,
                    pre_features, step2_features, factor_idx,
                    log_p1, log_p2, mask2, cached=from_cache,
                    factor_active=True, status=status, error=err_sig,
                    filename=filename, triple=triple,
                )

    except Exception as e:
        # A crash here forfeits the REST of this worker's shard for the epoch.
        # The traceback only reaches the worker's stderr; main would otherwise
        # see a clean worker_done and the loss would surface solely as a
        # sample-accounting shortfall.  Report it on the queue so it is loud.
        import traceback as _tb
        _log.error("Worker %d CRASHED — remaining loops in this shard are lost:"
                   "\n%s", rank, _tb.format_exc())
        result_q.put({"type": "worker_crashed", "rank": rank,
                      "error": f"{type(e).__name__}: {e}"})
    finally:
        result_q.put({"type": "worker_done", "rank": rank})


def _send_loop_result(
    result_q,
    mode: str,
    rank: int,
    bench_name: str,
    loop_idx: int,
    unmerge: int,
    factor: int,
    reward: float,
    value: float,
    is_timeout: bool,
    pre_features,
    step2_features,
    factor_idx: int,
    log_p1,
    log_p2,
    mask2=None,
    cached: bool = False,
    factor_active: bool = True,
    status: str = "ok",
    error: str = "",
    filename: str = "",
    triple: str = "",
) -> None:
    """
    Put one loop result onto result_q in the appropriate format.

    IMPORTANT: only plain Python types cross the queue — never torch tensors.
    `torch.multiprocessing`'s Queue passes tensors by SHARED MEMORY, and the
    backing store is released when the producing worker exits.  Once workers
    became fast (near-100% reward-cache hits) they finished and exited while
    main was still inside ppo_update, so main's queue.get() raised on unpickling
    and the sample was silently dropped — 15k of 30k samples lost in the
    2026-07 run.  Lists are pickled by value and have no such lifetime coupling.
    """
    if mode == "train":
        result_q.put({
            "type":      "entry",
            # RolloutEntry payload, as plain Python (rebuilt in main)
            "state1":        pre_features.detach().cpu().tolist(),
            "state2":        step2_features.detach().cpu().tolist(),
            "action1":       int(unmerge),
            "action2":       int(factor_idx),
            "log_prob1":     float(log_p1),
            "log_prob2":     float(log_p2),
            "mask2":         mask2.detach().cpu().tolist() if mask2 is not None else None,
            "factor_active": bool(factor_active),
            "benchmark": bench_name,
            "loop_idx":  loop_idx,
            "unmerge":   unmerge,
            "factor":    factor,
            "reward":    reward,
            "value":     value,
            "timeout":   is_timeout,
            "cached":    cached,
            "status":    status,
            "error":     error,
            # Needed to REPRODUCE this compile in the CPU-only analysis: the
            # UU pass is scoped by -uu-match-filename / -uu-match-targettriple,
            # so without them a loop index would match a same-numbered loop in
            # another TU, or on the host side. See analyze_compile_failures.py.
            "filename":  filename,
            "triple":    triple,
            "rank":      rank,
        })
    else:
        result_q.put({
            "type":      "eval_result",
            "benchmark": bench_name,
            "loop_idx":  loop_idx,
            "unmerge":   unmerge,
            "factor":    factor,
            "reward":    reward,
            "value":     value,
            "timeout":   is_timeout,
            "cached":    cached,
            "status":    status,
            "error":     error,
            # Needed to REPRODUCE this compile in the CPU-only analysis: the
            # UU pass is scoped by -uu-match-filename / -uu-match-targettriple,
            # so without them a loop index would match a same-numbered loop in
            # another TU, or on the host side. See analyze_compile_failures.py.
            "filename":  filename,
            "triple":    triple,
            "rank":      rank,
        })


def build_warm_start_entries(
    train_loop_assignments: list[dict],
    reward_cache: dict,
    postf_cache: dict,
    normalizer,
) -> "tuple[list[RolloutEntry], int, int]":
    """
    Turn the reward cache into a (state, action, reward) training set for the
    bandit agent's warm start.  Returns (entries, n_cached, n_anchors).

    Only TRAIN-split loops are included: the cache also holds val-harvested
    cells, and warming up on those would leak the val set into training and
    quietly bias best-checkpoint selection.

    Penalty cells (compile failure / timeout) are included deliberately —
    avoidance is part of what the Q-heads must learn.  A synthetic no-op
    anchor (reward 0 by definition) is added per loop so the factor==1 arm
    and the unmerge==0 row are grounded rather than left at their random
    initialisation during warm start.
    """
    by_loop = {
        f"{a['benchmark_name']}|{a['loop_idx']}": a
        for a in train_loop_assignments
    }
    zero = torch.tensor(0.0)
    entries: list[RolloutEntry] = []

    n_cached = 0
    for key, r in reward_cache.items():
        parts = key.split("|")
        if len(parts) != 4:
            continue
        bench, li, um, fac = parts
        la = by_loop.get(f"{bench}|{li}")
        if la is None:
            continue                      # val-split or no-longer-eligible loop
        um, fac = int(um), int(fac)
        if fac not in FACTOR_VALUES:
            continue
        s1 = normalizer.normalize(
            torch.tensor(la["pre_features_raw"], dtype=torch.float32))
        if um == 1:
            pf = postf_cache.get(f"{bench}|{li}")
            # Same fallback the live pipeline uses when post-unmerge feature
            # extraction is unavailable: condition on pre-unmerge features.
            s2 = (torch.tensor(pf, dtype=torch.float32)
                  if pf is not None else s1)
        else:
            s2 = s1
        entries.append(RolloutEntry(
            state1=s1, state2=s2, action1=um,
            action2=FACTOR_VALUES.index(fac),
            log_prob1=zero, log_prob2=zero,
            reward=float(r), mask2=None, factor_active=True,
        ))
        n_cached += 1

    noop_idx = FACTOR_VALUES.index(1)
    for a in train_loop_assignments:
        s1 = normalizer.normalize(
            torch.tensor(a["pre_features_raw"], dtype=torch.float32))
        entries.append(RolloutEntry(
            state1=s1, state2=s1, action1=0, action2=noop_idx,
            log_prob1=zero, log_prob2=zero,
            reward=0.0, mask2=None, factor_active=True,
        ))
    return entries, n_cached, len(train_loop_assignments)


def _bandit_epsilon(args, epoch: int, total_epochs: int) -> float:
    """Linear epsilon decay for --agent bandit, mirroring the entropy schedule."""
    if getattr(args, "agent", "ppo") != "bandit":
        return 0.0
    frac = (epoch - 1) / (total_epochs - 1) if total_epochs > 1 else 0.0
    return args.bandit_epsilon + frac * (
        args.bandit_epsilon_final - args.bandit_epsilon)


def run_parallel_eval(
    agent: Agent,
    loop_assignments: list[dict],
    normalizer: "FeatureNormalizer",
    baseline_cache: dict,
    n_workers: int,
    args,
    label: str = "test",
    reward_cache: "dict | None" = None,
    postf_cache: "dict | None" = None,
    use_reward_cache: bool = False,
) -> dict:
    """
    Multi-GPU evaluation of the current policy (greedy), mirroring the training
    pass's worker layout: loops are bin-packed across *n_workers*, worker k on
    GPU k.  Replaces the old single-GPU sequential `evaluate()` for the test set.

    use_reward_cache=False (default for TEST) forces a fresh compile+measure for
    every loop.  This matters: the cached path returns *frozen first
    measurements*, so evaluating against it measures "did the policy pick cells
    whose stored values are high", not out-of-distribution generalisation.

    Returns a metrics dict shaped like evaluate()'s, including per-benchmark
    stats with failure/no-signal counts.
    """
    hparams = {
        "arch":                    args.arch,
        "n_runs":                  args.n_runs,
        "nsys_timeout":            args.nsys_timeout,
        "tmp_dir":                 args.tmp_dir,
        "compile_timeout_penalty": args.compile_timeout_penalty,
        "compile_failure_penalty": args.compile_failure_penalty,
        "reward_deadzone":         args.reward_deadzone,
        "clip_eps":                args.clip_eps,
        "K":                       args.K,
        "batch_size":              args.batch_size,
        "lr":                      args.lr,
        "value_loss_coef":         args.value_loss_coef,
        "entropy_coef":            args.entropy_coef,
        "weight_decay":            args.weight_decay,
        "max_grad_norm":           args.max_grad_norm,
        "agent_type":              getattr(args, "agent", "ppo"),
        "bandit_epsilon":          0.0,   # eval is greedy — epsilon unused
        "normalizer_state":        normalizer.state_dict(),
        "baseline_cache":          baseline_cache,
        "reward_cache":            dict(reward_cache or {}) if use_reward_cache else {},
        "postf_cache":             dict(postf_cache or {}),
        "epoch":                   0,
        "total_epochs":            0,
    }
    per_worker = assign_loops_to_workers(loop_assignments, n_workers)
    weights = _get_weights(agent)
    rq: mp.Queue = mp.Queue()
    wqs: list[mp.Queue] = [mp.Queue() for _ in range(n_workers)]
    procs = []
    for rank in range(n_workers):
        p = mp.Process(target=_worker_fn,
                       args=(rank, rank, per_worker[rank], weights, hparams,
                             rq, wqs[rank], "eval"),
                       daemon=True)
        p.start()
        procs.append(p)

    msg_timeout = args.n_runs * args.nsys_timeout + 300
    by_bench: dict[str, list[float]] = {}
    status_by_bench: dict[str, dict] = {}
    rewards: list[float] = []
    failures: list[dict] = []
    samples = missed = done = 0

    while done < n_workers:
        try:
            msg = rq.get(timeout=msg_timeout)
        except queue.Empty:
            if not any(p.is_alive() for p in procs):
                log.error("[%s] all workers died", label)
                break
            continue
        except Exception as e:
            log.error("[%s] result message DROPPED (%s: %s)", label,
                      type(e).__name__, e)
            if not any(p.is_alive() for p in procs):
                break
            continue

        t = msg["type"]
        if t == "eval_result":
            b = msg["benchmark"]
            st = status_by_bench.setdefault(b, {})
            s = msg.get("status", "ok")
            st[s] = st.get(s, 0) + 1
            if s == "compile_failed":
                # Excluded from reported stats — a transform-coverage limit,
                # studied separately (CPU-only remark analysis), not a result.
                failures.append({
                    "benchmark": b, "loop_idx": msg["loop_idx"],
                    "unmerge": msg["unmerge"], "factor": msg["factor"],
                    "filename": msg.get("filename", ""),
                    "triple":   msg.get("triple", ""),
                    "error_signature": msg.get("error", ""),
                })
                continue
            by_bench.setdefault(b, []).append(msg["reward"])
            rewards.append(msg["reward"])
            samples += 1
            log.info("  [%s W%d] %s loop_idx=%d unmerge=%d factor=%d "
                     "reward=%+.4f%s",
                     label, msg["rank"], b, msg["loop_idx"], msg["unmerge"],
                     msg["factor"], msg["reward"],
                     "" if s in ("ok", "noop") else f" [{s.upper()}]")
        elif t == "step_failed":
            # A measurement failure never sends an eval_result, so attribute it
            # here — otherwise per-benchmark measure_failed/no_signal read 0
            # even when loops were dropped.
            missed += 1
            if msg.get("benchmark"):
                _st = status_by_bench.setdefault(msg["benchmark"], {})
                _st["measure_failed"] = _st.get("measure_failed", 0) + 1
        elif t == "worker_crashed":
            log.error("[%s] Worker %d crashed (%s) — its remaining loops are "
                      "missing from these results", label, msg["rank"],
                      msg.get("error"))
        elif t == "worker_done":
            done += 1

    for p in procs:
        p.join(timeout=30)

    per_benchmark = []
    for b in sorted(set(by_bench) | set(status_by_bench)):
        rs = by_bench.get(b)
        if not rs:
            # every loop in this benchmark failed to compile — no measurable
            # result; recorded in compile_failures.csv, omitted from the table
            continue
        st = status_by_bench.get(b, {})
        n_fail = st.get("compile_failed", 0)
        n_to   = st.get("compile_timeout", 0)
        n_mf   = st.get("measure_failed", 0)
        per_benchmark.append({
            "benchmark": b, "loops": len(rs),
            "avg_reward": sum(rs) / len(rs),
            "min_reward": min(rs), "max_reward": max(rs),
            "loops_win":        sum(1 for r in rs if r > 0.01),
            "loops_regression": sum(1 for r in rs if r < -0.01),
            "loops_noop":       st.get("noop", 0),
            "compile_failed":   n_fail,
            "compile_timeout":  n_to,
            "measure_failed":   n_mf,
            "no_signal":        n_mf,   # compile failures are scored, not no-signal
        })
    per_benchmark.sort(key=lambda e: e["benchmark"])
    return {
        f"{label}_avg_reward":    sum(rewards) / len(rewards) if rewards else 0.0,
        f"{label}_avg_advantage": float("nan"),
        f"{label}_samples":       samples,
        f"{label}_missed":        missed,
        f"{label}_per_benchmark": per_benchmark,
        f"{label}_compile_failures": failures,
    }


def _write_test_report(per_b: list, metrics: dict, label: str,
                       out_file: Path, tag: str) -> dict:
    """Score verdicts, log the table, write the CSV. Returns summary counts."""
    for e in per_b:
        e["verdict"] = ("win" if e["avg_reward"] > 0.01
                        else "regression" if e["avg_reward"] < -0.01
                        else "neutral")
    n_win = sum(1 for e in per_b if e["verdict"] == "win")
    n_reg = sum(1 for e in per_b if e["verdict"] == "regression")
    n_neu = len(per_b) - n_win - n_reg
    macro = sum(e["avg_reward"] for e in per_b) / len(per_b) if per_b else 0.0
    no_sig = sum(e.get("no_signal", 0) for e in per_b)

    log.info("[%s] per-loop avg=%+.4f | per-benchmark avg=%+.4f | "
             "%d win / %d neutral / %d regression | samples=%d no_signal=%d",
             tag, metrics[f"{label}_avg_reward"], macro, n_win, n_neu, n_reg,
             metrics[f"{label}_samples"], no_sig)
    for e in per_b:
        log.info("  %-38s loops=%3d avg=%+.4f min=%+.4f max=%+.4f "
                 "win/reg=%d/%d no_signal=%d [%s]",
                 e["benchmark"], e["loops"], e["avg_reward"], e["min_reward"],
                 e["max_reward"], e["loops_win"], e["loops_regression"],
                 e.get("no_signal", 0), e["verdict"])

    fields = ["benchmark", "loops", "avg_reward", "min_reward", "max_reward",
              "loops_win", "loops_regression", "loops_noop", "compile_failed",
              "compile_timeout", "measure_failed", "no_signal", "verdict"]
    with open(out_file, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="", extrasaction="ignore")
        w.writeheader()
        for e in per_b:
            w.writerow(e)
        w.writerow({"benchmark": "OVERALL_PER_LOOP",
                    "loops": metrics[f"{label}_samples"],
                    "avg_reward": round(metrics[f"{label}_avg_reward"], 6),
                    "no_signal": no_sig})
        w.writerow({"benchmark": "OVERALL_PER_BENCHMARK", "loops": len(per_b),
                    "avg_reward": round(macro, 6), "loops_win": n_win,
                    "loops_regression": n_reg,
                    "verdict": f"{n_win}W/{n_neu}N/{n_reg}R"})
    log.info("[%s] results saved: %s", tag, out_file)
    return {"tag": tag, "per_loop": metrics[f"{label}_avg_reward"],
            "macro": macro, "W": n_win, "N": n_neu, "R": n_reg,
            "no_signal": no_sig}


# ---------------------------------------------------------------------------
# Parallel epoch orchestrator
# ---------------------------------------------------------------------------

def run_parallel_epoch(
    agent: Agent,
    train_loop_assignments: list[dict],
    val_loop_assignments: list[dict],
    normalizer: "FeatureNormalizer",
    baseline_cache: dict,
    n_workers: int,
    buffer: RolloutBuffer,
    device: torch.device,
    args,
    current_epoch: int = 0,
    total_epochs: int = 0,
    reward_cache: "dict | None" = None,
    postf_cache: "dict | None" = None,
) -> tuple[dict, list[Path], list[Path]]:
    """
    Run one complete training + validation epoch across *n_workers* GPU workers.

    Worker k is assigned GPU k (CUDA_VISIBLE_DEVICES=k inside the process).
    Loops are distributed at loop granularity — different workers may handle
    different loops of the same benchmark, each in its own isolated copy under
        tmp_dir / worker_{k} / working_set / benchmark_name /

    train_loop_assignments / val_loop_assignments are flat lists of loop dicts
    as produced by build_loop_assignments(); assign_loops_to_workers() does
    the bin-packing per call so each epoch gets a fresh distribution after
    the caller shuffles the flat list.

    Returns (epoch_stats_dict, [], []) — the empty lists are kept for API
    compatibility with the sequential path; loop-level failures are handled
    inline in the worker and do not propagate back to main.
    """
    reward_cache = reward_cache if reward_cache is not None else {}
    postf_cache  = postf_cache  if postf_cache  is not None else {}

    hparams = {
        "arch":                    args.arch,
        "n_runs":                  args.n_runs,
        "nsys_timeout":            args.nsys_timeout,
        "tmp_dir":                 args.tmp_dir,
        "compile_timeout_penalty": args.compile_timeout_penalty,
        "compile_failure_penalty": args.compile_failure_penalty,
        "reward_deadzone":         args.reward_deadzone,
        "clip_eps":                args.clip_eps,
        "K":                       args.K,
        "batch_size":              args.batch_size,
        "lr":                      args.lr,
        "value_loss_coef":         args.value_loss_coef,
        "entropy_coef":            args.entropy_coef,
        "weight_decay":            args.weight_decay,
        "max_grad_norm":           args.max_grad_norm,
        "agent_type":              getattr(args, "agent", "ppo"),
        "bandit_epsilon":          _bandit_epsilon(args, current_epoch, total_epochs),
        "normalizer_state":        normalizer.state_dict(),
        "baseline_cache":          baseline_cache,
        # Read-only snapshots for the workers; main merges new results into
        # the live dicts as messages arrive and persists them per epoch.
        "reward_cache":            dict(reward_cache),
        "postf_cache":             dict(postf_cache),
        "epoch":                   current_epoch,
        "total_epochs":            total_epochs,
    }

    # Maximum time main will wait between consecutive worker messages.
    worker_msg_timeout = args.n_runs * args.nsys_timeout + 300

    # ------------------------------------------------------------------ #
    # Phase 1: Training pass                                               #
    # ------------------------------------------------------------------ #
    train_per_worker = assign_loops_to_workers(train_loop_assignments, n_workers)
    for w_idx, assignment in enumerate(train_per_worker):
        unique_bmarks = sorted({a["benchmark_name"] for a in assignment})
        log.info(
            "  Worker %d (GPU %d): %d train loops across %s",
            w_idx, w_idx, len(assignment), unique_bmarks,
        )

    initial_weights = _get_weights(agent)

    result_q: mp.Queue = mp.Queue()
    weight_qs: list[mp.Queue] = [mp.Queue() for _ in range(n_workers)]

    workers = []
    for rank in range(n_workers):
        p = mp.Process(
            target=_worker_fn,
            args=(
                rank,
                rank,                          # gpu_id == rank
                train_per_worker[rank],
                initial_weights,
                hparams,
                result_q,
                weight_qs[rank],
                "train",
            ),
            daemon=True,
        )
        p.start()
        workers.append(p)

    # Collect training results
    train_samples    = 0
    train_missed     = 0
    train_rewards:    list[float] = []
    train_advantages: list[float] = []
    train_actor_loss  = 0.0
    train_value_loss  = 0.0
    train_entropy     = 0.0
    train_updates     = 0
    train_cache_hits  = 0
    train_failures    = 0   # compile_failed + compile_timeout
    train_noops       = 0
    train_unmerges    = 0
    train_unrolls     = 0
    done_count = 0

    dropped_msgs = 0
    crashed_workers = 0
    # Keys whose cached value is a compile-failure PENALTY rather than a
    # measurement.  Recorded so the penalty stays re-tunable: once a cell holds
    # -0.15 it is no longer identifiable by value, and migrate_reward_cache.py
    # needs the key list to move it again.  Merged into the on-disk "migration"
    # block at save time (see the cache-persist block below).
    epoch_failure_keys: set[str] = set()
    while done_count < n_workers:
        try:
            msg = result_q.get(timeout=worker_msg_timeout)
        except queue.Empty:
            log.warning("result_q idle for %ds — checking workers are alive",
                        worker_msg_timeout)
            if not any(p.is_alive() for p in workers):
                log.error("All workers have died unexpectedly")
                break
            continue
        except Exception as e:
            # NOT a timeout: the message itself failed to arrive/deserialize.
            # Each occurrence is a LOST training sample, so log it distinctly —
            # conflating this with a timeout hid ~15k dropped samples in the
            # 2026-07 run.  Payloads are now plain Python (see
            # _send_loop_result), which should make this path unreachable.
            dropped_msgs += 1
            log.error("result_q message DROPPED (%s: %s) — sample lost, "
                      "%d dropped so far this epoch",
                      type(e).__name__, e, dropped_msgs)
            if not any(p.is_alive() for p in workers):
                log.error("All workers have died unexpectedly")
                break
            continue

        mtype = msg["type"]

        if mtype == "entry":
            # Rebuild the RolloutEntry from plain Python — workers send lists,
            # never tensors (see _send_loop_result for why).
            buffer.append(RolloutEntry(
                state1=torch.tensor(msg["state1"], dtype=torch.float32),
                state2=torch.tensor(msg["state2"], dtype=torch.float32),
                action1=msg["action1"],
                action2=msg["action2"],
                log_prob1=torch.tensor(msg["log_prob1"], dtype=torch.float32),
                log_prob2=torch.tensor(msg["log_prob2"], dtype=torch.float32),
                reward=msg["reward"],
                mask2=(torch.tensor(msg["mask2"], dtype=torch.bool)
                       if msg["mask2"] is not None else None),
                factor_active=msg["factor_active"],
            ))
            train_samples += 1
            train_rewards.append(msg["reward"])
            train_advantages.append(msg["reward"] - msg["value"])

            # Harvest into cross-epoch caches (skip no-ops — always free).
            # Three mutually-exclusive actions: no-op (unmerge==0, factor==1),
            # unroll-only (unmerge==0, factor>1), unmerge (unmerge==1).
            _is_noop = msg["unmerge"] == 0 and msg["factor"] == 1
            if _is_noop:
                train_noops += 1
            else:
                reward_cache[
                    f"{msg['benchmark']}|{msg['loop_idx']}|{msg['unmerge']}|{msg['factor']}"
                ] = msg["reward"]
                if msg["unmerge"] == 0:
                    train_unrolls += 1
            if msg["unmerge"] == 1:
                train_unmerges += 1
                _pf_key = f"{msg['benchmark']}|{msg['loop_idx']}"
                # Don't cache the pre-features fallback (extraction failure
                # sets state2 = state1); a genuine post-unmerge vector that
                # happens to equal state1 just gets re-extracted — harmless.
                if _pf_key not in postf_cache and msg["state2"] != msg["state1"]:
                    postf_cache[_pf_key] = msg["state2"]
            if msg.get("cached"):
                train_cache_hits += 1
            if msg.get("status") in ("compile_failed", "compile_timeout"):
                train_failures += 1
            if msg.get("status") == "compile_failed":
                epoch_failure_keys.add(
                    f"{msg['benchmark']}|{msg['loop_idx']}|{msg['unmerge']}|{msg['factor']}"
                )

            timeout_flag = " [compile timeout — penalty]" if msg.get("timeout") else ""
            cached_flag  = " [cached]" if msg.get("cached") else ""
            log.info(
                "  [W%d] %s loop_idx=%d unmerge=%d factor=%d "
                "reward=%.4f V(s)=%.4f%s%s",
                msg["rank"], msg["benchmark"], msg["loop_idx"],
                msg["unmerge"], msg["factor"],
                msg["reward"], msg["value"], timeout_flag, cached_flag,
            )
            if buffer.full():
                stats = agent.ppo_update(buffer)
                buffer.clear()
                train_updates += 1
                train_actor_loss += stats["actor_loss"]
                train_value_loss += stats["value_loss"]
                train_entropy    += stats["entropy"]
                log.info(
                    "  PPO update #%d | actor_loss=%.4f | value_loss=%.4f | entropy=%.4f",
                    train_updates, stats["actor_loss"], stats["value_loss"], stats["entropy"],
                )
                # Broadcast updated weights to all workers
                new_weights = _get_weights(agent)
                for wq in weight_qs:
                    try:
                        wq.put_nowait(new_weights)
                    except Exception:
                        pass  # worker already done or queue full — skip

        elif mtype == "step_failed":
            train_missed += 1

        elif mtype == "worker_crashed":
            crashed_workers += 1
            log.error("Worker %d CRASHED (%s) — the rest of its shard is lost "
                      "for this epoch", msg["rank"], msg.get("error"))

        elif mtype == "worker_done":
            done_count += 1
            log.info("Worker %d finished training pass", msg["rank"])

    for p in workers:
        p.join(timeout=30)
        if p.exitcode not in (0, None):
            log.error("Worker exited with code %s — check its traceback above",
                      p.exitcode)

    if crashed_workers:
        log.error("%d/%d workers crashed this epoch — the gradient step is "
                  "computed on a PARTIAL shard; treat this epoch as suspect",
                  crashed_workers, n_workers)

    # Sample accounting: every assigned loop should produce exactly one entry.
    # A shortfall means loops were skipped (no baseline / copy failure) or
    # messages were dropped — both previously silent.  Surface it every epoch.
    _assigned = len(train_loop_assignments)
    if train_samples < _assigned:
        log.warning(
            "Sample accounting: %d/%d assigned loops produced samples "
            "(%d missing: %d dropped messages, %d step_failed, rest skipped "
            "in-worker — grep 'No baseline cached' / 'Failed to copy')",
            train_samples, _assigned, _assigned - train_samples,
            dropped_msgs, train_missed,
        )

    # Flush partial buffer
    if len(buffer) > 0:
        stats = agent.ppo_update(buffer)
        buffer.clear()
        train_updates += 1
        train_actor_loss += stats["actor_loss"]
        train_value_loss += stats["value_loss"]
        train_entropy    += stats["entropy"]
        log.info(
            "Epoch-end PPO flush | actor_loss=%.4f | value_loss=%.4f | entropy=%.4f",
            stats["actor_loss"], stats["value_loss"], stats["entropy"],
        )

    # ------------------------------------------------------------------ #
    # Phase 2: Validation pass (frozen policy, all N workers)              #
    # ------------------------------------------------------------------ #
    val_avg_reward    = float("nan")
    val_avg_advantage = float("nan")
    val_samples       = 0
    val_missed        = 0
    val_per_benchmark: list[dict] = []

    val_cache_hits = 0
    val_noops      = 0
    val_unmerges   = 0
    val_unrolls    = 0
    if val_loop_assignments:
        val_per_worker = assign_loops_to_workers(val_loop_assignments, n_workers)
        val_weights = _get_weights(agent)
        # Refresh the cache snapshots so the val pass benefits from rewards
        # measured during this epoch's training pass.
        hparams["reward_cache"] = dict(reward_cache)
        hparams["postf_cache"]  = dict(postf_cache)
        val_result_q: mp.Queue = mp.Queue()
        val_weight_qs: list[mp.Queue] = [mp.Queue() for _ in range(n_workers)]

        val_workers = []
        for rank in range(n_workers):
            p = mp.Process(
                target=_worker_fn,
                args=(
                    rank,
                    rank,
                    val_per_worker[rank],
                    val_weights,
                    hparams,
                    val_result_q,
                    val_weight_qs[rank],
                    "eval",
                ),
                daemon=True,
            )
            p.start()
            val_workers.append(p)

        all_val_rewards:    list[float] = []
        all_val_advantages: list[float] = []
        per_bench_data: dict[str, list[float]] = {}
        val_done_count = 0

        while val_done_count < n_workers:
            try:
                msg = val_result_q.get(timeout=worker_msg_timeout)
            except queue.Empty:
                log.warning("val_result_q idle for %ds — checking workers",
                            worker_msg_timeout)
                if not any(p.is_alive() for p in val_workers):
                    log.error("All val workers have died unexpectedly")
                    break
                continue
            except Exception as e:
                # Dropped message = lost val sample (see the train loop).
                log.error("val_result_q message DROPPED (%s: %s) — sample lost",
                          type(e).__name__, e)
                if not any(p.is_alive() for p in val_workers):
                    log.error("All val workers have died unexpectedly")
                    break
                continue

            mtype = msg["type"]

            if mtype == "eval_result":
                all_val_rewards.append(msg["reward"])
                all_val_advantages.append(msg["reward"] - msg["value"])
                per_bench_data.setdefault(msg["benchmark"], []).append(msg["reward"])
                val_samples += 1
                # Val measurements are just as deterministic — harvest them too.
                if msg.get("unmerge") == 0 and msg.get("factor") == 1:
                    val_noops += 1
                else:
                    reward_cache[
                        f"{msg['benchmark']}|{msg['loop_idx']}|{msg['unmerge']}|{msg['factor']}"
                    ] = msg["reward"]
                    if msg.get("unmerge") == 0:
                        val_unrolls += 1
                if msg.get("unmerge") == 1:
                    val_unmerges += 1
                if msg.get("cached"):
                    val_cache_hits += 1
                if msg.get("status") == "compile_failed":
                    epoch_failure_keys.add(
                        f"{msg['benchmark']}|{msg['loop_idx']}|"
                        f"{msg['unmerge']}|{msg['factor']}"
                    )
                timeout_flag = " [compile timeout — penalty]" if msg.get("timeout") else ""
                cached_flag  = " [cached]" if msg.get("cached") else ""
                log.info(
                    "  [val W%d] %s loop_idx=%d reward=%.4f V(s)=%.4f%s%s",
                    msg["rank"], msg["benchmark"], msg["loop_idx"],
                    msg["reward"], msg["value"], timeout_flag, cached_flag,
                )

            elif mtype == "step_failed":
                val_missed += 1

            elif mtype == "worker_crashed":
                crashed_workers += 1
                log.error("Worker %d CRASHED during val (%s) — val metrics for "
                          "this epoch are incomplete", msg["rank"],
                          msg.get("error"))

            elif mtype == "worker_done":
                val_done_count += 1
                log.info("Worker %d finished val pass", msg["rank"])

        for p in val_workers:
            p.join(timeout=30)

        if all_val_rewards:
            val_avg_reward    = sum(all_val_rewards) / len(all_val_rewards)
            val_avg_advantage = sum(all_val_advantages) / len(all_val_advantages)
        for bname, rs in per_bench_data.items():
            val_per_benchmark.append({
                "benchmark": bname,
                "loops":     len(rs),
                "avg_reward": sum(rs) / len(rs),
            })

    # Persist the cross-epoch caches.  Tagged with a normalizer fingerprint:
    # postf_cache stores NORMALISED feature vectors, so a refitted normalizer
    # (e.g. precheck re-run with different eligible loops) invalidates them.
    try:
        _sig = hashlib.md5(
            json.dumps(normalizer.state_dict(), sort_keys=True).encode()
        ).hexdigest()[:12]
        cache_path = Path(args.checkpoint_dir) / "reward_cache.json"

        # Carry forward the "migration" block written by
        # migrate_reward_cache.py.  Rewriting the file from scratch used to drop
        # it after the first epoch, which silently destroyed the ability to
        # re-tune --compile-failure-penalty later: a cell holding -0.15 is
        # indistinguishable from a genuine -0.15 measurement, so the key list is
        # the only record of which cells are failures.  Newly observed failures
        # from THIS run are unioned in, so the list stays complete.
        _mig: dict = {}
        if cache_path.exists():
            try:
                _mig = json.loads(cache_path.read_text()).get("migration") or {}
            except Exception:
                _mig = {}
        if epoch_failure_keys or _mig:
            _mig["failure_keys"] = sorted(
                set(_mig.get("failure_keys", [])) | epoch_failure_keys
            )
            _mig.setdefault("failure_penalty", args.compile_failure_penalty)
            _mig.setdefault("deadzone", args.reward_deadzone)
            _mig.setdefault("deadzoned_keys", [])
            _mig.setdefault("history", [])

        _payload = {
            "normalizer_sig": _sig,
            "rewards":        reward_cache,
            "post_features":  postf_cache,
        }
        if _mig:
            _payload["migration"] = _mig
        cache_path.write_text(json.dumps(_payload))
        log.info(
            "Caches saved: %d rewards, %d post-unmerge feature vectors, "
            "%d known failure cells (%s)",
            len(reward_cache), len(postf_cache),
            len(_mig.get("failure_keys", [])), cache_path,
        )
    except Exception as e:
        log.warning("Could not save reward cache: %s", e)

    n_upd = max(train_updates, 1)
    _n_s = max(train_samples, 1)
    epoch_stats = {
        "train_samples":       train_samples,
        "train_missed":        train_missed,
        "crashed_workers":     crashed_workers,
        "assigned_loops":      _assigned,
        "train_rewards":       train_rewards,
        "train_advantages":    train_advantages,
        "train_actor_loss":    train_actor_loss / n_upd,
        "train_value_loss":    train_value_loss / n_upd,
        "train_entropy":       train_entropy / n_upd,
        "train_updates":       train_updates,
        "train_noop_rate":     train_noops / _n_s,
        "train_unmerge_rate":  train_unmerges / _n_s,
        "train_unroll_rate":   train_unrolls / _n_s,
        "train_cache_hit_rate": train_cache_hits / _n_s,
        "train_failure_rate":  train_failures / _n_s,
        "val_avg_reward":      val_avg_reward,
        "val_avg_advantage":   val_avg_advantage,
        "val_samples":         val_samples,
        "val_missed":          val_missed,
        "val_cache_hit_rate":  val_cache_hits / max(val_samples, 1),
        "val_noop_rate":       val_noops / max(val_samples, 1),
        "val_unmerge_rate":    val_unmerges / max(val_samples, 1),
        "val_unroll_rate":     val_unrolls / max(val_samples, 1),
        "val_per_benchmark":   val_per_benchmark,
    }
    return epoch_stats, [], []


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Fail fast if the feature schema and the network input dim disagree — a
    # mismatch here would otherwise surface as an opaque shape error deep in
    # the first forward pass, hours into a run.
    assert len(FEATURE_COLUMNS) == N_FEATURES, (
        f"FEATURE_COLUMNS has {len(FEATURE_COLUMNS)} entries but agent.N_FEATURES "
        f"is {N_FEATURES} — update both together."
    )

    # spawn is required for CUDA safety: forking after CUDA init is unsupported.
    # Set it early, before any torch.cuda usage.
    if args.num_workers > 1:
        mp.set_start_method("spawn", force=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s, GPU arch: %s", device, args.arch)
    if args.num_workers > 1:
        log.info("Parallel mode: %d workers, one GPU each", args.num_workers)

    _agent_common = dict(
        clip_eps=args.clip_eps,
        K=args.K,
        batch_size=args.batch_size,
        lr=args.lr,
        value_loss_coef=args.value_loss_coef,
        entropy_coef=args.entropy_coef,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        device=device,
    )
    if args.agent == "bandit":
        agent = BanditAgent(epsilon=args.bandit_epsilon, **_agent_common)
        log.info("Agent: value-based bandit (epsilon %.2f -> %.2f, "
                 "warm-start epochs %d)",
                 args.bandit_epsilon, args.bandit_epsilon_final,
                 args.bandit_warm_epochs)
    else:
        agent = Agent(**_agent_common)
    if args.resume:
        agent.load(args.resume)
        log.info("Resumed from %s", args.resume)

    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    log.info("Pipeline tmp directory: %s", tmp_dir)

    buffer = RolloutBuffer(capacity=args.buffer_size)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = str(ckpt_dir / "metrics.csv")

    # --- Benchmark discovery ---
    from hecbench import HECBENCH_SRC
    src = Path(args.hecbench_src) if args.hecbench_src else HECBENCH_SRC
    all_benchmarks = discover_benchmarks(src)

    if args.benchmarks:
        requested = set(args.benchmarks)
        all_benchmarks = [b for b in all_benchmarks if b.name in requested]
        missing = requested - {b.name for b in all_benchmarks}
        if missing:
            log.warning("Benchmarks not found or ineligible: %s", sorted(missing))

    # --- Pre-flight eligibility check ---
    cache_file = ckpt_dir / "eligible_benchmarks.json"
    all_benchmarks, loop_counts, loop_records_map, normalizer = precheck_benchmarks(
        all_benchmarks, cache_file, skip=args.skip_precheck
    )

    if not all_benchmarks:
        log.error("No eligible benchmarks found — cannot train. Exiting.")
        return

    # --- Split ---
    train_bmarks, val_bmarks, test_bmarks = split_benchmarks(
        all_benchmarks, args.val_ratio, args.test_ratio, args.split_seed
    )
    log.info(
        "Benchmark split (seed=%d): train=%d  val=%d  test=%d",
        args.split_seed, len(train_bmarks), len(val_bmarks), len(test_bmarks),
    )
    log.info("  train: %s", [b.name for b in train_bmarks])
    log.info("  val:   %s", [b.name for b in val_bmarks])
    log.info("  test:  %s", [b.name for b in test_bmarks])

    # --- Baseline measurement (once per run, after split) ---
    # Train and val baselines are pre-measured so every epoch's reward uses
    # the same reference value.  Test baselines are measured lazily on first
    # access in GpuLoopEnv.reset() (test is only evaluated once at the end).
    baseline_cache = measure_baselines(
        train_bmarks + val_bmarks,
        loop_records_map=loop_records_map,
        arch=args.arch,
        n_runs=args.n_runs,
        nsys_timeout=args.nsys_timeout,
        tmp_dir=tmp_dir,
        gpu_id=0,
        cache_file=ckpt_dir / "baseline_cache.json",
    )

    # --- Cross-epoch reward / post-unmerge-feature caches ---
    # Deterministic environment: same (loop, action) → same reward.  Persisted
    # per epoch by run_parallel_epoch; a restarted run never re-pays compiles
    # or measurements it has already done.  postf_cache stores NORMALISED
    # vectors, so it is only valid under the same fitted normalizer — checked
    # via fingerprint.
    reward_cache: dict[str, float] = {}
    postf_cache: dict[str, list] = {}
    _rc_file = ckpt_dir / "reward_cache.json"
    if _rc_file.exists():
        try:
            _rc_data = json.loads(_rc_file.read_text())
            _sig = hashlib.md5(
                json.dumps(normalizer.state_dict(), sort_keys=True).encode()
            ).hexdigest()[:12]
            reward_cache = _rc_data.get("rewards", {})
            if _rc_data.get("normalizer_sig") == _sig:
                postf_cache = _rc_data.get("post_features", {})
                log.info(
                    "Loaded caches: %d rewards, %d post-unmerge feature vectors",
                    len(reward_cache), len(postf_cache),
                )
            else:
                log.warning(
                    "Normalizer changed since cache was written — keeping %d "
                    "rewards, discarding post-unmerge features",
                    len(reward_cache),
                )
        except Exception as e:
            log.warning("Could not read reward cache (%s): %s", _rc_file, e)

    # Build sequential-path env (also used for test eval at the end)
    env = GpuLoopEnv(
        arch=args.arch,
        n_runs=args.n_runs,
        nsys_timeout=args.nsys_timeout,
        tmp_dir=tmp_dir,
        compile_timeout_penalty=args.compile_timeout_penalty,
        compile_failure_penalty=args.compile_failure_penalty,
        reward_deadzone=args.reward_deadzone,
        normalizer=normalizer,
        baseline_cache=baseline_cache,
    )

    # --- Build flat loop assignment lists for the parallel path ---
    # Each epoch the train list is shuffled before bin-packing so workers
    # get a different loop distribution each time (exploration diversity).
    # Val assignments are stable (evaluation uses a frozen policy).
    train_loop_assignments = build_loop_assignments(train_bmarks, loop_records_map)
    val_loop_assignments   = build_loop_assignments(val_bmarks,   loop_records_map)
    log.info(
        "Loop assignments: train=%d loops  val=%d loops",
        len(train_loop_assignments), len(val_loop_assignments),
    )

    # --- Bandit warm start: regress Q-heads on the cached measurements ---
    # Q-regression needs no behaviour-policy log-probs, so every cached cell
    # is valid supervision — the PPO path structurally cannot do this.
    if args.agent == "bandit" and args.bandit_warm_epochs > 0:
        if args.resume:
            log.info("Bandit warm start SKIPPED (--resume: the checkpoint "
                     "already encodes its training history)")
        else:
            ws_entries, n_cells, n_anchors = build_warm_start_entries(
                train_loop_assignments, reward_cache, postf_cache, normalizer)
            if n_cells == 0:
                log.warning(
                    "Bandit warm start: 0 usable cached cells (empty or "
                    "non-train-split cache) — starting from scratch; "
                    "expect the first epochs to be mostly exploration")
            else:
                log.info(
                    "Bandit warm start: %d cached cells + %d no-op anchors, "
                    "%d passes x K=%d update epochs",
                    n_cells, n_anchors, args.bandit_warm_epochs, args.K)
                ws_stats = agent.warm_start(ws_entries, args.bandit_warm_epochs)
                log.info(
                    "Warm start done | q_loss=%.4f value_loss=%.4f",
                    ws_stats.get("actor_loss", float("nan")),
                    ws_stats.get("value_loss", float("nan")))

    total_updates = 0
    rng = random.Random(args.split_seed)

    # Best-checkpoint tracking: the final policy is not necessarily the best
    # one over a long run.  Track greedy val reward and keep best.pt; the test
    # evaluation at the end uses it instead of the last epoch's weights.
    best_val_reward = float("-inf")
    best_val_epoch: "int | None" = None
    starved_epochs = 0
    peak_samples   = 0

    for epoch in range(1, args.epochs + 1):
        _epoch_filter.set(epoch, args.epochs)
        log.info("=== Epoch %d / %d ===", epoch, args.epochs)

        # Entropy-coefficient decay: linear from --entropy-coef (epoch 1) to
        # --entropy-coef-final (last epoch).  PPO updates run on this (main)
        # agent in both paths, so setting it here is sufficient.
        frac = (epoch - 1) / (args.epochs - 1) if args.epochs > 1 else 0.0
        agent.entropy_coef = (
            args.entropy_coef + frac * (args.entropy_coef_final - args.entropy_coef)
        )
        if args.agent == "bandit":
            # Workers get their epsilon via hparams (run_parallel_epoch computes
            # the same schedule); set it on the main agent too so the sequential
            # path and the entropy metric column (= epsilon) stay correct.
            agent.epsilon = _bandit_epsilon(args, epoch, args.epochs)
            log.info("Bandit epsilon this epoch: %.4f", agent.epsilon)
        else:
            log.info("Entropy coefficient this epoch: %.5f", agent.entropy_coef)

        # ==============================================================
        # PARALLEL PATH  (--num-workers > 1)
        # ==============================================================
        if args.num_workers > 1:
            # Shuffle the flat train assignment list each epoch so workers
            # receive a different loop distribution — equivalent to shuffling
            # benchmark order but at loop granularity.
            rng.shuffle(train_loop_assignments)

            epoch_stats, _, _ = run_parallel_epoch(
                agent=agent,
                train_loop_assignments=train_loop_assignments,
                val_loop_assignments=val_loop_assignments,
                normalizer=normalizer,
                baseline_cache=baseline_cache,
                n_workers=args.num_workers,
                buffer=buffer,
                device=device,
                args=args,
                current_epoch=epoch,
                total_epochs=args.epochs,
                reward_cache=reward_cache,
                postf_cache=postf_cache,
            )

            total_updates += epoch_stats["train_updates"]

            # ----------------------------------------------------------
            # Run health gate.  A worker crash is a CODE bug by definition —
            # every expected failure (no baseline, copy failure, compile
            # failure/timeout, measurement failure) is handled in-worker.  A
            # crashed worker forfeits its whole shard, and because workers are
            # re-spawned per epoch the same bug re-arms every epoch: a single
            # unbound-variable bug once produced 23 consecutive near-empty
            # epochs that still "completed" cleanly.  Burn a minute of GPU
            # time, not a night of it.
            _crashed  = epoch_stats.get("crashed_workers", 0)
            _assigned = epoch_stats.get("assigned_loops", 0)
            _got      = epoch_stats["train_samples"]
            if _crashed > args.tolerate_worker_crashes:
                log.error(
                    "ABORTING: %d worker(s) crashed this epoch (tolerance %d). "
                    "The shard is partial, so this gradient step is computed on "
                    "a biased subset of benchmarks. Fix the traceback above and "
                    "restart; reward_cache.json in this run dir is valid and "
                    "can be carried forward. Override with "
                    "--tolerate-worker-crashes.",
                    _crashed, args.tolerate_worker_crashes,
                )
                raise SystemExit(2)
            # Starvation is measured against the BEST epoch so far, not against
            # the assigned count: loops legitimately skipped in-worker (no
            # cached baseline, copy failure) are skipped every epoch, so an
            # absolute fraction of `assigned` would false-abort a healthy run
            # and burn GPU hours for nothing.  A self-calibrating floor only
            # fires on a real collapse.
            peak_samples = max(peak_samples, _got)
            if _got == 0 or (peak_samples
                             and _got < args.min_sample_frac * peak_samples):
                starved_epochs += 1
                log.error(
                    "Epoch %d collected %d samples (%d assigned, best epoch so "
                    "far %d, floor %.0f%% of best) — %d consecutive starved "
                    "epoch(s)",
                    epoch, _got, _assigned, peak_samples,
                    100 * args.min_sample_frac, starved_epochs,
                )
                if starved_epochs >= args.max_starved_epochs:
                    log.error(
                        "ABORTING: %d consecutive starved epochs. Training is "
                        "not seeing its data — check the worker logs for "
                        "'CRASHED', 'No baseline cached', or 'Failed to copy'. "
                        "Override with --max-starved-epochs / --min-sample-frac.",
                        starved_epochs,
                    )
                    raise SystemExit(2)
            else:
                starved_epochs = 0

            epoch_rewards    = epoch_stats["train_rewards"]
            epoch_advantages = epoch_stats["train_advantages"]
            train_avg_reward = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0.0
            train_avg_adv    = sum(epoch_advantages) / len(epoch_advantages) if epoch_advantages else 0.0

            log.info(
                "Epoch %d complete | train: samples=%d missed=%d "
                "avg_reward=%.4f avg_advantage=%.4f | val: avg_reward=%.4f",
                epoch,
                epoch_stats["train_samples"], epoch_stats["train_missed"],
                train_avg_reward, train_avg_adv,
                epoch_stats["val_avg_reward"],
            )
            if val_bmarks:
                log.info(
                    "  val | avg_reward=%.4f avg_advantage=%.4f samples=%d missed=%d",
                    epoch_stats["val_avg_reward"], epoch_stats["val_avg_advantage"],
                    epoch_stats["val_samples"],    epoch_stats["val_missed"],
                )

            append_metrics(metrics_file, {
                "epoch":               epoch,
                "train_samples":       epoch_stats["train_samples"],
                "train_missed":        epoch_stats["train_missed"],
                "train_avg_reward":    round(train_avg_reward, 6),
                "train_avg_advantage": round(train_avg_adv, 6),
                "train_actor_loss":    round(epoch_stats["train_actor_loss"], 6),
                "train_value_loss":    round(epoch_stats["train_value_loss"], 6),
                "train_entropy":       round(epoch_stats["train_entropy"], 6),
                "train_noop_rate":     round(epoch_stats["train_noop_rate"], 6),
                "train_unmerge_rate":  round(epoch_stats["train_unmerge_rate"], 6),
                "train_unroll_rate":   round(epoch_stats["train_unroll_rate"], 6),
                "train_cache_hit_rate": round(epoch_stats["train_cache_hit_rate"], 6),
                "train_failure_rate":   round(epoch_stats["train_failure_rate"], 6),
                "val_avg_reward":      round(epoch_stats["val_avg_reward"]
                                            if epoch_stats["val_avg_reward"] == epoch_stats["val_avg_reward"]
                                            else float("nan"), 6),
                "val_avg_advantage":   round(epoch_stats["val_avg_advantage"]
                                            if epoch_stats["val_avg_advantage"] == epoch_stats["val_avg_advantage"]
                                            else float("nan"), 6),
                "val_samples":         epoch_stats["val_samples"],
                "val_missed":          epoch_stats["val_missed"],
                "val_cache_hit_rate":  round(epoch_stats["val_cache_hit_rate"], 6),
                "val_noop_rate":       round(epoch_stats["val_noop_rate"], 6),
                "val_unmerge_rate":    round(epoch_stats["val_unmerge_rate"], 6),
                "val_unroll_rate":     round(epoch_stats["val_unroll_rate"], 6),
                "entropy_coef":        round(agent.entropy_coef, 6),
            })

            epoch_val_reward = epoch_stats["val_avg_reward"]

        # ==============================================================
        # SEQUENTIAL PATH  (--num-workers 1, default — unchanged logic)
        # ==============================================================
        else:
            # Shuffle benchmark order each epoch so the rollout buffer is
            # filled in a different order, preventing systematic bias.
            rng.shuffle(train_bmarks)

            epoch_samples    = 0
            epoch_missed     = 0
            epoch_rewards:    list[float] = []
            epoch_advantages: list[float] = []
            epoch_actor_loss  = 0.0
            epoch_value_loss  = 0.0
            epoch_entropy     = 0.0
            epoch_updates     = 0

            # Iterate over a snapshot; failed benchmarks are removed after the loop
            failed_train: list[Path] = []

            for benchmark_dir in list(train_bmarks):
                log.info("Benchmark: %s", benchmark_dir.name)

                try:
                    first_features = env.reset(benchmark_dir)
                except Exception as e:
                    log.warning(
                        "reset failed for %s — removing from training: %s",
                        benchmark_dir.name, e,
                    )
                    failed_train.append(benchmark_dir)
                    continue

                if first_features is None:
                    log.info("  No eligible loops, skipping")
                    continue

                for loop_record in env.eligible_loops:
                    pre_features = loop_record.pre_features.to(device)

                    unmerge, log_p1 = agent.select_unmerge(pre_features)

                    # Study A action space {no-op, unroll-only, unmerge(+unroll)}:
                    # the FactorActor decides on both branches; only its input
                    # state differs (post-unmerge vs. pre-unmerge features).
                    # Trip count is invariant under unmerge, so the pre-features
                    # mask is valid for the unroll-only branch.
                    if unmerge == 1:
                        try:
                            step2_features = env.get_post_unmerge_features(loop_record).to(device)
                        except Exception as e:
                            log.debug("post-unmerge feature extraction failed: %s", e)
                            step2_features = pre_features
                    else:
                        step2_features = pre_features
                    factor_idx, log_p2, mask2 = agent.select_factor(
                        step2_features,
                        trip_known=loop_record.trip_count_known,
                        trip_count=loop_record.trip_count,
                        loop_idx=loop_record.loop_idx,
                    )
                    # Pure no-op (unmerge==0, factor==1) has no unroll decision to
                    # learn, so the FactorActor is not trained on it.
                    factor_active = not (unmerge == 0 and FACTOR_VALUES[factor_idx] == 1)

                    try:
                        next_features, reward, done = env.step(loop_record, unmerge, factor_idx)
                    except Exception as e:
                        log.warning(
                            "  step failed for loop_idx=%d: %s", loop_record.loop_idx, e
                        )
                        epoch_missed += 1
                        continue

                    epoch_samples += 1
                    v = agent.predict_value(pre_features)
                    epoch_rewards.append(reward)
                    epoch_advantages.append(reward - v)

                    timeout_flag = (
                        " [compile timeout — penalty]"
                        if reward == env.compile_timeout_penalty and reward < 0
                        else ""
                    )
                    log.info(
                        "  loop_idx=%d unmerge=%d factor=%d "
                        "reward=%.4f V(s)=%.4f advantage=%.4f%s",
                        loop_record.loop_idx, unmerge, FACTOR_VALUES[factor_idx],
                        reward, v, reward - v, timeout_flag,
                    )

                    buffer.append(RolloutEntry(
                        state1=pre_features.cpu(),
                        state2=step2_features.cpu(),
                        action1=unmerge,
                        action2=factor_idx,
                        log_prob1=log_p1.cpu(),
                        log_prob2=log_p2.cpu(),
                        reward=reward,
                        mask2=mask2.cpu() if mask2 is not None else None,
                        factor_active=factor_active,
                    ))

                    if buffer.full():
                        stats = agent.ppo_update(buffer)
                        buffer.clear()
                        total_updates += 1
                        epoch_updates += 1
                        epoch_actor_loss += stats["actor_loss"]
                        epoch_value_loss += stats["value_loss"]
                        epoch_entropy    += stats["entropy"]
                        log.info(
                            "  PPO update #%d | actor_loss=%.4f | value_loss=%.4f | entropy=%.4f",
                            total_updates, stats["actor_loss"], stats["value_loss"], stats["entropy"],
                        )

                    if done:
                        break

            # Remove failed benchmarks from future training epochs
            for b in failed_train:
                train_bmarks.remove(b)
                log.warning("Removed %s from training set — %d benchmarks remain",
                            b.name, len(train_bmarks))

            # Flush partial buffer at epoch end
            if len(buffer) > 0:
                stats = agent.ppo_update(buffer)
                buffer.clear()
                total_updates += 1
                epoch_updates += 1
                epoch_actor_loss += stats["actor_loss"]
                epoch_value_loss += stats["value_loss"]
                epoch_entropy    += stats["entropy"]
                log.info(
                    "Epoch-end PPO update #%d | actor_loss=%.4f | value_loss=%.4f | entropy=%.4f",
                    total_updates, stats["actor_loss"], stats["value_loss"], stats["entropy"],
                )

            # --- Validation ---
            val_metrics: dict = {}
            if val_bmarks:
                log.info("--- Validation (epoch %d) ---", epoch)
                val_metrics, failed_val = evaluate(agent, env, val_bmarks, device, label="val")
                for b in failed_val:
                    val_bmarks.remove(b)
                    log.warning("Removed %s from validation set — %d benchmarks remain",
                                b.name, len(val_bmarks))
                log.info(
                    "  val | avg_reward=%.4f avg_advantage=%.4f samples=%d missed=%d",
                    val_metrics["val_avg_reward"], val_metrics["val_avg_advantage"],
                    val_metrics["val_samples"], val_metrics["val_missed"],
                )

            # --- Epoch summary + metrics ---
            n_upd = max(epoch_updates, 1)
            train_avg_reward = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0.0
            train_avg_adv    = sum(epoch_advantages) / len(epoch_advantages) if epoch_advantages else 0.0

            log.info(
                "Epoch %d complete | train: samples=%d missed=%d "
                "avg_reward=%.4f avg_advantage=%.4f | val: avg_reward=%.4f",
                epoch, epoch_samples, epoch_missed,
                train_avg_reward, train_avg_adv,
                val_metrics.get("val_avg_reward", float("nan")),
            )

            append_metrics(metrics_file, {
                "epoch":               epoch,
                "train_samples":       epoch_samples,
                "train_missed":        epoch_missed,
                "train_avg_reward":    round(train_avg_reward, 6),
                "train_avg_advantage": round(train_avg_adv, 6),
                "train_actor_loss":    round(epoch_actor_loss / n_upd, 6),
                "train_value_loss":    round(epoch_value_loss / n_upd, 6),
                "train_entropy":       round(epoch_entropy / n_upd, 6),
                "val_avg_reward":      round(val_metrics.get("val_avg_reward", float("nan")), 6),
                "val_avg_advantage":   round(val_metrics.get("val_avg_advantage", float("nan")), 6),
                "val_samples":         val_metrics.get("val_samples", 0),
                "val_missed":          val_metrics.get("val_missed", 0),
                "entropy_coef":        round(agent.entropy_coef, 6),
            })

            epoch_val_reward = val_metrics.get("val_avg_reward", float("nan"))

        # --- Best-checkpoint tracking (both paths) ---
        # Greedy val reward decides "best"; NaN (empty val set) never wins.
        if epoch_val_reward == epoch_val_reward and epoch_val_reward > best_val_reward:
            best_val_reward = epoch_val_reward
            best_val_epoch = epoch
            agent.save(str(ckpt_dir / "best.pt"))
            log.info(
                "New best val reward %.4f (epoch %d) — saved best.pt",
                best_val_reward, epoch,
            )

        # --- Checkpoint (both paths) ---
        if epoch % args.checkpoint_every == 0:
            ckpt_path = ckpt_dir / f"epoch_{epoch:04d}.pt"
            agent.save(str(ckpt_path))
            log.info("Checkpoint saved: %s", ckpt_path)

    # --- Test evaluation ---
    # Runs on MULTIPLE checkpoints (default best + last): best-by-val can be
    # picked very early — in the 95-epoch run it landed at epoch 27 — so the
    # converged policy must be reported alongside it.
    # Always FRESH-measured (no reward cache): the cached path returns frozen
    # first-measurements, which measures cache lookup, not generalisation.
    if test_bmarks:
        test_loop_assignments = build_loop_assignments(test_bmarks, loop_records_map)
        log.info("=== Test Evaluation: %d benchmarks, %d loops ===",
                 len(test_bmarks), len(test_loop_assignments))

        wanted = [s.strip() for s in args.test_checkpoints.split(",") if s.strip()]
        candidates: list[tuple[str, Path]] = []
        for w in wanted:
            if w == "best":
                p = ckpt_dir / "best.pt"
                if p.exists():
                    candidates.append((f"best(ep{best_val_epoch},val={best_val_reward:+.4f})", p))
                else:
                    log.warning("test-checkpoints: no best.pt found — skipping")
            elif w == "last":
                p = ckpt_dir / f"epoch_{args.epochs:04d}.pt"
                if not p.exists():
                    saved = sorted(ckpt_dir.glob("epoch_*.pt"))
                    p = saved[-1] if saved else None
                if p is not None and p.exists():
                    candidates.append((f"last({p.stem})", p))
                else:
                    log.warning("test-checkpoints: no epoch checkpoint found — skipping")
            else:
                p = Path(w)
                if p.exists():
                    candidates.append((p.stem, p))
                else:
                    log.warning("test-checkpoints: %s not found — skipping", w)

        summaries = []
        for tag, ckpt_path in candidates:
            agent.load(str(ckpt_path))
            log.info("--- Test on %s (%s) ---", tag, ckpt_path.name)
            if args.num_workers > 1:
                tm = run_parallel_eval(
                    agent, test_loop_assignments, normalizer, baseline_cache,
                    args.num_workers, args, label="test",
                    use_reward_cache=False,       # fresh measurement
                    postf_cache=postf_cache,
                )
            else:
                tm, _ = evaluate(agent, env, test_bmarks, device, label="test")
            out = ckpt_dir / f"test_results_{ckpt_path.stem}.csv"
            summaries.append(_write_test_report(
                tm.get("test_per_benchmark", []), tm, "test", out, tag))

            # Compile failures are NOT part of the reported speedup statistics —
            # they are a transform-coverage limitation, not a policy result.
            # Dumped here for the separate CPU-only remark analysis
            # (analyze_compile_failures.py).
            cf = tm.get("test_compile_failures", [])
            if cf:
                cf_path = ckpt_dir / f"compile_failures_{ckpt_path.stem}.csv"
                with open(cf_path, "w", newline="") as f:
                    # filename/triple are REQUIRED by analyze_compile_failures.py
                    # to reproduce the exact compile — the UU pass is scoped by
                    # -uu-match-filename / -uu-match-targettriple.
                    w = csv.DictWriter(f, fieldnames=[
                        "benchmark", "loop_idx", "unmerge", "factor",
                        "filename", "triple", "error_signature"],
                        extrasaction="ignore", restval="")
                    w.writeheader()
                    w.writerows(cf)
                by_sig: dict = {}
                for r in cf:
                    by_sig[r["error_signature"]] = by_sig.get(r["error_signature"], 0) + 1
                log.info("[%s] %d compile failures EXCLUDED from the stats above "
                         "-> %s", tag, len(cf), cf_path)
                for sig, n in sorted(by_sig.items(), key=lambda kv: -kv[1])[:5]:
                    log.info("      %4d x  %s", n, sig[:110])

        if len(summaries) > 1:
            log.info("=== Test summary across checkpoints ===")
            for s in summaries:
                log.info("  %-34s per-loop=%+.4f macro=%+.4f  %dW/%dN/%dR  "
                         "no_signal=%d",
                         s["tag"], s["per_loop"], s["macro"], s["W"], s["N"],
                         s["R"], s["no_signal"])

    # --- Plots ---
    plot_training_curves(metrics_file, str(ckpt_dir))


if __name__ == "__main__":
    main()
