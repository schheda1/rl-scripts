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

from agent import Agent, RolloutBuffer, RolloutEntry, FACTOR_VALUES, N_FEATURES
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
    p.add_argument("--arch", type=str, default=ARCH)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--checkpoint-every", type=int, default=1,
                   help="Save a checkpoint every N epochs (default: every epoch)")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--lr", type=float, default=3e-4)
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

    log.info("Baseline cache: %d benchmarks total (%d newly measured)",
             len(cache), len(benchmarks))

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
    samples = 0
    missed = 0

    for benchmark_dir in benchmarks:
        bmark_rewards: list[float] = []

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

            v = agent.predict_value(pre_features)
            log.info(
                "  [%s] %s loop_idx=%d unmerge=%d factor=%d "
                "reward=%.4f V(s)=%.4f",
                label, benchmark_dir.name, loop_record.loop_idx,
                unmerge, FACTOR_VALUES[factor_idx], reward, v,
            )

            all_rewards.append(reward)
            all_advantages.append(reward - v)
            bmark_rewards.append(reward)
            samples += 1

            if done:
                break

        if bmark_rewards:
            per_benchmark.append({
                "benchmark":   benchmark_dir.name,
                "loops":       len(bmark_rewards),
                "avg_reward":  sum(bmark_rewards) / len(bmark_rewards),
                "min_reward":  min(bmark_rewards),
                "max_reward":  max(bmark_rewards),
                # ±1% classification thresholds — below that is measurement noise
                "loops_win":        sum(1 for r in bmark_rewards if r > 0.01),
                "loops_regression": sum(1 for r in bmark_rewards if r < -0.01),
            })

    avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
    avg_adv    = sum(all_advantages) / len(all_advantages) if all_advantages else 0.0

    metrics = {
        f"{label}_avg_reward":    avg_reward,
        f"{label}_avg_advantage": avg_adv,
        f"{label}_samples":       samples,
        f"{label}_missed":        missed,
        f"{label}_per_benchmark": per_benchmark,
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
        Agent, RolloutEntry, FACTOR_VALUES,
        _IDX_TRIP_COUNT_KNOWN, _IDX_TRIP_COUNT,
    )
    from environment import GpuLoopEnv, LoopRecord
    from hecbench import FeatureNormalizer, compile_single_loop, demangle, demangled_to_filter, measure_kernel_time

    _epoch = hparams.get("epoch", 0)
    _total = hparams.get("total_epochs", 0)
    _epoch_tag = f"[{_epoch}/{_total}] " if _total > 0 else ""
    _log = logging.getLogger(f"worker.{rank}")
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s %(levelname)s {_epoch_tag}[W{rank}] %(message)s",
        datefmt="%H:%M:%S",
    )

    # ------------------------------------------------------------------
    # GPU assignment: set CUDA_VISIBLE_DEVICES BEFORE any CUDA call so
    # PyTorch maps device 0 → physical GPU gpu_id in this process.
    # ------------------------------------------------------------------
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    agent = Agent(
        clip_eps=hparams["clip_eps"],
        K=hparams["K"],
        batch_size=hparams["batch_size"],
        lr=hparams["lr"],
        value_loss_coef=hparams["value_loss_coef"],
        entropy_coef=hparams["entropy_coef"],
        device=device,
    )
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
        gpu_id=gpu_id,
        normalizer=worker_normalizer,
        baseline_cache=baseline_cache,
    )

    try:
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
                    )
                    continue

                # --- Compile + measure (unmerge, factor) ---
                is_timeout = False
                from_cache = False
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
                        ok = compile_single_loop(
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
                            "%s loop_idx=%d compile timeout — penalty=%.2f",
                            bench_name, loop_idx, reward,
                        )
                        v = agent.predict_value(pre_features)
                        _send_loop_result(
                            result_q, mode, rank, bench_name, loop_idx,
                            unmerge, factor, reward, v, is_timeout,
                            pre_features, step2_features, factor_idx,
                            log_p1, log_p2, mask2, factor_active=True,
                        )
                        continue

                    if not ok:
                        # Compile error — treat as no-op (no training signal)
                        modified_ms = baseline_ms
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
                            modified_ms = baseline_ms

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

                v = agent.predict_value(pre_features)
                _send_loop_result(
                    result_q, mode, rank, bench_name, loop_idx,
                    unmerge, factor, reward, v, is_timeout,
                    pre_features, step2_features, factor_idx,
                    log_p1, log_p2, mask2, cached=from_cache,
                    factor_active=True,
                )

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
) -> None:
    """Put one loop result onto result_q in the appropriate format."""
    import torch
    from agent import RolloutEntry
    if mode == "train":
        result_q.put({
            "type":      "entry",
            "entry":     RolloutEntry(
                state1=pre_features.cpu(),
                state2=step2_features.cpu(),
                action1=unmerge,
                action2=factor_idx,
                log_prob1=log_p1.cpu(),
                log_prob2=log_p2.cpu(),
                reward=reward,
                mask2=mask2.cpu() if mask2 is not None else None,
                factor_active=factor_active,
            ),
            "benchmark": bench_name,
            "loop_idx":  loop_idx,
            "unmerge":   unmerge,
            "factor":    factor,
            "reward":    reward,
            "value":     value,
            "timeout":   is_timeout,
            "cached":    cached,
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
            "rank":      rank,
        })


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
        "clip_eps":                args.clip_eps,
        "K":                       args.K,
        "batch_size":              args.batch_size,
        "lr":                      args.lr,
        "value_loss_coef":         args.value_loss_coef,
        "entropy_coef":            args.entropy_coef,
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
    train_noops       = 0
    train_unmerges    = 0
    train_unrolls     = 0
    done_count = 0

    while done_count < n_workers:
        try:
            msg = result_q.get(timeout=worker_msg_timeout)
        except Exception:
            log.warning("result_q timed out waiting for workers — checking alive")
            if not any(p.is_alive() for p in workers):
                log.error("All workers have died unexpectedly")
                break
            continue

        mtype = msg["type"]

        if mtype == "entry":
            buffer.append(msg["entry"])
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
                if _pf_key not in postf_cache and not torch.equal(
                    msg["entry"].state2, msg["entry"].state1
                ):
                    postf_cache[_pf_key] = msg["entry"].state2.tolist()
            if msg.get("cached"):
                train_cache_hits += 1

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

        elif mtype == "worker_done":
            done_count += 1
            log.info("Worker %d finished training pass", msg["rank"])

    for p in workers:
        p.join(timeout=30)

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
            except Exception:
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
                timeout_flag = " [compile timeout — penalty]" if msg.get("timeout") else ""
                cached_flag  = " [cached]" if msg.get("cached") else ""
                log.info(
                    "  [val W%d] %s loop_idx=%d reward=%.4f V(s)=%.4f%s%s",
                    msg["rank"], msg["benchmark"], msg["loop_idx"],
                    msg["reward"], msg["value"], timeout_flag, cached_flag,
                )

            elif mtype == "step_failed":
                val_missed += 1

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
        cache_path.write_text(json.dumps({
            "normalizer_sig": _sig,
            "rewards":        reward_cache,
            "post_features":  postf_cache,
        }))
        log.info(
            "Caches saved: %d rewards, %d post-unmerge feature vectors (%s)",
            len(reward_cache), len(postf_cache), cache_path,
        )
    except Exception as e:
        log.warning("Could not save reward cache: %s", e)

    n_upd = max(train_updates, 1)
    _n_s = max(train_samples, 1)
    epoch_stats = {
        "train_samples":       train_samples,
        "train_missed":        train_missed,
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

    agent = Agent(
        clip_eps=args.clip_eps,
        K=args.K,
        batch_size=args.batch_size,
        lr=args.lr,
        value_loss_coef=args.value_loss_coef,
        entropy_coef=args.entropy_coef,
        device=device,
    )
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

    total_updates = 0
    rng = random.Random(args.split_seed)

    # Best-checkpoint tracking: the final policy is not necessarily the best
    # one over a long run.  Track greedy val reward and keep best.pt; the test
    # evaluation at the end uses it instead of the last epoch's weights.
    best_val_reward = float("-inf")
    best_val_epoch: "int | None" = None

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
    # Uses the BEST checkpoint by greedy val reward, not the last epoch's
    # weights: over a long run the final policy can be past its peak.
    if test_bmarks:
        best_ckpt = ckpt_dir / "best.pt"
        if best_ckpt.exists():
            agent.load(str(best_ckpt))
            log.info(
                "=== Test Evaluation (best checkpoint: epoch %s, val=%.4f) ===",
                best_val_epoch, best_val_reward,
            )
        else:
            log.info("=== Test Evaluation (final model — no best.pt found) ===")

        test_metrics, _ = evaluate(agent, env, test_bmarks, device, label="test")

        per_b = test_metrics.get("test_per_benchmark", [])
        # Per-benchmark verdicts at ±1% (below that is measurement noise)
        for entry in per_b:
            entry["verdict"] = (
                "win" if entry["avg_reward"] > 0.01
                else "regression" if entry["avg_reward"] < -0.01
                else "neutral"
            )
        n_win = sum(1 for e in per_b if e["verdict"] == "win")
        n_reg = sum(1 for e in per_b if e["verdict"] == "regression")
        n_neu = len(per_b) - n_win - n_reg
        macro_avg = (
            sum(e["avg_reward"] for e in per_b) / len(per_b) if per_b else 0.0
        )

        log.info(
            "Test | per-loop avg_reward=%.4f | per-benchmark avg=%.4f | "
            "benchmarks: %d win / %d neutral / %d regression | samples=%d missed=%d",
            test_metrics["test_avg_reward"], macro_avg,
            n_win, n_neu, n_reg,
            test_metrics["test_samples"], test_metrics["test_missed"],
        )
        log.info("Per-benchmark test results:")
        for entry in per_b:
            log.info(
                "  %-40s loops=%3d  avg=%+.4f  min=%+.4f  max=%+.4f  "
                "win/reg loops=%d/%d  [%s]",
                entry["benchmark"], entry["loops"], entry["avg_reward"],
                entry["min_reward"], entry["max_reward"],
                entry["loops_win"], entry["loops_regression"], entry["verdict"],
            )

        test_results_file = str(ckpt_dir / "test_results.csv")
        fieldnames = ["benchmark", "loops", "avg_reward", "min_reward",
                      "max_reward", "loops_win", "loops_regression", "verdict"]
        with open(test_results_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
            writer.writeheader()
            for entry in per_b:
                writer.writerow(entry)
            writer.writerow({
                "benchmark": "OVERALL_PER_LOOP",
                "loops": test_metrics["test_samples"],
                "avg_reward": round(test_metrics["test_avg_reward"], 6),
            })
            writer.writerow({
                "benchmark": "OVERALL_PER_BENCHMARK",
                "loops": len(per_b),
                "avg_reward": round(macro_avg, 6),
                "loops_win": n_win,
                "loops_regression": n_reg,
                "verdict": f"{n_win}W/{n_neu}N/{n_reg}R",
            })
        log.info("Test results saved: %s", test_results_file)

    # --- Plots ---
    plot_training_curves(metrics_file, str(ckpt_dir))


if __name__ == "__main__":
    main()
