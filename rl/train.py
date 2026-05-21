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
import heapq
import json
import logging
import queue
import random
import sys
from datetime import datetime
from pathlib import Path

import torch.multiprocessing as mp

import matplotlib
matplotlib.use("Agg")   # non-interactive — safe on headless servers
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).parent))

from agent import Agent, RolloutBuffer, RolloutEntry, FACTOR_VALUES
from environment import GpuLoopEnv
from hecbench import ARCH, discover_benchmarks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--buffer-size", type=int, default=32)
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
    p.add_argument("--K", type=int, default=4, dest="K",
                   help="PPO epochs per rollout update")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--value-loss-coef", type=float, default=0.5)
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

def precheck_benchmarks(
    benchmarks: list[Path],
    cache_file: Path,
    skip: bool,
) -> tuple[list[Path], dict[str, int], "FeatureNormalizer"]:
    """
    Return (eligible_benchmarks, loop_counts, normalizer).

    loop_counts  — benchmark name → number of eligible loops (for bin-packing)
    normalizer   — FeatureNormalizer fitted on all eligible loop rows

    If *skip* is True and a valid cache exists, load from cache.
    Otherwise run LoopCount on each benchmark and save results to cache.
    """
    from hecbench import FeatureNormalizer, _row_to_tensor, get_loop_features

    # --- Try to load from cache ---
    if skip and cache_file.exists():
        try:
            data = json.loads(cache_file.read_text())
            eligible_names = set(data["eligible"])
            loop_counts: dict[str, int] = data.get("loop_counts", {})
            normalizer = FeatureNormalizer.from_state_dict(data.get("normalizer", {}))
            result = [b for b in benchmarks if b.name in eligible_names]
            log.info(
                "Pre-flight check skipped — loaded %d eligible benchmarks "
                "from cache (%s)%s",
                len(result), cache_file,
                " [normalizer loaded]" if normalizer._fitted else " [no normalizer in cache — will be identity]",
            )
            return result, loop_counts, normalizer
        except Exception as e:
            log.warning("Could not read precheck cache (%s): %s — running check", cache_file, e)

    if skip:
        log.info("--skip-precheck set but no cache found — running pre-flight check anyway")

    log.info("Pre-flight check: testing %d benchmarks for eligible loops...", len(benchmarks))

    eligible: list[Path] = []
    loop_counts: dict[str, int] = {}
    excluded: list[tuple[str, str]] = []
    all_feature_tensors = []

    for b in benchmarks:
        try:
            file_map, _, _ = get_loop_features(b)
            n = sum(len(df) for df in file_map.values())
            if n > 0:
                eligible.append(b)
                loop_counts[b.name] = n
                # Collect raw (un-normalised) feature tensors for fitting
                for df in file_map.values():
                    for _, row in df.iterrows():
                        all_feature_tensors.append(_row_to_tensor(row))
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

    # Fit normalizer on all collected loop feature vectors
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
            "eligible": [b.name for b in eligible],
            "loop_counts": loop_counts,
            "normalizer": normalizer.state_dict(),
            "excluded": [{"name": n, "reason": r} for n, r in excluded],
        }, indent=2))
        log.info("Pre-flight cache saved: %s", cache_file)
    except Exception as e:
        log.warning("Could not save precheck cache: %s", e)

    return eligible, loop_counts, normalizer


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
) -> tuple[dict, list[Path]]:
    """
    Run the current policy over *benchmarks* without any gradient updates.

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

            unmerge, _ = agent.select_unmerge(pre_features)

            if unmerge == 1:
                try:
                    step2_features = env.get_post_unmerge_features(loop_record).to(device)
                except Exception:
                    step2_features = pre_features
            else:
                step2_features = pre_features

            factor_idx, _ = agent.select_factor(
                step2_features, loop_idx=loop_record.loop_idx
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
                "benchmark": benchmark_dir.name,
                "loops": len(bmark_rewards),
                "avg_reward": sum(bmark_rewards) / len(bmark_rewards),
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

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.plot(df["epoch"], df["train_actor_loss"], label="actor_loss")
    ax.plot(df["epoch"], df["train_value_loss"], label="value_loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Training Loss"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(df["epoch"], df["train_avg_reward"], label="train")
    if "val_avg_reward" in df.columns:
        ax.plot(df["epoch"], df["val_avg_reward"], label="val", linestyle="--")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Avg Reward")
    ax.set_title("Average Reward (higher = better)"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[2]
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

def assign_benchmarks_to_workers(
    benchmarks: list[Path],
    loop_counts: dict[str, int],
    n_workers: int,
) -> list[list[Path]]:
    """
    Greedy min-heap bin-packing: assign benchmarks to workers so that
    each worker's total loop count is as equal as possible.

    All loops of a single benchmark always go to the same worker (no splitting)
    to avoid make/compilation conflicts in shared directories.

    Sort benchmarks descending by loop count first, then greedily assign
    each to the least-loaded worker.
    """
    sorted_bmarks = sorted(
        benchmarks,
        key=lambda b: loop_counts.get(b.name, 0),
        reverse=True,
    )
    # heap entries: (total_loops_assigned, worker_idx, benchmark_list)
    heap: list[tuple[int, int, list]] = [(0, i, []) for i in range(n_workers)]
    heapq.heapify(heap)

    for b in sorted_bmarks:
        count = loop_counts.get(b.name, 1)  # default 1 if unknown
        total, idx, blist = heapq.heappop(heap)
        blist.append(b)
        heapq.heappush(heap, (total + count, idx, blist))

    # Re-order by worker index before returning
    assignments: list[list[Path]] = [[] for _ in range(n_workers)]
    for total, idx, blist in heap:
        assignments[idx] = blist
    return assignments


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
    benchmark_list: list,        # list[Path] — passed as posix strings from main
    initial_weights: dict,
    hparams: dict,
    result_q,                    # mp.Queue: worker → main
    weight_q,                    # mp.Queue: main → worker
    mode: str,                   # "train" or "eval"
) -> None:
    """
    Worker process: iterates over *benchmark_list*, runs episodes, and streams
    result dicts to *result_q*.

    Message types sent to result_q:
      {"type": "entry",        "entry": RolloutEntry, "benchmark": str,
       "loop_idx": int, "unmerge": int, "factor": int,
       "reward": float, "value": float}           — training sample
      {"type": "eval_result",  "benchmark": str, "loop_idx": int,
       "reward": float, "value": float}           — eval sample (no entry)
      {"type": "reset_failed", "benchmark": str, "rank": int}
      {"type": "step_failed",  "loop_idx": int,  "rank": int}
      {"type": "worker_done",  "rank": int}

    Weight updates (train mode only): main puts a new weight dict into
    *weight_q* after each PPO update; the worker drains the queue between
    benchmarks.
    """
    import logging
    import os
    import sys
    from pathlib import Path as _Path

    # Re-insert scripts/rl into sys.path (spawn starts fresh)
    _here = _Path(__file__).parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))

    import torch
    from agent import Agent, RolloutEntry, FACTOR_VALUES
    from environment import GpuLoopEnv
    from hecbench import FeatureNormalizer

    _log = logging.getLogger(f"worker.{rank}")
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [W{rank}] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # ------------------------------------------------------------------
    # GPU assignment: set CUDA_VISIBLE_DEVICES BEFORE any CUDA call so
    # PyTorch maps device 0 → physical GPU gpu_id in this process.
    # ------------------------------------------------------------------
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Build agent with the initial weights from main
    agent = Agent(
        clip_eps=hparams["clip_eps"],
        K=hparams["K"],
        batch_size=hparams["batch_size"],
        lr=hparams["lr"],
        value_loss_coef=hparams["value_loss_coef"],
        device=device,
    )
    _load_weights(agent, initial_weights)

    # Per-worker tmp dir to avoid nsys report filename collisions
    tmp_dir = _Path(hparams["tmp_dir"]) / f"worker_{rank}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    worker_normalizer = FeatureNormalizer.from_state_dict(
        hparams.get("normalizer_state", {})
    )
    env = GpuLoopEnv(
        arch=hparams["arch"],
        n_runs=hparams["n_runs"],
        nsys_timeout=hparams["nsys_timeout"],
        tmp_dir=tmp_dir,
        compile_timeout_penalty=hparams["compile_timeout_penalty"],
        gpu_id=gpu_id,
        normalizer=worker_normalizer,
    )

    # benchmark_list may be Path objects or plain strings depending on pickling
    benchmark_dirs = [_Path(b) for b in benchmark_list]

    try:
        for benchmark_dir in benchmark_dirs:
            # ----------------------------------------------------------
            # Between benchmarks: absorb any weight updates from main
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
            # Episode
            # ----------------------------------------------------------
            try:
                first_features = env.reset(benchmark_dir)
            except Exception as e:
                _log.warning("reset failed for %s: %s", benchmark_dir.name, e)
                result_q.put({
                    "type": "reset_failed",
                    "benchmark": benchmark_dir.name,
                    "rank": rank,
                })
                continue

            if first_features is None:
                _log.info("%s — no eligible loops", benchmark_dir.name)
                continue

            for loop_record in env.eligible_loops:
                pre_features = loop_record.pre_features.to(device)

                unmerge, log_p1 = agent.select_unmerge(pre_features)

                if unmerge == 1:
                    try:
                        step2_features = env.get_post_unmerge_features(
                            loop_record
                        ).to(device)
                    except Exception:
                        step2_features = pre_features
                else:
                    step2_features = pre_features

                factor_idx, log_p2 = agent.select_factor(
                    step2_features, loop_idx=loop_record.loop_idx
                )

                try:
                    _, reward, done = env.step(loop_record, unmerge, factor_idx)
                except Exception as e:
                    _log.warning(
                        "step failed for loop_idx=%d: %s", loop_record.loop_idx, e
                    )
                    result_q.put({
                        "type": "step_failed",
                        "loop_idx": loop_record.loop_idx,
                        "rank": rank,
                    })
                    if done:
                        break
                    continue

                v = agent.predict_value(pre_features)

                if mode == "train":
                    entry = RolloutEntry(
                        state1=pre_features.cpu(),
                        state2=step2_features.cpu(),
                        action1=unmerge,
                        action2=factor_idx,
                        log_prob1=log_p1.cpu(),
                        log_prob2=log_p2.cpu(),
                        reward=reward,
                    )
                    result_q.put({
                        "type": "entry",
                        "entry": entry,
                        "benchmark": benchmark_dir.name,
                        "loop_idx": loop_record.loop_idx,
                        "unmerge": unmerge,
                        "factor": FACTOR_VALUES[factor_idx],
                        "reward": reward,
                        "value": v,
                        "rank": rank,
                    })
                else:
                    result_q.put({
                        "type": "eval_result",
                        "benchmark": benchmark_dir.name,
                        "loop_idx": loop_record.loop_idx,
                        "reward": reward,
                        "value": v,
                        "rank": rank,
                    })

                if done:
                    break

    finally:
        result_q.put({"type": "worker_done", "rank": rank})


# ---------------------------------------------------------------------------
# Parallel epoch orchestrator
# ---------------------------------------------------------------------------

def run_parallel_epoch(
    agent: Agent,
    train_benchmarks: list[Path],
    val_benchmarks: list[Path],
    loop_counts: dict[str, int],
    normalizer: "FeatureNormalizer",
    n_workers: int,
    buffer: RolloutBuffer,
    device: torch.device,
    args,
) -> tuple[dict, list[Path], list[Path]]:
    """
    Run one complete training + validation epoch across *n_workers* GPU workers.

    Worker k is assigned GPU k (CUDA_VISIBLE_DEVICES=k inside the process).
    All loops of a benchmark are kept on the same worker to avoid make conflicts.

    Returns:
        (epoch_stats_dict, failed_train_list, failed_val_list)

    epoch_stats_dict keys:
        train_samples, train_missed, train_rewards (list), train_advantages (list),
        train_actor_loss, train_value_loss, train_updates,
        val_avg_reward, val_avg_advantage, val_samples, val_missed,
        val_per_benchmark (list of dicts)
    """
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
        "normalizer_state":        normalizer.state_dict(),
    }

    # ------------------------------------------------------------------ #
    # Phase 1: Training pass                                               #
    # ------------------------------------------------------------------ #
    train_assignments = assign_benchmarks_to_workers(
        train_benchmarks, loop_counts, n_workers
    )
    for w_idx, assignment in enumerate(train_assignments):
        log.info(
            "  Worker %d (GPU %d): %d train benchmarks — %s",
            w_idx, w_idx, len(assignment), [b.name for b in assignment],
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
                train_assignments[rank],
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
    train_samples   = 0
    train_missed    = 0
    train_rewards:    list[float] = []
    train_advantages: list[float] = []
    train_actor_loss  = 0.0
    train_value_loss  = 0.0
    train_updates     = 0
    failed_train:  list[Path] = []
    done_count = 0

    while done_count < n_workers:
        try:
            msg = result_q.get(timeout=600)
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
            log.info(
                "  [W%d] %s loop_idx=%d unmerge=%d factor=%d "
                "reward=%.4f V(s)=%.4f",
                msg["rank"], msg["benchmark"], msg["loop_idx"],
                msg["unmerge"], msg["factor"],
                msg["reward"], msg["value"],
            )
            if buffer.full():
                stats = agent.ppo_update(buffer)
                buffer.clear()
                train_updates += 1
                train_actor_loss += stats["actor_loss"]
                train_value_loss += stats["value_loss"]
                log.info(
                    "  PPO update #%d | actor_loss=%.4f | value_loss=%.4f",
                    train_updates, stats["actor_loss"], stats["value_loss"],
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

        elif mtype == "reset_failed":
            bench_name = msg["benchmark"]
            matched = [b for b in train_benchmarks if b.name == bench_name]
            if matched:
                failed_train.extend(matched)
            log.warning("Train reset failed: %s (worker %d)", bench_name, msg["rank"])

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
        log.info(
            "Epoch-end PPO flush | actor_loss=%.4f | value_loss=%.4f",
            stats["actor_loss"], stats["value_loss"],
        )

    # ------------------------------------------------------------------ #
    # Phase 2: Validation pass (frozen policy, all N workers)              #
    # ------------------------------------------------------------------ #
    val_avg_reward    = float("nan")
    val_avg_advantage = float("nan")
    val_samples       = 0
    val_missed        = 0
    val_per_benchmark: list[dict] = []
    failed_val: list[Path] = []

    if val_benchmarks:
        val_assignments = assign_benchmarks_to_workers(
            val_benchmarks, loop_counts, n_workers
        )
        val_weights = _get_weights(agent)
        val_result_q: mp.Queue = mp.Queue()
        val_weight_qs: list[mp.Queue] = [mp.Queue() for _ in range(n_workers)]

        val_workers = []
        for rank in range(n_workers):
            p = mp.Process(
                target=_worker_fn,
                args=(
                    rank,
                    rank,
                    val_assignments[rank],
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
                msg = val_result_q.get(timeout=600)
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
                log.info(
                    "  [val W%d] %s loop_idx=%d reward=%.4f V(s)=%.4f",
                    msg["rank"], msg["benchmark"], msg["loop_idx"],
                    msg["reward"], msg["value"],
                )

            elif mtype == "step_failed":
                val_missed += 1

            elif mtype == "reset_failed":
                bench_name = msg["benchmark"]
                matched = [b for b in val_benchmarks if b.name == bench_name]
                if matched:
                    failed_val.extend(matched)
                log.warning("Val reset failed: %s (worker %d)", bench_name, msg["rank"])

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

    n_upd = max(train_updates, 1)
    epoch_stats = {
        "train_samples":       train_samples,
        "train_missed":        train_missed,
        "train_rewards":       train_rewards,
        "train_advantages":    train_advantages,
        "train_actor_loss":    train_actor_loss / n_upd,
        "train_value_loss":    train_value_loss / n_upd,
        "train_updates":       train_updates,
        "val_avg_reward":      val_avg_reward,
        "val_avg_advantage":   val_avg_advantage,
        "val_samples":         val_samples,
        "val_missed":          val_missed,
        "val_per_benchmark":   val_per_benchmark,
    }
    return epoch_stats, failed_train, failed_val


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

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
        device=device,
    )
    if args.resume:
        agent.load(args.resume)
        log.info("Resumed from %s", args.resume)

    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    log.info("Pipeline tmp directory: %s", tmp_dir)

    env = GpuLoopEnv(
        arch=args.arch,
        n_runs=args.n_runs,
        nsys_timeout=args.nsys_timeout,
        tmp_dir=tmp_dir,
        compile_timeout_penalty=args.compile_timeout_penalty,
        normalizer=normalizer,
    )
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
    all_benchmarks, loop_counts, normalizer = precheck_benchmarks(
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

    total_updates = 0

    for epoch in range(1, args.epochs + 1):
        log.info("=== Epoch %d / %d ===", epoch, args.epochs)

        # ==============================================================
        # PARALLEL PATH  (--num-workers > 1)
        # ==============================================================
        if args.num_workers > 1:
            epoch_stats, failed_train, failed_val = run_parallel_epoch(
                agent=agent,
                train_benchmarks=list(train_bmarks),
                val_benchmarks=list(val_bmarks),
                loop_counts=loop_counts,
                normalizer=normalizer,
                n_workers=args.num_workers,
                buffer=buffer,
                device=device,
                args=args,
            )

            # Remove persistently failing benchmarks
            for b in failed_train:
                if b in train_bmarks:
                    train_bmarks.remove(b)
                    log.warning("Removed %s from training set — %d remain",
                                b.name, len(train_bmarks))
            for b in failed_val:
                if b in val_bmarks:
                    val_bmarks.remove(b)
                    log.warning("Removed %s from val set — %d remain",
                                b.name, len(val_bmarks))

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
                "val_avg_reward":      round(epoch_stats["val_avg_reward"]
                                            if epoch_stats["val_avg_reward"] == epoch_stats["val_avg_reward"]
                                            else float("nan"), 6),
                "val_avg_advantage":   round(epoch_stats["val_avg_advantage"]
                                            if epoch_stats["val_avg_advantage"] == epoch_stats["val_avg_advantage"]
                                            else float("nan"), 6),
                "val_samples":         epoch_stats["val_samples"],
                "val_missed":          epoch_stats["val_missed"],
            })

        # ==============================================================
        # SEQUENTIAL PATH  (--num-workers 1, default — unchanged logic)
        # ==============================================================
        else:
            epoch_samples    = 0
            epoch_missed     = 0
            epoch_rewards:    list[float] = []
            epoch_advantages: list[float] = []
            epoch_actor_loss  = 0.0
            epoch_value_loss  = 0.0
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

                    if unmerge == 1:
                        try:
                            step2_features = env.get_post_unmerge_features(loop_record).to(device)
                        except Exception as e:
                            log.debug("post-unmerge feature extraction failed: %s", e)
                            step2_features = pre_features
                    else:
                        step2_features = pre_features

                    factor_idx, log_p2 = agent.select_factor(
                        step2_features, loop_idx=loop_record.loop_idx
                    )

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
                    ))

                    if buffer.full():
                        stats = agent.ppo_update(buffer)
                        buffer.clear()
                        total_updates += 1
                        epoch_updates += 1
                        epoch_actor_loss += stats["actor_loss"]
                        epoch_value_loss += stats["value_loss"]
                        log.info(
                            "  PPO update #%d | actor_loss=%.4f | value_loss=%.4f",
                            total_updates, stats["actor_loss"], stats["value_loss"],
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
                log.info(
                    "Epoch-end PPO update #%d | actor_loss=%.4f | value_loss=%.4f",
                    total_updates, stats["actor_loss"], stats["value_loss"],
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
                "val_avg_reward":      round(val_metrics.get("val_avg_reward", float("nan")), 6),
                "val_avg_advantage":   round(val_metrics.get("val_avg_advantage", float("nan")), 6),
                "val_samples":         val_metrics.get("val_samples", 0),
                "val_missed":          val_metrics.get("val_missed", 0),
            })

        # --- Checkpoint (both paths) ---
        if epoch % args.checkpoint_every == 0:
            ckpt_path = ckpt_dir / f"epoch_{epoch:04d}.pt"
            agent.save(str(ckpt_path))
            log.info("Checkpoint saved: %s", ckpt_path)

    # --- Test evaluation (final model) ---
    if test_bmarks:
        log.info("=== Test Evaluation (final model) ===")
        test_metrics, _ = evaluate(agent, env, test_bmarks, device, label="test")

        log.info(
            "Test | avg_reward=%.4f avg_advantage=%.4f samples=%d missed=%d",
            test_metrics["test_avg_reward"], test_metrics["test_avg_advantage"],
            test_metrics["test_samples"], test_metrics["test_missed"],
        )
        log.info("Per-benchmark test results:")
        for entry in test_metrics.get("test_per_benchmark", []):
            log.info("  %-40s loops=%d  avg_reward=%.4f",
                     entry["benchmark"], entry["loops"], entry["avg_reward"])

        test_results_file = str(ckpt_dir / "test_results.csv")
        with open(test_results_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["benchmark", "loops", "avg_reward"])
            writer.writeheader()
            for entry in test_metrics.get("test_per_benchmark", []):
                writer.writerow(entry)
            writer.writerow({
                "benchmark": "OVERALL",
                "loops": test_metrics["test_samples"],
                "avg_reward": round(test_metrics["test_avg_reward"], 6),
            })
        log.info("Test results saved: %s", test_results_file)

    # --- Plots ---
    plot_training_curves(metrics_file, str(ckpt_dir))


if __name__ == "__main__":
    main()
