"""
Standalone test/val/train evaluation for a chosen checkpoint.

Runs the greedy (deployment-mode) policy of ANY saved checkpoint over a chosen
benchmark split and writes a per-benchmark report — the same format train.py
produces at the end of a run, but for a checkpoint of your choosing (a specific
epoch, best.pt, or any .pt), not just the auto-selected best-by-val model.

CRITICAL — the split must match the training run:
  The test set is derived from --split-seed, --val-ratio, --test-ratio, AND the
  set of eligible benchmarks (loaded from the run's eligible_benchmarks.json).
  Pass the SAME values the training run used, and point --run-dir at that run's
  checkpoint directory, or you will evaluate on a different set of benchmarks
  than the model was held out from.

Usage:
  python eval_model.py checkpoints/run_long/epoch_0042.pt
  python eval_model.py checkpoints/run_long/best.pt --split test
  python eval_model.py some_model.pt --split val --sampled --n-runs 3

By default it reuses the run's eligible_benchmarks.json (precheck + normalizer)
and baseline_cache.json, so evaluation is fast: most measurements come straight
from those caches plus the reward cache if present.
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch

from agent import Agent
from environment import GpuLoopEnv
from hecbench import ARCH, HECBENCH_SRC, discover_benchmarks
from train import (
    precheck_benchmarks,
    split_benchmarks,
    measure_baselines,
    evaluate,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("eval")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("checkpoint", help="Path to the .pt checkpoint to evaluate")
    p.add_argument("--split", choices=["test", "val", "train"], default="test",
                   help="Which split to evaluate (default: test)")
    p.add_argument("--run-dir", default=None,
                   help="Dir with eligible_benchmarks.json / baseline_cache.json "
                        "(default: the checkpoint's parent directory)")
    p.add_argument("--out", default=None,
                   help="Output CSV path (default: {run-dir}/eval_{split}_{ckpt}.csv)")
    p.add_argument("--sampled", action="store_true",
                   help="Use the sampled policy instead of greedy argmax "
                        "(default: greedy = deployment mode)")
    # --- Must match the training run ---
    p.add_argument("--val-ratio",  type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--split-seed", type=int,   default=42)
    # --- Measurement ---
    p.add_argument("--arch", default=ARCH)
    p.add_argument("--n-runs", type=int, default=2)
    p.add_argument("--nsys-timeout", type=int, default=300)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--compile-timeout-penalty", type=float, default=-1.0)
    p.add_argument("--tmp-dir", default=None,
                   help="Temp dir for nsys reports (default: {run-dir}/eval_tmp)")
    p.add_argument("--hecbench-src", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        log.error("Checkpoint not found: %s", ckpt)
        sys.exit(1)

    run_dir = Path(args.run_dir) if args.run_dir else ckpt.parent
    tmp_dir = Path(args.tmp_dir) if args.tmp_dir else run_dir / "eval_tmp"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info("Evaluating checkpoint: %s", ckpt)
    log.info("  split=%s  policy=%s  arch=%s  n_runs=%d",
             args.split, "sampled" if args.sampled else "greedy",
             args.arch, args.n_runs)
    log.info("  run-dir=%s (split-seed=%d val=%.2f test=%.2f)",
             run_dir, args.split_seed, args.val_ratio, args.test_ratio)

    # --- Discover + precheck (reuse the run's cache so the split matches) ---
    src = Path(args.hecbench_src) if args.hecbench_src else HECBENCH_SRC
    all_benchmarks = discover_benchmarks(src)
    cache_file = run_dir / "eligible_benchmarks.json"
    if not cache_file.exists():
        log.warning(
            "No eligible_benchmarks.json in %s — running a fresh precheck. "
            "The split will only match training if the eligible set comes out "
            "identical. Prefer pointing --run-dir at the training run's dir.",
            run_dir,
        )
    all_benchmarks, _, loop_records_map, normalizer = precheck_benchmarks(
        all_benchmarks, cache_file, skip=True,
    )
    if not all_benchmarks:
        log.error("No eligible benchmarks — cannot evaluate.")
        sys.exit(1)

    # --- Reproduce the exact split ---
    train_b, val_b, test_b = split_benchmarks(
        all_benchmarks, args.val_ratio, args.test_ratio, args.split_seed,
    )
    chosen = {"train": train_b, "val": val_b, "test": test_b}[args.split]
    log.info("%s split: %d benchmarks", args.split, len(chosen))

    # --- Baselines (reuse cache; measure only the missing ones) ---
    baseline_cache = measure_baselines(
        chosen,
        loop_records_map=loop_records_map,
        arch=args.arch,
        n_runs=args.n_runs,
        nsys_timeout=args.nsys_timeout,
        tmp_dir=tmp_dir,
        gpu_id=args.gpu_id,
        cache_file=run_dir / "baseline_cache.json",
    )

    # --- Env + agent ---
    env = GpuLoopEnv(
        arch=args.arch,
        n_runs=args.n_runs,
        nsys_timeout=args.nsys_timeout,
        tmp_dir=tmp_dir,
        compile_timeout_penalty=args.compile_timeout_penalty,
        gpu_id=args.gpu_id,
        normalizer=normalizer,
        baseline_cache=baseline_cache,
    )
    agent = Agent(device=device)
    agent.load(str(ckpt))

    # --- Evaluate ---
    metrics, _ = evaluate(
        agent, env, chosen, device,
        label=args.split, greedy=not args.sampled,
    )

    per_b = metrics.get(f"{args.split}_per_benchmark", [])
    per_loop_samples = metrics.get(f"{args.split}_samples", 0)
    per_loop_avg = metrics.get(f"{args.split}_avg_reward", 0.0)

    # Per-benchmark verdicts at ±1% (below that is measurement noise)
    for e in per_b:
        e["verdict"] = (
            "win" if e["avg_reward"] > 0.01
            else "regression" if e["avg_reward"] < -0.01
            else "neutral"
        )
    n_win = sum(1 for e in per_b if e["verdict"] == "win")
    n_reg = sum(1 for e in per_b if e["verdict"] == "regression")
    n_neu = len(per_b) - n_win - n_reg
    macro_avg = sum(e["avg_reward"] for e in per_b) / len(per_b) if per_b else 0.0

    log.info("")
    log.info("=== %s results — %s (%s policy) ===",
             args.split, ckpt.name, "sampled" if args.sampled else "greedy")
    log.info("Per-loop avg reward      : %+.4f  (%d loops)", per_loop_avg, per_loop_samples)
    log.info("Per-benchmark avg reward : %+.4f  (%d benchmarks)", macro_avg, len(per_b))
    log.info("Benchmark verdicts       : %d win / %d neutral / %d regression",
             n_win, n_neu, n_reg)
    log.info("")
    for e in sorted(per_b, key=lambda x: x["avg_reward"], reverse=True):
        log.info("  %-40s loops=%3d  avg=%+.4f  min=%+.4f  max=%+.4f  "
                 "win/reg=%d/%d  [%s]",
                 e["benchmark"], e["loops"], e["avg_reward"],
                 e["min_reward"], e["max_reward"],
                 e["loops_win"], e["loops_regression"], e["verdict"])

    # --- Write CSV ---
    out = Path(args.out) if args.out else run_dir / f"eval_{args.split}_{ckpt.stem}.csv"
    fields = ["benchmark", "loops", "avg_reward", "min_reward", "max_reward",
              "loops_win", "loops_regression", "verdict"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        for e in per_b:
            w.writerow({k: e.get(k, "") for k in fields})
        w.writerow({"benchmark": "OVERALL_PER_LOOP",
                    "loops": per_loop_samples,
                    "avg_reward": round(per_loop_avg, 6)})
        w.writerow({"benchmark": "OVERALL_PER_BENCHMARK",
                    "loops": len(per_b),
                    "avg_reward": round(macro_avg, 6),
                    "loops_win": n_win, "loops_regression": n_reg,
                    "verdict": f"{n_win}W/{n_neu}N/{n_reg}R"})
    log.info("")
    log.info("Report written: %s", out)


if __name__ == "__main__":
    main()
