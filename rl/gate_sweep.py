"""
Tier-2 (cheap-GPU) instrumented gate-sweep evaluation.

Runs the greedy policy of ANY checkpoint over a split, recording per loop:
the measured reward, the critic's V(s), the k-NN novelty distance to the
training set, and the policy's own confidence — then computes the full
abstention-tradeoff curves (coverage vs. gain vs. regressions) for a
value gate and a novelty gate.

Compared to analyze_run.py (offline, log-derived, best.pt only), this script
evaluates ANY checkpoint and takes fresh measurements where needed.  It rides
the run's caches — eligible_benchmarks.json, baseline_cache.json, and
reward_cache.json — so repeat (loop, action) pairs cost nothing; newly
measured rewards are merged back into reward_cache.json for future use.

Usage (adjust paths — pass the SAME split params the training run used):
  python gate_sweep.py checkpoints/run_long/best.pt --split test
  python gate_sweep.py checkpoints/run_long/epoch_0090.pt --split test \
         --hecbench-src /path/to/HeCBench/src

Outputs in --out-dir (default {run-dir}/gate_sweep):
  loops_{split}_{ckpt}.csv        per-loop record (reward, V, novelty, conf)
  gate_curves_{split}_{ckpt}.csv  threshold sweep for both gates
"""

import argparse
import csv
import json
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch

from agent import Agent, FACTOR_VALUES
from analyze_run import compute_gate_curves, min_distance, rank_auc
from environment import GpuLoopEnv
from hecbench import (
    ARCH, HECBENCH_SRC, compile_single_loop, discover_benchmarks,
    measure_kernel_time,
)
from train import precheck_benchmarks, split_benchmarks, measure_baselines

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("gate_sweep")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--split", choices=["test", "val", "train"], default="test")
    p.add_argument("--run-dir", default=None,
                   help="Dir with the run's caches (default: checkpoint's parent)")
    p.add_argument("--out-dir", default=None,
                   help="Output dir (default: {run-dir}/gate_sweep)")
    # --- must match the training run ---
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    # --- measurement ---
    p.add_argument("--arch", default=ARCH)
    p.add_argument("--n-runs", type=int, default=2)
    p.add_argument("--nsys-timeout", type=int, default=300)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--hecbench-src", default=None)
    p.add_argument("--tmp-dir", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = Path(args.checkpoint)
    run_dir = Path(args.run_dir) if args.run_dir else ckpt.parent
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "gate_sweep"
    tmp_dir = Path(args.tmp_dir) if args.tmp_dir else run_dir / "eval_tmp"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Precheck cache + split (must reproduce training split) ────────────
    src = Path(args.hecbench_src) if args.hecbench_src else HECBENCH_SRC
    benchmarks = discover_benchmarks(src)
    benchmarks, _, loop_records_map, normalizer = precheck_benchmarks(
        benchmarks, run_dir / "eligible_benchmarks.json", skip=True)
    train_b, val_b, test_b = split_benchmarks(
        benchmarks, args.val_ratio, args.test_ratio, args.split_seed)
    chosen = {"train": train_b, "val": val_b, "test": test_b}[args.split]
    log.info("%s split: %d benchmarks", args.split, len(chosen))

    # ── Train feature matrix for the novelty signal ───────────────────────
    train_matrix = [
        normalizer.normalize(
            torch.tensor(rec["pre_features_raw"], dtype=torch.float32)
        ).tolist()
        for b in train_b for rec in loop_records_map.get(b.name, [])
    ]
    log.info("Novelty reference: %d training loops", len(train_matrix))

    # ── Caches ────────────────────────────────────────────────────────────
    baseline_cache = measure_baselines(
        chosen, loop_records_map=loop_records_map, arch=args.arch,
        n_runs=args.n_runs, nsys_timeout=args.nsys_timeout,
        tmp_dir=tmp_dir, gpu_id=args.gpu_id,
        cache_file=run_dir / "baseline_cache.json",
    )
    rc_file = run_dir / "reward_cache.json"
    rc_data = json.loads(rc_file.read_text()) if rc_file.exists() else {}
    reward_cache: dict = rc_data.get("rewards", {})
    postf_cache: dict = rc_data.get("post_features", {})
    log.info("Reward cache: %d entries", len(reward_cache))

    # ── Env + agent ───────────────────────────────────────────────────────
    env = GpuLoopEnv(arch=args.arch, n_runs=args.n_runs,
                     nsys_timeout=args.nsys_timeout, tmp_dir=tmp_dir,
                     gpu_id=args.gpu_id, normalizer=normalizer,
                     baseline_cache=baseline_cache)
    agent = Agent(device=device)
    agent.load(str(ckpt))

    # ── Instrumented greedy evaluation ────────────────────────────────────
    loop_rows: list = []
    fresh = 0
    for bench in chosen:
        try:
            first = env.reset(bench)
        except Exception as e:
            log.warning("reset failed for %s — skipping: %s", bench.name, e)
            continue
        if first is None:
            continue

        for lr in env.eligible_loops:
            pre = lr.pre_features.to(device)
            value = agent.predict_value(pre)
            novelty = min_distance(lr.pre_features.tolist(), train_matrix)

            unmerge, lp1 = agent.select_unmerge(pre, greedy=True)
            # Study A: unmerge==0 is a pure no-op — no factor decision, reward 0.
            # (Selecting a factor here and compiling unmerge=0,factor>1 would
            # invoke the confounded unroll-only path and misreport the reward.)
            if unmerge == 0:
                factor_idx, factor = 0, 1
                confidence = float(torch.exp(lp1))
                reward, cached = 0.0, True
                loop_rows.append({
                    "benchmark": bench.name, "loop_idx": lr.loop_idx,
                    "unmerge": 0, "factor": 1,
                    "reward": 0.0, "value": round(value, 6),
                    "novelty": round(novelty, 4),
                    "confidence": round(confidence, 4), "cached": 1,
                })
                continue

            pf = postf_cache.get(f"{bench.name}|{lr.loop_idx}")
            if pf is not None:
                step2 = torch.tensor(pf, dtype=torch.float32).to(device)
            else:
                try:
                    step2 = env.get_post_unmerge_features(lr).to(device)
                except Exception:
                    step2 = pre
            factor_idx, lp2, _ = agent.select_factor(
                step2, trip_known=lr.trip_count_known,
                trip_count=lr.trip_count, loop_idx=lr.loop_idx, greedy=True)
            factor = FACTOR_VALUES[factor_idx]
            confidence = float(torch.exp(lp1) * torch.exp(lp2))

            # --- Reward (unmerge==1 only): cache → measure ---
            cached = True
            key = f"{bench.name}|{lr.loop_idx}|{unmerge}|{factor}"
            hit = reward_cache.get(key)
            if hit is not None:
                reward = float(hit)
            else:
                cached = False
                fresh += 1
                kernel_filter, baseline_ms = env._resolve_measurement(lr)
                try:
                    ok = compile_single_loop(
                        env._benchmark_dir, loop_idx=lr.loop_idx,
                        unmerge=unmerge, factor=factor,
                        filename=lr.filename, triple=lr.triple,
                        arch=args.arch)
                except subprocess.TimeoutExpired:
                    ok, reward = False, -1.0
                if ok:
                    try:
                        modified = measure_kernel_time(
                            env._benchmark_dir, arch=args.arch,
                            n_runs=args.n_runs,
                            nsys_timeout=args.nsys_timeout,
                            tmp_dir=tmp_dir, gpu_id=args.gpu_id,
                            kernel_filter=kernel_filter)
                        reward = max(
                            (baseline_ms - modified) / max(baseline_ms, 1e-9),
                            -1.0)
                    except RuntimeError:
                        reward = 0.0
                elif reward != -1.0:
                    reward = 0.0     # compile error → no-op fallback
                reward_cache[key] = reward

            loop_rows.append({
                "benchmark": bench.name, "loop_idx": lr.loop_idx,
                "unmerge": unmerge, "factor": factor,
                "reward": round(reward, 6), "value": round(value, 6),
                "novelty": round(novelty, 4),
                "confidence": round(confidence, 4), "cached": int(cached),
            })
            log.info("  %s loop=%d u=%d f=%d r=%+.4f V=%+.4f nov=%.2f%s",
                     bench.name, lr.loop_idx, unmerge, factor, reward,
                     value, novelty, "" if cached else " [measured]")

    log.info("Evaluated %d loops (%d fresh measurements)", len(loop_rows), fresh)

    # ── Persist merged reward cache ───────────────────────────────────────
    if fresh:
        try:
            rc_data["rewards"] = reward_cache
            rc_data.setdefault("post_features", postf_cache)
            rc_file.write_text(json.dumps(rc_data))
            log.info("Reward cache updated (+%d entries)", fresh)
        except Exception as e:
            log.warning("Could not update reward cache: %s", e)

    # ── Outputs ───────────────────────────────────────────────────────────
    tag = f"{args.split}_{ckpt.stem}"
    with open(out_dir / f"loops_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(loop_rows[0].keys()))
        w.writeheader()
        w.writerows(loop_rows)

    curves = (compute_gate_curves(loop_rows, "value", act_if_above=True)
              + compute_gate_curves(loop_rows, "novelty", act_if_above=False)
              + compute_gate_curves(loop_rows, "confidence", act_if_above=True))
    with open(out_dir / f"gate_curves_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(curves[0].keys()))
        w.writeheader()
        w.writerows(curves)

    # ── Summary ───────────────────────────────────────────────────────────
    reg = [l["reward"] for l in loop_rows if l["reward"] < -0.01]
    ok_ = [l["reward"] for l in loop_rows if l["reward"] >= -0.01]
    log.info("")
    log.info("=== %s | %s ===", args.split, ckpt.name)
    ungated = next(c for c in curves if c["coverage"] == 1.0)
    log.info("ungated:  macro=%+.4f  %dW/%dN/%dR",
             ungated["macro_mean"], ungated["n_win"],
             ungated["n_neutral"], ungated["n_regression"])
    for gate in ["value", "novelty", "confidence"]:
        safe = [c for c in curves if c["gate"] == gate and c["n_regression"] == 0]
        if safe:
            b = max(safe, key=lambda c: c["macro_mean"])
            log.info("%-10s best zero-regression: tau=%s coverage=%.0f%% "
                     "macro=%+.4f %dW/%dN/0R",
                     gate, b["threshold"], 100 * b["coverage"],
                     b["macro_mean"], b["n_win"], b["n_neutral"])
        else:
            log.info("%-10s no threshold removes all regressions", gate)
    if reg and ok_:
        vals_r = [l["novelty"] for l in loop_rows if l["reward"] < -0.01]
        vals_o = [l["novelty"] for l in loop_rows if l["reward"] >= -0.01]
        log.info("novelty rank-AUC (regressed above others): %.3f",
                 rank_auc(vals_r, vals_o))
    log.info("Outputs: %s", out_dir.resolve())


if __name__ == "__main__":
    main()
