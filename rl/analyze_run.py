"""
Tier-1 (zero-GPU) post-run analysis.  No torch, no GPU, no compilation —
everything is derived from the training log + the run's precheck cache.

Three analyses:

  A. Model re-selection — recompute alternative "best checkpoint" criteria
     (macro per-benchmark mean, risk-adjusted, worst-benchmark, CVaR) from the
     per-loop val rewards logged every epoch, and report which epoch each
     criterion would have selected.

  B. Novelty analysis — for every test loop, distance in normalized feature
     space to the nearest training loop; correlate novelty and the critic's
     logged V(s) with the measured test outcome.  Answers: would a novelty
     gate / value gate have separated the regressions from the wins?

  C. Offline gate sweep — because a gated loop scores exactly 0 and gating
     never changes other loops' decisions, the full threshold-sweep curve for
     the tested checkpoint is computable from the logged per-loop test rewards.
     Produces coverage vs. gain vs. regression curves for both gates.

Usage (adjust paths):
  python analyze_run.py --log train.log --run-dir checkpoints/run_long \
                        [--split-seed 42 --val-ratio 0.15 --test-ratio 0.15] \
                        [--out-dir analysis]

Outputs in --out-dir:
  selection_criteria.csv   per-epoch value of every selection criterion
  test_loops.csv           per test loop: reward, V(s), novelty distance
  gate_curves_offline.csv  sweep curves for value gate and novelty gate
  (verdict summary printed to stdout)
"""

import argparse
import csv
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

# Parallel val line:  "[7/125]   [val W2] contract-cuda loop_idx=14 reward=0.0812 V(s)=0.0790"
_VAL_PAR = re.compile(
    r"\[(\d+)/\d+\]\s+\[val W\d+\]\s+(\S+)\s+loop_idx=(\d+)\s+"
    r"reward=(-?[0-9.]+)\s+V\(s\)=(-?[0-9.]+)"
)
# Sequential evaluate() line (val or test):
#   "[test] geglu-cuda loop_idx=3 unmerge=0 factor=4 reward=0.3723 V(s)=0.1102"
_EVAL_SEQ = re.compile(
    r"\[(val|test)\]\s+(\S+)\s+loop_idx=(\d+)\s+unmerge=(\d)\s+factor=(\d+)\s+"
    r"reward=(-?[0-9.]+)\s+V\(s\)=(-?[0-9.]+)"
)


def parse_log(log_path: Path):
    """
    Returns:
      val_by_epoch: {epoch: {benchmark: [reward, ...]}}
      test_loops:   {(benchmark, loop_idx): {"reward","value","unmerge","factor"}}
    """
    val_by_epoch: dict = defaultdict(lambda: defaultdict(list))
    test_loops: dict = {}

    for line in open(log_path, errors="replace"):
        m = _VAL_PAR.search(line)
        if m:
            epoch, bench = int(m.group(1)), m.group(2)
            val_by_epoch[epoch][bench].append(float(m.group(4)))
            continue
        m = _EVAL_SEQ.search(line)
        if m:
            label, bench, idx = m.group(1), m.group(2), int(m.group(3))
            if label == "test":
                # keep the LAST occurrence (final test eval)
                test_loops[(bench, idx)] = {
                    "reward":  float(m.group(6)),
                    "value":   float(m.group(7)),
                    "unmerge": int(m.group(4)),
                    "factor":  int(m.group(5)),
                }
    return val_by_epoch, test_loops


# ---------------------------------------------------------------------------
# Selection criteria
# ---------------------------------------------------------------------------

def cvar(rewards: list, frac: float = 0.10) -> float:
    """Mean of the worst *frac* of rewards (tail risk)."""
    if not rewards:
        return float("nan")
    k = max(1, int(len(rewards) * frac))
    return sum(sorted(rewards)[:k]) / k


def selection_criteria(val_by_epoch: dict) -> list:
    """One row of criterion values per epoch."""
    rows = []
    for epoch in sorted(val_by_epoch):
        bench_rewards = val_by_epoch[epoch]
        bench_avgs = {b: sum(rs) / len(rs) for b, rs in bench_rewards.items()}
        all_loops = [r for rs in bench_rewards.values() for r in rs]
        regressions = [a for a in bench_avgs.values() if a < -0.01]
        macro = sum(bench_avgs.values()) / len(bench_avgs)
        rows.append({
            "epoch":          epoch,
            "micro_mean":     sum(all_loops) / len(all_loops),
            "macro_mean":     macro,
            # macro mean penalised by mean regression magnitude (lambda = 2)
            "risk_adjusted":  macro - 2.0 * (sum(-a for a in regressions)
                                             / max(len(bench_avgs), 1)),
            "worst_benchmark": min(bench_avgs.values()),
            "cvar10":         cvar(all_loops, 0.10),
            "n_regressing_benchmarks": len(regressions),
        })
    return rows


# ---------------------------------------------------------------------------
# Split reproduction + features (no torch)
# ---------------------------------------------------------------------------

def reproduce_split(eligible_names: list, val_ratio: float, test_ratio: float,
                    seed: int):
    """MUST mirror train.split_benchmarks exactly (same rng, same rounding)."""
    rng = random.Random(seed)
    shuffled = list(eligible_names)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test = max(1, round(n * test_ratio))
    n_val = max(1, round(n * val_ratio))
    n_train = max(1, n - n_val - n_test)
    n_val = max(0, n - n_train - n_test)
    return (shuffled[:n_train],
            shuffled[n_train:n_train + n_val],
            shuffled[n_train + n_val:])


def normalize(vec: list, mean: list, std: list) -> list:
    return [(v - m) / s for v, m, s in zip(vec, mean, std)]


def min_distance(vec: list, matrix: list) -> float:
    """Euclidean distance to the nearest row of *matrix*."""
    best = float("inf")
    for row in matrix:
        d = 0.0
        for a, b in zip(vec, row):
            d += (a - b) * (a - b)
            if d >= best:      # early exit
                break
        if d < best:
            best = d
    return math.sqrt(best)


# ---------------------------------------------------------------------------
# Rank statistics + gate curves
# ---------------------------------------------------------------------------

def rank_auc(group_a: list, group_b: list) -> float:
    """
    P(random a > random b), Mann-Whitney U / (n_a * n_b), ties averaged.
    0.5 = no separation; 1.0 = every a above every b; 0.0 = every a below.
    """
    if not group_a or not group_b:
        return float("nan")
    combined = sorted([(v, 0) for v in group_a] + [(v, 1) for v in group_b],
                      key=lambda t: t[0])
    rank_sum_a = 0.0
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1          # 1-based, ties averaged
        for k in range(i, j + 1):
            if combined[k][1] == 0:
                rank_sum_a += avg_rank
        i = j + 1
    n_a, n_b = len(group_a), len(group_b)
    u_a = rank_sum_a - n_a * (n_a + 1) / 2
    return u_a / (n_a * n_b)


def compute_gate_curves(loops: list, signal_key: str, act_if_above: bool,
                        n_thresholds: int = 15) -> list:
    """
    Sweep a gate over per-loop records.

    loops: dicts with at least {benchmark, reward, <signal_key>}.
    act_if_above=True  → transform only when signal >= threshold (value gate)
    act_if_above=False → transform only when signal <= threshold (novelty gate)
    A gated loop scores exactly 0 (the no-op reproduces the baseline).
    """
    vals = sorted(l[signal_key] for l in loops)
    # threshold grid over the observed signal range (quantiles, incl. extremes)
    grid = [vals[min(len(vals) - 1, int(q * (len(vals) - 1)))]
            for q in [i / (n_thresholds - 1) for i in range(n_thresholds)]]
    rows = []
    for tau in grid:
        by_bench: dict = defaultdict(list)
        acted = 0
        for l in loops:
            act = (l[signal_key] >= tau) if act_if_above else (l[signal_key] <= tau)
            r = l["reward"] if act else 0.0
            acted += int(act)
            by_bench[l["benchmark"]].append(r)
        bench_avgs = {b: sum(rs) / len(rs) for b, rs in by_bench.items()}
        micro = sum(sum(rs) for rs in by_bench.values()) / len(loops)
        rows.append({
            "gate":       signal_key,
            "threshold":  round(tau, 6),
            "coverage":   round(acted / len(loops), 4),
            "micro_mean": round(micro, 6),
            "macro_mean": round(sum(bench_avgs.values()) / len(bench_avgs), 6),
            "n_win":      sum(1 for a in bench_avgs.values() if a > 0.01),
            "n_neutral":  sum(1 for a in bench_avgs.values() if -0.01 <= a <= 0.01),
            "n_regression": sum(1 for a in bench_avgs.values() if a < -0.01),
            "worst_benchmark": round(min(bench_avgs.values()), 6),
        })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log", required=True, help="Training log file")
    p.add_argument("--run-dir", required=True,
                   help="Checkpoint dir with eligible_benchmarks.json")
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--out-dir", default="analysis")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Parse log ─────────────────────────────────────────────────────────
    val_by_epoch, test_loops = parse_log(Path(args.log))
    print(f"Parsed: {len(val_by_epoch)} epochs of val data, "
          f"{len(test_loops)} test loop results")
    if not val_by_epoch and not test_loops:
        print("ERROR: nothing parsed — check the log path/format.")
        sys.exit(1)

    # ── A. Selection criteria ─────────────────────────────────────────────
    if val_by_epoch:
        rows = selection_criteria(val_by_epoch)
        with open(out / "selection_criteria.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("\n=== A. Best epoch under each selection criterion ===")
        for crit in ["micro_mean", "macro_mean", "risk_adjusted",
                     "worst_benchmark", "cvar10"]:
            best = max(rows, key=lambda r: r[crit])
            print(f"  {crit:<18} → epoch {best['epoch']:>4}  "
                  f"({crit}={best[crit]:+.4f}, "
                  f"regressing_benchmarks={best['n_regressing_benchmarks']})")

    # ── B. Novelty analysis ───────────────────────────────────────────────
    cache_file = Path(args.run_dir) / "eligible_benchmarks.json"
    if not cache_file.exists():
        print(f"\nERROR: {cache_file} not found — cannot run novelty analysis.")
        sys.exit(1)
    data = json.loads(cache_file.read_text())
    eligible = data["eligible"]
    records = data["loop_records"]
    norm = data.get("normalizer", {})
    mean, std = norm.get("mean"), norm.get("std")
    if not (norm.get("fitted") and mean):
        print("WARNING: no fitted normalizer in cache — using raw features.")
        n_feat = len(next(iter(records.values()))[0]["pre_features_raw"])
        mean, std = [0.0] * n_feat, [1.0] * n_feat

    train_names, _, test_names = reproduce_split(
        eligible, args.val_ratio, args.test_ratio, args.split_seed)
    print(f"\nSplit reproduced: train={len(train_names)} test={len(test_names)}")

    train_matrix = [
        normalize(rec["pre_features_raw"], mean, std)
        for name in train_names for rec in records.get(name, [])
    ]
    print(f"Train feature matrix: {len(train_matrix)} loops")

    loop_rows = []
    for name in test_names:
        for rec in records.get(name, []):
            key = (name, rec["loop_idx"])
            if key not in test_loops:
                continue
            vec = normalize(rec["pre_features_raw"], mean, std)
            entry = dict(test_loops[key])
            entry.update({
                "benchmark": name,
                "loop_idx":  rec["loop_idx"],
                "novelty":   round(min_distance(vec, train_matrix), 4),
            })
            loop_rows.append(entry)

    with open(out / "test_loops.csv", "w", newline="") as f:
        fields = ["benchmark", "loop_idx", "unmerge", "factor",
                  "reward", "value", "novelty"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in loop_rows:
            w.writerow({k: r[k] for k in fields})
    print(f"Matched {len(loop_rows)} test loops with log results")

    reg = [l for l in loop_rows if l["reward"] < -0.01]
    ok  = [l for l in loop_rows if l["reward"] >= -0.01]
    if reg and ok:
        med = lambda xs: sorted(xs)[len(xs) // 2]
        print("\n=== B. Do the gate signals separate regressions? ===")
        print(f"  {len(reg)} regressing loops vs {len(ok)} non-regressing")
        print(f"  novelty  median: regressed={med([l['novelty'] for l in reg]):.3f}  "
              f"others={med([l['novelty'] for l in ok]):.3f}  "
              f"rank-AUC={rank_auc([l['novelty'] for l in reg], [l['novelty'] for l in ok]):.3f} "
              f"(1.0 = regressions are always the most novel)")
        print(f"  V(s)     median: regressed={med([l['value'] for l in reg]):.4f}  "
              f"others={med([l['value'] for l in ok]):.4f}  "
              f"rank-AUC={rank_auc([l['value'] for l in reg], [l['value'] for l in ok]):.3f} "
              f"(0.0 = regressions always have the LOWEST V(s))")
    else:
        print("\n=== B. No regressing loops matched — nothing to separate ===")

    # ── C. Offline gate sweep ─────────────────────────────────────────────
    if loop_rows:
        curves = (compute_gate_curves(loop_rows, "value", act_if_above=True)
                  + compute_gate_curves(loop_rows, "novelty", act_if_above=False))
        with open(out / "gate_curves_offline.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(curves[0].keys()))
            w.writeheader()
            w.writerows(curves)

        print("\n=== C. Gate sweep (offline, from logged rewards) ===")
        ungated = [c for c in curves if c["coverage"] == 1.0]
        if ungated:
            u = ungated[0]
            print(f"  ungated:            macro={u['macro_mean']:+.4f}  "
                  f"{u['n_win']}W/{u['n_neutral']}N/{u['n_regression']}R")
        for gate in ["value", "novelty"]:
            safe = [c for c in curves if c["gate"] == gate and c["n_regression"] == 0]
            if safe:
                best = max(safe, key=lambda c: c["macro_mean"])
                print(f"  {gate:<8} best zero-regression point: "
                      f"tau={best['threshold']}  coverage={best['coverage']:.0%}  "
                      f"macro={best['macro_mean']:+.4f}  "
                      f"{best['n_win']}W/{best['n_neutral']}N/0R")
            else:
                print(f"  {gate:<8} no threshold eliminates all regressions")

    print(f"\nAll outputs in: {out.resolve()}")


if __name__ == "__main__":
    main()
