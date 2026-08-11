"""
Is there room for a top-k factor policy, and what does the reward look like?

WHY THIS EXISTS
---------------
factor_lowrank.py showed that a loop's observed cells do not predict its
unobserved ones under SQUARED ERROR. That is not the same as showing the
ordering is unlearnable, and the decision only ever consumes the ordering. A
model can be useless at predicting a cell's value and still put the good cells
near the top -- especially when most cells are harmful and the real question is
which REGION is safe rather than which cell is best.

It is also worth measuring because top-k is deployable on its own terms: cutting
19 candidates to 3 is a 6x reduction in autotuning cost, which is what learned
cost models in TVM/Ansor actually deliver -- they prune the space, they do not
pick the winner.

WHAT IT REPORTS, AND WHY THE BASELINES ARE THE POINT
----------------------------------------------------
A top-3 capture number on its own means nothing. Two things bound it:

  random-k    pick k legal factors at random, take the best, decline if all
              are worse than no-op. Computed EXACTLY, not sampled: for an arm
              whose sorted cells are r_1..r_n, a random k-subset has maximum
              r_i with probability C(i-1,k-1)/C(n,k).

  constant-k  the best FIXED set of k factors -- "always try {2,4,8}". Two
              versions: chosen on the same arms it is scored on (an upper bound
              on any fixed strategy) and chosen on one half of the benchmarks
              and scored on the other (what you could actually ship).

If constant-3 already sits near the oracle, there is no room for a model at k=3
and the answer is to ship the constant set. If there is a gap, that gap is the
budget a learned top-3 has to earn.

Every policy may DECLINE: realised = max(0, best of the tried cells), because
the no-op is free. Capture is restricted to arms with headroom, matching
offline_data.score_decisions.

Ran-only, same rules as factor_signal.py. Pure stdlib.

Usage:
  python3 factor_topk.py RUN_DIR
  python3 factor_topk.py RUN_DIR --deadzone 0.005 --max-k 5
"""

import argparse
import json
import random
import statistics as stats
import sys
from itertools import combinations
from math import comb
from pathlib import Path

# MIRRORED from offline_data.py / agent.py — see factor_signal.py.
FAILURE_VALUES = (-0.16, -0.161)
CLIP_FLOOR = -1.0
IDX_TRIP_KNOWN = 10
IDX_TRIP_COUNT = 11
FACTOR_VALUES = tuple(range(1, 11))


def legal_factors(raw: list, unmerge: int) -> set:
    """MIRROR: label_loops.valid_factors + category_factor_mask."""
    if raw[IDX_TRIP_KNOWN] > 0.5 and int(raw[IDX_TRIP_COUNT]) > 0:
        tc = int(raw[IDX_TRIP_COUNT])
        ok = {f for f in FACTOR_VALUES if f == 1 or f <= tc}
    else:
        ok = set(FACTOR_VALUES)
    return ok - {1} if unmerge == 0 else ok


def realised(cells: dict, tried) -> float:
    """What a policy earns on one arm. Declining is free and always available,
    so anything worse than the no-op costs 0.0, not its own value."""
    got = [cells[f] for f in tried if f in cells]
    return max(0.0, max(got)) if got else 0.0


def expected_random_k(cells: dict, k: int) -> float:
    """
    EXACT expected realised value of k legal factors drawn at random.

    Not sampled. Sorting the arm's cells ascending, a random k-subset has its
    maximum at r_i with probability C(i-1, k-1)/C(n, k), so the expectation is a
    weighted sum over the order statistics. Sampling here would put Monte Carlo
    noise into the baseline that the model is judged against.
    """
    r = sorted(cells.values())
    n = len(r)
    if k >= n:
        return max(0.0, r[-1])
    denom = comb(n, k)
    return sum(max(0.0, r[i]) * comb(i, k - 1) for i in range(n)) / denom


def capture(arms: dict, chooser, deadzone: float) -> float:
    """sum(realised) / sum(oracle) over arms that have headroom.

    A ratio of sums, not a mean of per-arm ratios: arms differ enormously in how
    much is available, and averaging ratios lets the ones with almost nothing at
    stake dominate.
    """
    num = den = 0.0
    for key, cells in arms.items():
        orc = max(0.0, max(cells.values()))
        if orc <= deadzone:
            continue
        num += chooser(key, cells)
        den += orc
    return num / den if den > 0 else float("nan")


def best_constant_set(arms: dict, k: int, deadzone: float) -> tuple:
    """(set, capture) for the best fixed k factors over these arms. C(10,k) is
    at most 252, so this is exhaustive rather than greedy."""
    best = (None, -float("inf"))
    for S in combinations(FACTOR_VALUES, k):
        c = capture(arms, lambda _k, cells, S=S: realised(cells, S), deadzone)
        if c == c and c > best[1]:
            best = (S, c)
    return best


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--deadzone", type=float, default=0.005)
    p.add_argument("--max-k", type=int, default=5)
    p.add_argument("--min-cells", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rc_f = args.run_dir / "reward_cache.json"
    if not rc_f.exists():
        sys.exit(f"missing {rc_f}")
    rewards = json.loads(rc_f.read_text()).get("rewards", {})

    elig_f = args.run_dir / "eligible_benchmarks.json"
    raws = {}
    if elig_f.exists():
        for b, recs in json.loads(elig_f.read_text()).get("loop_records", {}).items():
            for r in recs:
                raws[(b, int(r["loop_idx"]))] = r["pre_features_raw"]
    else:
        print(f"WARNING: no {elig_f} — trip-count mask not applied.\n")

    arms: dict = {}
    n_fail = n_clip = 0
    for key, v in rewards.items():
        parts = key.split("|")
        if len(parts) != 4:
            continue
        try:
            b, li, u, f = parts[0], int(parts[1]), int(parts[2]), int(parts[3])
        except ValueError:
            continue
        v = float(v)
        if any(abs(v - x) < 1e-9 for x in FAILURE_VALUES):
            n_fail += 1
            continue
        if v <= CLIP_FLOOR + 1e-9:
            n_clip += 1
            continue
        raw = raws.get((b, li))
        if raw is not None and f not in legal_factors(raw, u):
            continue
        arms.setdefault((b, li, u), {})[f] = v

    arms = {a: c for a, c in arms.items() if len(c) >= args.min_cells}
    if not arms:
        sys.exit("no arm has enough observed cells")

    allv = [v for c in arms.values() for v in c.values()]
    n = len(allv)
    below = sum(1 for v in allv if v < -args.deadzone)
    flat = sum(1 for v in allv if abs(v) <= args.deadzone)
    above = n - below - flat
    q = sorted(allv)

    print(f"{len(arms)} arms, {n} ran cells "
          f"({n_fail} failures and {n_clip} clip cells excluded)\n")
    print("WHERE THE CELLS SIT RELATIVE TO NO-OP")
    print(f"  harmful  (< -dz) {below:6d}  {below/n:6.1%}")
    print(f"  neutral  (|r|<dz){flat:6d}  {flat/n:6.1%}")
    print(f"  helpful  (> +dz) {above:6d}  {above/n:6.1%}")
    print(f"  quantiles p10 {q[n//10]:+.4f}  p50 {q[n//2]:+.4f}  "
          f"p90 {q[9*n//10]:+.4f}")
    print(f"  mean {stats.fmean(allv):+.4f}   <- a transform picked blind is "
          f"worth this\n")

    # Deployable constant set: chosen on one half of the BENCHMARKS, scored on
    # the other, both ways round. Split by benchmark, not by arm — arms in one
    # benchmark share source and would leak the choice across the boundary.
    benches = sorted({k[0] for k in arms})
    random.Random(args.seed).shuffle(benches)
    half = set(benches[:len(benches) // 2])
    A = {k: c for k, c in arms.items() if k[0] in half}
    B = {k: c for k, c in arms.items() if k[0] not in half}

    print("CAPTURE OF THE ORACLE, BY BUDGET k")
    print("  every policy may decline; capture is over arms with headroom\n")
    print(f"  {'k':>2} {'random-k':>10} {'const-k (fit here)':>20} "
          f"{'const-k (held out)':>20}  best set")
    for k in range(1, args.max_k + 1):
        rnd = capture(arms, lambda _k, cells, k=k: expected_random_k(cells, k),
                      args.deadzone)
        S, con = best_constant_set(arms, k, args.deadzone)
        SA, _ = best_constant_set(A, k, args.deadzone)
        SB, _ = best_constant_set(B, k, args.deadzone)
        # Each half scored under the set chosen on the OTHER half, then pooled.
        ho = capture({**{x: c for x, c in B.items()}},
                     lambda _k, cells, S=SA: realised(cells, S), args.deadzone)
        ho2 = capture({**{x: c for x, c in A.items()}},
                      lambda _k, cells, S=SB: realised(cells, S), args.deadzone)
        hom = stats.fmean([x for x in (ho, ho2) if x == x]) \
            if any(x == x for x in (ho, ho2)) else float("nan")
        print(f"  {k:>2} {rnd:9.1%} {con:19.1%} {hom:19.1%}  {S}")
    print(f"  all {capture(arms, lambda _k, c: max(0.0, max(c.values())), args.deadzone):8.1%}"
          f"   <- oracle, by definition")

    print("\n  READ — the gap between 'const-k (held out)' and the oracle is the "
          "room a\n  learned top-k policy has. Which of three worlds you are in "
          "is told by how\n  const-k compares to random-k:")
    print("\n    const-k >> random-k, and near the oracle")
    print("      One region of factors is good for everyone. Ship the constant "
          "set; a\n      model has nothing left to earn.")
    print("\n    const-k ~ random-k, and both far below the oracle")
    print("      The good factor MOVES per loop, so no fixed set can track it "
          "and a\n      fixed set is no better than sampling. This is the case "
          "where a learned\n      per-loop policy is the only thing that can "
          "close the gap — the widest\n      room, not the narrowest.")
    print("\n    oracle ~ const-k ~ random-k")
    print("      k is doing all the work: trying several candidates captures "
          "the value\n      whatever they are. Spend the budget, not the model.")
    print("\n  A model's top-k drops into this same table as another row: take "
          "its k\n  highest-scoring LEGAL factors, try them, keep the best, "
          "decline if all are\n  worse than no-op. Identical metric, so the "
          "comparison is direct.")


if __name__ == "__main__":
    main()
