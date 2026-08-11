"""
How many numbers does a loop's factor response actually take to describe?

The factor head predicts ten free values per loop. If the response shapes across
loops live in a low-dimensional family, a loop is described by k numbers with k
far below ten, and the feature map only has to predict those k.

    rank 1-3 clearly beats the null  ->  simple family; predict k numbers
    nothing beats the null           ->  no shared structure; stop

ROWS ARE CENTRED FIRST, AND THAT IS THE POINT
---------------------------------------------
74.7% of the reward variance in this cache is the loop's overall LEVEL, which
argmax cannot see. Left in, rank 1 would capture the level, "explain" most of
the variance, and mean nothing. Every row is centred before the fit, so what is
being ranked is the SHAPE of the response.

Centring uses TRAINING cells only — otherwise a held-out value leaks into the
row mean it is then scored against.

WHY EM-SVD AND NOT ALS
----------------------
The first version of this used alternating least squares from a random
initialisation. It was wrong in a way that looked like a result: rank 2 scored
-168% (worse than predicting nothing) while ranks 1 and 3 scored ~70%, and the
same data scaled by 100 gave different answers. ALS on this problem is
non-convex and was landing in different places on every run.

This fits by EM-SVD instead: fill the unobserved cells with 0 (which, after
centring, IS the null prediction), take the rank-k projection, re-impute, repeat.
Deterministic, no initialisation to get unlucky with, monotone in training error,
and it degrades to the null rather than to noise. The 10x10 eigenproblem is
solved by cyclic Jacobi, which is unconditionally convergent for symmetric input.

Ran-only, same rules as factor_signal.py: compile failures and the -1.0 clip are
not measurements of speed. Pure stdlib.

Usage:
  python3 factor_lowrank.py RUN_DIR
  python3 factor_lowrank.py RUN_DIR --max-rank 4 --repeats 5 --min-cells 4
"""

import argparse
import json
import random
import statistics as stats
import sys
from pathlib import Path

# MIRRORED from offline_data.py / agent.py — see factor_signal.py for why these
# are copied rather than imported.
FAILURE_VALUES = (-0.16, -0.161)
CLIP_FLOOR = -1.0
IDX_TRIP_KNOWN = 10
IDX_TRIP_COUNT = 11
FACTOR_VALUES = tuple(range(1, 11))
NF = len(FACTOR_VALUES)


def legal_factors(raw: list, unmerge: int) -> set:
    """MIRROR: label_loops.valid_factors + category_factor_mask. Factor 1 on the
    unroll branch IS the no-op, not a factor choice, so it is not a legal cell
    there and must not enter the matrix."""
    if raw[IDX_TRIP_KNOWN] > 0.5 and int(raw[IDX_TRIP_COUNT]) > 0:
        tc = int(raw[IDX_TRIP_COUNT])
        ok = {f for f in FACTOR_VALUES if f == 1 or f <= tc}
    else:
        ok = set(FACTOR_VALUES)
    return ok - {1} if unmerge == 0 else ok


def jacobi(A: list, sweeps: int = 60, tol: float = 1e-14) -> tuple:
    """
    (eigenvalues, eigenvectors) of a SYMMETRIC matrix, sorted descending.

    Cyclic Jacobi rotations: deterministic, and unconditionally convergent for
    symmetric input, which is why this rather than a power iteration or anything
    needing an initial guess. Eigenvectors are returned as rows of the second
    list, so `vec[a]` is the a-th one.
    """
    n = len(A)
    a = [row[:] for row in A]
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(sweeps):
        off = sum(a[i][j] ** 2 for i in range(n) for j in range(n) if i != j)
        if off < tol:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(a[p][q]) < 1e-18:
                    continue
                theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
                t = ((1.0 if theta >= 0 else -1.0)
                     / (abs(theta) + (theta * theta + 1.0) ** 0.5))
                c = 1.0 / (t * t + 1.0) ** 0.5
                s = t * c
                for i in range(n):                       # columns p, q
                    aip, aiq = a[i][p], a[i][q]
                    a[i][p] = c * aip - s * aiq
                    a[i][q] = s * aip + c * aiq
                for i in range(n):                       # rows p, q
                    api, aqi = a[p][i], a[q][i]
                    a[p][i] = c * api - s * aqi
                    a[q][i] = s * api + c * aqi
                for i in range(n):                       # accumulate rotations
                    vip, viq = v[i][p], v[i][q]
                    v[i][p] = c * vip - s * viq
                    v[i][q] = s * vip + c * viq
    ev = [(a[i][i], [v[r][i] for r in range(n)]) for i in range(n)]
    ev.sort(key=lambda e: -e[0])
    return [e[0] for e in ev], [e[1] for e in ev]


def em_svd(train: dict, n_rows: int, k: int, iters: int, tol: float) -> list:
    """
    Rank-k reconstruction of an incomplete matrix, by EM.

    `train` maps row -> {col: centred value}. Unobserved cells start at 0, which
    after centring is exactly the null prediction, so the fit begins at "no
    structure" and can only move away from it if the observed cells say so.

    Each step: project the current filled matrix onto its top-k right singular
    directions (via the k largest eigenvectors of X^T X), then overwrite the
    unobserved cells with that projection. Observed cells are always restored,
    so the fit is never allowed to drift off the data.
    """
    X = [[0.0] * NF for _ in range(n_rows)]
    for r, cells in train.items():
        for c, v in cells.items():
            X[r][c] = v

    R = [row[:] for row in X]
    for _ in range(iters):
        C = [[sum(X[r][i] * X[r][j] for r in range(n_rows)) for j in range(NF)]
             for i in range(NF)]
        _, vec = jacobi(C)
        Vk = vec[:k]
        R = []
        for r in range(n_rows):
            coef = [sum(X[r][j] * Vk[a][j] for j in range(NF)) for a in range(k)]
            R.append([sum(coef[a] * Vk[a][j] for a in range(k)) for j in range(NF)])
        delta = 0.0
        for r in range(n_rows):
            obs = train.get(r, {})
            for c in range(NF):
                if c in obs:
                    X[r][c] = obs[c]                     # observed: never moves
                else:
                    delta = max(delta, abs(X[r][c] - R[r][c]))
                    X[r][c] = R[r][c]                    # unobserved: imputed
        if delta < tol:
            break
    return R


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--max-rank", type=int, default=4)
    p.add_argument("--repeats", type=int, default=5,
                   help="Independent held-out draws. The spread across them is "
                        "how much of any improvement is the draw.")
    p.add_argument("--holdout", type=float, default=0.2)
    p.add_argument("--min-cells", type=int, default=4,
                   help="Arms with fewer observed cells cannot support a rank "
                        "fit and only add noise to the held-out score.")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--tol", type=float, default=1e-7)
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
        print(f"WARNING: no {elig_f} — trip-count mask not applied, so illegal "
              f"cells\n         are in the matrix.\n")

    arms: dict = {}
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
            continue
        if v <= CLIP_FLOOR + 1e-9:
            continue
        raw = raws.get((b, li))
        if raw is not None and f not in legal_factors(raw, u):
            continue
        arms.setdefault((b, li, u), {})[f] = v

    arms = {a: c for a, c in arms.items() if len(c) >= args.min_cells}
    if not arms:
        sys.exit("no arm has enough observed cells — lower --min-cells")

    keys = sorted(arms)
    col = {f: i for i, f in enumerate(FACTOR_VALUES)}
    cells = [(r, col[f], v) for r, a in enumerate(keys) for f, v in arms[a].items()]
    print(f"{len(keys)} arms x {NF} factors, {len(cells)} ran cells "
          f"({len(cells) / (len(keys) * NF):.0%} filled), "
          f"min {args.min_cells} cells per arm\n")
    print("Held-out prediction of a CENTRED response. Rows are centred on their")
    print("training cells only, so a held-out value cannot leak into the row mean")
    print("it is scored against. 'explained' is measured against predicting that")
    print("row mean, which is what no-shared-structure looks like.\n")
    print(f"  {'rank':>4} {'held-out RMSE':>20} {'explained':>14}")

    base, results = [], {}
    for rep in range(args.repeats):
        rng = random.Random(args.seed + rep)
        shuf = cells[:]
        rng.shuffle(shuf)
        n_hold = max(1, int(args.holdout * len(shuf)))
        hold, train = shuf[:n_hold], shuf[n_hold:]

        per_row: dict = {}
        for r, c, v in train:
            per_row.setdefault(r, {})[c] = v
        mean = {r: stats.fmean(cs.values()) for r, cs in per_row.items()}
        tr = {r: {c: v - mean[r] for c, v in cs.items()}
              for r, cs in per_row.items()}
        # An arm with no training cell left has no mean to centre on, so its
        # held-out cells are unscoreable rather than scored against zero.
        ho = [(r, c, v - mean[r]) for r, c, v in hold if r in mean]
        if not ho:
            continue

        base.append((sum(v * v for _, _, v in ho) / len(ho)) ** 0.5)
        for k in range(1, args.max_rank + 1):
            R = em_svd(tr, len(keys), k, args.iters, args.tol)
            se = sum((v - R[r][c]) ** 2 for r, c, v in ho)
            results.setdefault(k, []).append((se / len(ho)) ** 0.5)

    if not base:
        sys.exit("every held-out draw was empty — lower --holdout")

    b0 = stats.fmean(base)
    print(f"  {'0':>4} {b0:20.5f} {'— (the null)':>14}")
    best_k, best_ex = 0, 0.0
    for k in sorted(results):
        m = stats.fmean(results[k])
        sd = stats.pstdev(results[k]) if len(results[k]) > 1 else 0.0
        ex = 1.0 - (m / b0) ** 2
        if ex > best_ex:
            best_k, best_ex = k, ex
        print(f"  {k:>4} {m:13.5f} +-{sd:.5f} {ex:13.1%}")

    print()
    if best_ex < 0.05:
        print("  READ: nothing beats predicting the row mean. The response "
              "shapes share no\n  low-dimensional structure across loops, so "
              "there is no k-number summary\n  for a feature map to predict. "
              "The factor line ends here.")
    elif best_ex < 0.20:
        print(f"  READ: rank {best_k} explains only {best_ex:.0%} of the "
              f"held-out shape. Some shared\n  structure, but a feature map "
              f"predicting it inherits that ceiling.")
    else:
        print(f"  READ: rank {best_k} explains {best_ex:.0%} of the held-out "
              f"shape. The family IS\n  low-dimensional — a loop's response is "
              f"{best_k} number(s), not ten. That is what\n  the feature map "
              f"should predict, and it is a far smaller job than choosing\n  "
              f"among ten free outputs.")
    print("\n  This says nothing about whether the 93 FEATURES can predict "
          "those numbers.\n  It says only whether the numbers exist. Regressing "
          "features onto them is\n  stage two, and is only worth doing if this "
          "stage passed.")


if __name__ == "__main__":
    main()
