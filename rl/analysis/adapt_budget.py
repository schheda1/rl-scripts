"""
Few-shot factor adaptation: does application-level factor preference exist, and
how should the measurement budget be spent?

THE MODEL
---------
    value(loop, f)  =  mu(f)  +  alpha_bench(f)  +  eps_loop(f)  +  noise
                       global    application        loop

Zero-shot tried to predict eps from features and could not. This targets alpha,
which is a different quantity and the one this study keeps finding is strong
(benchmark-dominant near the ceiling, high purity, the category few-shot working).

NO NETWORK, ON PURPOSE
----------------------
The trained factor head ties the marginal — it matched a deployable constant at
k=1,2,3 to within 0.1pp. So mu(f) IS what the head learned, and the empirical
marginal over the training benchmarks gives it for free. Everything here is a
table computation: deterministic, seconds, no GPU, nothing to misconfigure.

WHAT IT FIXES IN factor_adapt
-----------------------------
  estimator  650 parameters (net[6]) fitted to ~16 informative cells at a
             per-cell signal-to-noise near 0.6. Replaced by a shrinkage
             estimate that is pooled toward the global prior, so it CANNOT do
             worse than the prior. A 650-parameter fit can, and did.

  budget     alpha is estimated by AVERAGING OVER LOOPS, so its error falls as
             1/sqrt(n_loops). Measuring loops exhaustively spends the budget on
             factor resolution instead. The sweep below trades one against the
             other at fixed cost.

  selection  Adaptation loops are picked by shuffling. Two near-duplicate loops
             teach less than two representative ones, and at n=2 that matters.
             'central' picks nearest the benchmark's feature centroid; 'randsel'
             keeps the current behaviour so the difference is measured.

ONE VALUE RULE
--------------
A cell is worth what it realises: 0.0 if it failed to build or is absent from the
cache (the transform did not happen, the original program stands), otherwise its
stored reward. Clip-floor cells keep their -1.0 — they ran, and the cache cannot
say whether they timed out.

DEPLOYMENT MODEL
----------------
Measure a few loops of the target, then SHIP one factor for every remaining loop
with no further measurement. Picks are committed and charged, so these numbers
are comparable to factor_only's `probe` row, NOT to its top-k rows, which assume
you measure the shortlist and can still decline.

Usage:
  python3 adapt_budget.py RUN_DIR --deadzone 0.005
  python3 adapt_budget.py RUN_DIR --deadzone 0.005 --budget 20 --splits 5
"""

import argparse
import csv
import json
import random
import statistics as stats
import sys
from pathlib import Path

# MIRRORED from offline_data.py / agent.py — importing either pulls in torch.
FAILURE_VALUES = (-0.16, -0.161)
IDX_TRIP_KNOWN = 10
IDX_TRIP_COUNT = 11
FACTOR_VALUES = tuple(range(1, 11))
NF = len(FACTOR_VALUES)


def legal_factors(raw: list, unmerge: int) -> list:
    """MIRROR: label_loops.valid_factors + category_factor_mask. Factor 1 on the
    unroll branch IS the no-op, not a factor choice."""
    if raw[IDX_TRIP_KNOWN] > 0.5 and int(raw[IDX_TRIP_COUNT]) > 0:
        tc = int(raw[IDX_TRIP_COUNT])
        ok = [f for f in FACTOR_VALUES if f == 1 or f <= tc]
    else:
        ok = list(FACTOR_VALUES)
    return [f for f in ok if f != 1] if unmerge == 0 else ok


def load_arms(run_dir: Path, deadzone: float, min_cells: int) -> list:
    """
    One arm per labelled transform loop, on its TRUE branch — the population
    factor_only's probe scores, so the numbers line up.

    Arms with fewer than `min_cells` observed factors are dropped: every value
    below is centred on the arm's own mean, and a mean over two cells is too
    unstable to centre on.
    """
    for name in ("reward_cache.json", "eligible_benchmarks.json",
                 "loop_labels.csv"):
        if not (run_dir / name).exists():
            sys.exit(f"missing {run_dir / name}")
    rc = json.loads((run_dir / "reward_cache.json").read_text())
    elig = json.loads((run_dir / "eligible_benchmarks.json").read_text())
    raws = {(b, int(r["loop_idx"])): r["pre_features_raw"]
            for b, recs in elig.get("loop_records", {}).items() for r in recs}

    labels = {}
    with open(run_dir / "loop_labels.csv") as fh:
        for r in csv.DictReader(fh):
            if r.get("labelable") == "1":
                labels[(r["benchmark"], int(r["loop_idx"]))] = r["category"]

    cells: dict = {}
    for key, v in rc.get("rewards", {}).items():
        p = key.split("|")
        if len(p) != 4:
            continue
        try:
            b, li, u, f = p[0], int(p[1]), int(p[2]), int(p[3])
        except ValueError:
            continue
        v = float(v)
        if any(abs(v - x) < 1e-9 for x in FAILURE_VALUES):
            v = 0.0                      # failed to build -> original stands
        cells.setdefault((b, li, u), {})[f] = v

    arms = []
    for (b, li), cat in sorted(labels.items()):
        if cat == "noop":
            continue                     # no factor question on a no-op loop
        u = 1 if cat == "unmerge_unroll" else 0
        raw = raws.get((b, li))
        if raw is None:
            continue
        legal = legal_factors(raw, u)
        got = cells.get((b, li, u), {})
        val = {f: got[f] for f in legal if f in got}
        if len(val) < min_cells:
            continue
        orc = max(val.values())
        if orc <= deadzone:
            continue                     # nothing to win; not in the denominator
        m = stats.fmean(val.values())
        arms.append({"bench": b, "loop": li, "u": u, "legal": legal,
                     "val": val, "cen": {f: y - m for f, y in val.items()},
                     "orc": orc, "feat": raw})
    return arms


def realised(arm: dict, f: int) -> float:
    """A committed pick, charged. Absent from the cache means it was never
    built, which leaves the program unchanged at 0.0."""
    return arm["val"].get(f, 0.0)


# ---------------------------------------------------------------------------
# Global prior and variance components — TRAIN benchmarks only
# ---------------------------------------------------------------------------

def global_prior(train: list) -> dict:
    """mu(f): mean centred value of each factor over the training arms. What the
    trained head converged to, without training it."""
    acc: dict = {}
    for a in train:
        for f, y in a["cen"].items():
            acc.setdefault(f, []).append(y)
    return {f: stats.fmean(v) for f, v in acc.items()}


def variance_components(train: list) -> tuple:
    """
    (tau2, sigma2): between-application and within-application variance of the
    factor shape, by one-way random effects per factor, averaged over factors.

    tau2 is not an optional diagnostic — the shrinkage weight needs it — and it
    is also the ceiling on the entire idea. If applications do not differ in
    which factors they like, tau2 is ~0, every weight collapses to 0, and no
    adaptation scheme of any kind beats the global prior.
    """
    by_f: dict = {}
    for a in train:
        for f, y in a["cen"].items():
            by_f.setdefault(f, {}).setdefault(a["bench"], []).append(y)

    taus, sigmas = [], []
    for groups in by_f.values():
        k, N = len(groups), sum(len(v) for v in groups.values())
        if k < 2 or N <= k:
            continue
        grand = stats.fmean(y for v in groups.values() for y in v)
        msw = sum((y - stats.fmean(v)) ** 2
                  for v in groups.values() for y in v) / (N - k)
        msb = sum(len(v) * (stats.fmean(v) - grand) ** 2
                  for v in groups.values()) / (k - 1)
        # Effective group size for an unbalanced design, not the plain mean.
        n0 = (N - sum(len(v) ** 2 for v in groups.values()) / N) / (k - 1)
        if n0 <= 0:
            continue
        sigmas.append(msw)
        taus.append(max(0.0, (msb - msw) / n0))
    if not taus:
        return 0.0, 0.0
    return stats.fmean(taus), stats.fmean(sigmas)


# ---------------------------------------------------------------------------
# Adaptation
# ---------------------------------------------------------------------------

def pick_adapt_loops(arms: list, n: int, how: str, seed: int,
                     mean: list, sd: list) -> list:
    """
    Which loops of the target to spend the budget on.

    'central' takes the loops nearest the benchmark's own feature centroid: the
    budget is buying an estimate of what this application looks like on average,
    so the most typical loops carry the most information about it. 'random' is
    factor_adapt's current behaviour, kept so the difference is measured rather
    than assumed.
    """
    if len(arms) <= n:
        return list(arms)
    if how == "random":
        out = list(arms)
        random.Random(seed).shuffle(out)
        return out[:n]
    z = [[(a["feat"][i] - mean[i]) / sd[i] for i in range(len(mean))]
         for a in arms]
    cen = [stats.fmean(c) for c in zip(*z)]
    order = sorted(range(len(arms)),
                   key=lambda i: (sum((z[i][j] - cen[j]) ** 2
                                      for j in range(len(cen))),
                                  arms[i]["loop"]))
    return [arms[i] for i in order[:n]]


def adapt_score(pick: list, cands: list, mu: dict,
                tau2: float, sigma2: float, shrink: bool) -> dict:
    """
    score(f) = mu(f) + alpha_hat(f).

    alpha_hat is the measured application deviation pooled toward 0 by
        w = tau2 / (tau2 + sigma2 / n_f)
    so a factor measured once barely leaves the prior and one measured a dozen
    times moves most of the way. At tau2 = 0 every weight is 0 and this returns
    the prior exactly — the property that makes it safe.

    shrink=False uses the raw application mean, reported alongside to show what
    the pooling is worth.
    """
    obs: dict = {}
    for a in pick:
        for f in cands:
            if f in a["cen"]:            # only cells the budget actually paid for
                obs.setdefault(f, []).append(a["cen"][f])
    out = dict(mu)
    for f, ys in obs.items():
        a_hat = stats.fmean(ys)
        if shrink:
            w = (tau2 / (tau2 + sigma2 / len(ys))) if tau2 > 0 and sigma2 > 0 \
                else 0.0
            a_hat *= w
        out[f] = mu.get(f, 0.0) + a_hat
    return out


def choose(arm: dict, score: dict) -> int:
    """
    Highest-scoring LEGAL factor, ties to the lower factor.

    A factor missing from `score` was never observed in training. It must NOT
    default to 0.0: most values here are negative, so an unseen factor would
    score above every measured one and win every arm. It defaults to the worst
    known score instead.
    """
    worst = min(score.values()) if score else 0.0
    return min(arm["legal"], key=lambda f: (-score.get(f, worst), f))


# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--deadzone", type=float, default=0.005)
    p.add_argument("--budget", type=int, default=20,
                   help="Compiles spent on the target before shipping. The "
                        "loops-vs-factors trade only exists below n_loops*10, "
                        "since there are only 10 factors to measure.")
    p.add_argument("--splits", type=int, default=5)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--min-cells", type=int, default=4)
    p.add_argument("--min-eval", type=int, default=1)
    args = p.parse_args()

    arms = load_arms(args.run_dir, args.deadzone, args.min_cells)
    if not arms:
        sys.exit("no labelled transform loop has headroom — check --deadzone")
    benches = sorted({a["bench"] for a in arms})
    per_b = [sum(1 for a in arms if a["bench"] == b) for b in benches]
    nfe = len(arms[0]["feat"])

    print(f"{len(arms)} loops with headroom over {len(benches)} benchmarks")
    print(f"loops per benchmark: median {stats.median(per_b):.0f}, "
          f"max {max(per_b)}, "
          f"{sum(1 for n in per_b if n >= 2)} have >=2, "
          f"{sum(1 for n in per_b if n >= 4)} have >=4")
    print(f"budget {args.budget} compiles per target, {args.splits} split(s)\n")

    # n_cand is capped at 10 because only 10 factors exist; a config asking for
    # more is the same experiment at a lower true cost, and would misreport it.
    configs = []
    for n in (1, 2, 3, 4, 6):
        c = min(NF, max(1, args.budget // n))
        if (n, c) not in configs:
            configs.append((n, c))

    taus, sigmas = [], []
    rows: dict = {}
    for sp in range(args.splits):
        bs = list(benches)
        random.Random(args.split_seed + sp).shuffle(bs)
        n_test = max(1, round(args.test_frac * len(bs)))
        te, tr = set(bs[:n_test]), set(bs[n_test:])
        train = [a for a in arms if a["bench"] in tr]
        if not train:
            continue
        mu = global_prior(train)
        tau2, sigma2 = variance_components(train)
        taus.append(tau2)
        sigmas.append(sigma2)
        # Feature statistics from TRAIN only. They are used to z-score before
        # taking a benchmark's centroid, and computing them over the test
        # benchmarks too would let held-out data into the choice of which loops
        # to measure — small, but it is exactly the kind of leak that is
        # indefensible once it is in a paper.
        mean = [stats.fmean(a["feat"][i] for a in train) for i in range(nfe)]
        sd = [max(stats.pstdev([a["feat"][i] for a in train]), 1e-9)
              for i in range(nfe)]

        for n_loops, n_cand in configs:
            cands = sorted(mu, key=lambda f: (-mu[f], f))[:n_cand]
            for how in ("central", "random"):
                num_g = num_a = num_r = den = 0.0
                n_tgt = n_ev = 0
                for b in sorted(te):
                    ba = [a for a in arms if a["bench"] == b]
                    if len(ba) < n_loops + args.min_eval:
                        continue
                    pick = pick_adapt_loops(ba, n_loops, how,
                                            args.split_seed + sp, mean, sd)
                    keys = {(x["loop"], x["u"]) for x in pick}
                    evl = [a for a in ba if (a["loop"], a["u"]) not in keys]
                    if not evl:
                        continue
                    s_a = adapt_score(pick, cands, mu, tau2, sigma2, True)
                    s_r = adapt_score(pick, cands, mu, tau2, sigma2, False)
                    for a in evl:
                        # The global bar is scored on the SAME eval loops as the
                        # adapted policy, inside the same strategy. Comparing
                        # against a bar computed on a different held-out set
                        # would not be a paired comparison at all.
                        num_g += realised(a, choose(a, mu))
                        num_a += realised(a, choose(a, s_a))
                        num_r += realised(a, choose(a, s_r))
                        den += a["orc"]
                    n_tgt += 1
                    n_ev += len(evl)
                if den <= 0:
                    continue
                k = (n_loops, n_cand, how)
                r = rows.setdefault(k, {"g": [], "a": [], "r": [],
                                        "tgt": [], "ev": []})
                r["g"].append(num_g / den)
                r["a"].append(num_a / den)
                r["r"].append(num_r / den)
                r["tgt"].append(n_tgt)
                r["ev"].append(n_ev)

    if not rows:
        sys.exit("no target had enough loops for any configuration — this data "
                 "may have too few transform loops per benchmark")

    t2 = stats.fmean(taus) if taus else 0.0
    s2 = stats.fmean(sigmas) if sigmas else 0.0
    print("CAN ADAPTATION HELP AT ALL?")
    print(f"  between-application variance  tau^2   {t2:.6f}")
    print(f"  within-application variance   sigma^2 {s2:.6f}")
    if t2 > 0 and s2 > 0:
        print("  shrinkage weight              " +
              "  ".join(f"n={n}: {t2 / (t2 + s2 / n):.2f}" for n in (1, 2, 4, 8)))
    else:
        print("  shrinkage weight              0.00 at every n")
    print("  tau^2 is how much applications differ in which factors they like.")
    print("  Near zero means they do not, and adaptation returns the global")
    print("  prior however it is implemented.\n")

    print("SPENDING THE SAME BUDGET DIFFERENTLY")
    print("  capture of the oracle; picks are committed and charged, so compare")
    print("  these to factor_only's `probe` row, not to its top-k rows.\n")
    print(f"  {'loops':>5} {'fac':>4} {'cost':>5} {'tgts':>5} {'eval':>5} | "
          f"{'global':>7} {'adapted':>8} {'gain +- sd across splits':>26} | "
          f"{'unshrunk':>9} {'randsel':>8}")
    best = (None, -float("inf"), 0.0)
    for n_loops, n_cand in configs:
        c = rows.get((n_loops, n_cand, "central"))
        if not c:
            continue
        rnd = rows.get((n_loops, n_cand, "random"))
        g, a = stats.fmean(c["g"]), stats.fmean(c["a"])
        # Per-split gain, not the difference of the two means. A gain smaller
        # than its own spread across splits is not a gain, and the mean alone
        # cannot show that.
        gains = [x - y for x, y in zip(c["a"], c["g"])]
        sd = stats.pstdev(gains) if len(gains) > 1 else 0.0
        rd = (stats.fmean([x - y for x, y in zip(rnd["a"], rnd["g"])])
              if rnd else float("nan"))
        tg = stats.fmean(c["tgt"])
        # A row built from a couple of unusual benchmarks is not the same
        # experiment as the rows above it, and looks identical unless flagged.
        thin = "  <- few targets" if tg < 5 else ""
        print(f"  {n_loops:>5} {n_cand:>4} {n_loops * n_cand:>5} "
              f"{tg:>5.0f} {stats.fmean(c['ev']):>5.0f} | "
              f"{g:6.1%} {a:7.1%} {a - g:+9.1%} +- {sd:5.1%} "
              f"[{min(gains):+.1%}, {max(gains):+.1%}] | "
              f"{stats.fmean(c['r']):8.1%} {rd:+7.1%}{thin}")
        if a - g > best[1]:
            best = ((n_loops, n_cand), a - g, sd)

    print("\n  global    best legal factor by the TRAIN prior, shipped "
          "everywhere. The bar.")
    print("  adapted   shrunk per-application tilt; loops chosen centrally.")
    print("  unshrunk  same, raw application mean — the value of the pooling.")
    print("  gain      adapted minus global on the SAME eval loops.")
    print("  randsel   gain when adaptation loops are chosen at random instead.")

    cfg, gain, sd = best
    print(f"\n  READ: best is {cfg[0]} loop(s) x {cfg[1]} factors, "
          f"{gain:+.1%} +- {sd:.1%} over the\n  global constant, for "
          f"{cfg[0] * cfg[1]} compiles per target.")
    if sd > 0 and gain < sd:
        print(f"  That gain is SMALLER than its own spread across splits. It is "
              f"not yet a\n  result — it is one or two lucky splits. More "
              f"splits would tell you which.")
    elif gain > 0:
        print(f"  The gain clears its own spread, so it is a real effect at "
              f"this budget.\n  Whether it is worth {cfg[0] * cfg[1]} compiles "
              f"per application is a separate\n  question, and the answer "
              f"depends on how often you recompile.")
    else:
        print("  No configuration beats shipping the global constant. Adaptation "
              "does not\n  earn its compiles on this data.")
    print("\n  Rows marked 'few targets' come from a handful of unusually "
          "loop-rich\n  benchmarks, not the population above them. Do not read "
          "them as a trend.")
    if t2 <= 1e-9:
        print("\n  tau^2 is zero: applications do NOT differ in factor "
              "preference here, so\n  no adaptation can beat the global "
              "constant. That is the answer, and it\n  cost no GPU.")


if __name__ == "__main__":
    main()
