"""
How much within-arm signal is there for the factor head to learn?

The factor decision is an argmax WITHIN a (loop, branch) arm, so it is invariant
to that arm's overall level. Any variance that lives between arms is a nuisance
the head spends capacity on and the decision never consumes. This decomposes the
measured reward into the two parts and reports the one that matters.

    within-arm    the ceiling on what a factor policy can ever be worth
    between-arm   loop level; argmax cannot see it, centering removes it

WHY THE NUMBER YOU ALREADY HAVE IS WRONG
----------------------------------------
factor_response.py:118 computes `best_r - worst_r` over OBSERVED cells, which
still contains compile failures at -0.16/-0.161 and clip-floor cells at -1.0. So
the "+1.06 median spread" that motivated the factor work is mostly "this factor
does not build", not "this factor is slower". Everything here is ran-only.

Read the spread against your measurement noise. The cache holds ONE measurement
per cell and never remeasured, so that noise is unquantified — if the within-arm
spread turns out to be the same size as it, the factor is not learnable by any
model and that is the result.

No torch, no GPU, no toolchain: reads the cache and the eligible list, nothing
else. Loop records are already deduplicated when precheck writes them.

Usage:
  python3 factor_signal.py RUN_DIR
  python3 factor_signal.py RUN_DIR --deadzone 0.005 --csv arms.csv
  python3 factor_signal.py RUN_DIR --keep-clip        # count -1.0 as a measurement
"""

import argparse
import csv
import json
import statistics as stats
import sys
from pathlib import Path

# MIRRORED from offline_data.py / agent.py — importing either pulls in torch and
# this is meant to run anywhere. If they change there, change them here.
FAILURE_VALUES = (-0.16, -0.161)
CLIP_FLOOR = -1.0
IDX_TRIP_KNOWN = 10
IDX_TRIP_COUNT = 11
FACTOR_VALUES = tuple(range(1, 11))


def legal_factors(raw: list, unmerge: int) -> set:
    """
    Factors a policy could actually choose on this arm.

    MIRROR: label_loops.valid_factors — factor 1 is always legal, the rest must
    not exceed a KNOWN trip count. Plus category_factor_mask's extra rule: on the
    unroll branch factor 1 IS the no-op, not a factor choice, so a policy never
    picks it there and its reward must not enter this spread.
    """
    if raw[IDX_TRIP_KNOWN] > 0.5 and int(raw[IDX_TRIP_COUNT]) > 0:
        tc = int(raw[IDX_TRIP_COUNT])
        ok = {f for f in FACTOR_VALUES if f == 1 or f <= tc}
    else:
        ok = set(FACTOR_VALUES)
    return ok - {1} if unmerge == 0 else ok


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--deadzone", type=float, default=0.005,
                   help="Arms whose best-worst spread is under this have no "
                        "factor question to answer (default: 0.005)")
    p.add_argument("--keep-clip", action="store_true",
                   help="Treat -1.0 cells as measurements. They are a mix of "
                        "timeouts and >=100%% slowdowns and the cache cannot "
                        "say which, so they are dropped by default.")
    p.add_argument("--csv", type=Path, default=None,
                   help="Per-arm rows: benchmark, loop, branch, n, spread, mean")
    args = p.parse_args()

    rc_f = args.run_dir / "reward_cache.json"
    if not rc_f.exists():
        sys.exit(f"missing {rc_f}")
    rewards = json.loads(rc_f.read_text()).get("rewards", {})
    if not rewards:
        sys.exit(f"{rc_f} holds no rewards")

    # Trip counts, so illegal cells never enter the spread. Without them a
    # factor no policy can choose still widens the arm.
    elig_f = args.run_dir / "eligible_benchmarks.json"
    raws = {}
    if elig_f.exists():
        for b, recs in json.loads(elig_f.read_text()).get("loop_records", {}).items():
            for r in recs:
                raws[(b, int(r["loop_idx"]))] = r["pre_features_raw"]
    else:
        print(f"WARNING: no {elig_f} — cannot apply the trip-count mask, so "
              f"illegal\n         cells are included and the spread is an "
              f"overestimate.\n")

    n_fail = n_clip = n_illegal = 0
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
            n_fail += 1
            continue
        if v <= CLIP_FLOOR + 1e-9 and not args.keep_clip:
            n_clip += 1
            continue
        raw = raws.get((b, li))
        if raw is not None and f not in legal_factors(raw, u):
            n_illegal += 1
            continue
        arms.setdefault((b, li, u), []).append(v)

    # An arm with one cell has no within-variance and no factor question.
    single = sum(1 for v in arms.values() if len(v) < 2)
    arms = {k: v for k, v in arms.items() if len(v) >= 2}
    if not arms:
        sys.exit("no arm has two or more ran cells — nothing to decompose")

    allv = [x for v in arms.values() for x in v]
    grand = stats.fmean(allv)
    within = sum((x - stats.fmean(v)) ** 2 for v in arms.values() for x in v)
    between = sum(len(v) * (stats.fmean(v) - grand) ** 2 for v in arms.values())
    total = within + between

    spreads = sorted(max(v) - min(v) for v in arms.values())
    def q(p):                      # nearest-rank, no interpolation
        return spreads[min(len(spreads) - 1, int(p * len(spreads)))]
    flat = sum(1 for s in spreads if s <= args.deadzone)

    print(f"cells dropped : {n_fail} failures, {n_clip} at the clip floor, "
          f"{n_illegal} trip-count illegal")
    print(f"arms dropped  : {single} with a single ran cell")
    print(f"kept          : {len(arms)} arms, {len(allv)} ran cells\n")

    print("VARIANCE OF THE MEASURED REWARD")
    print(f"  within-arm   {within / total:6.1%}   <- the ceiling on a factor policy")
    print(f"  between-arm  {between / total:6.1%}   <- loop level; argmax cannot "
          f"see it\n")

    print("BEST-MINUS-WORST WITHIN AN ARM (ran cells only)")
    for name, val in (("p25", q(0.25)), ("median", q(0.50)),
                      ("p75", q(0.75)), ("p90", q(0.90)), ("max", spreads[-1])):
        print(f"  {name:<7} {val:+.4f}")
    print(f"\n  arms where the spread is under the deadzone ({args.deadzone}): "
          f"{flat}/{len(arms)} ({flat / len(arms):.0%})")
    print("  On those the factor genuinely does not matter, whatever a model "
          "predicts.")
    print("\n  Compare the median against your measurement noise. The cache has "
          "one\n  measurement per cell and no replicates, so that noise is "
          "unmeasured — if\n  the two are the same size, the factor is not "
          "learnable and that is the result.")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["benchmark", "loop_idx", "unmerge", "n_ran",
                        "spread", "mean", "best", "worst"])
            for (b, li, u), v in sorted(arms.items()):
                w.writerow([b, li, u, len(v), round(max(v) - min(v), 6),
                            round(stats.fmean(v), 6), round(max(v), 6),
                            round(min(v), 6)])
        print(f"\nper-arm: {args.csv}")


if __name__ == "__main__":
    main()
