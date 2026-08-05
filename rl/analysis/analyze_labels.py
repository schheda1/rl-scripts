"""
Population statistics over the labelled loop table.

Answers the study's framing question directly — "is Unmerge+Unroll universally
useful, and if not, where?" — and reports how much the answer can be trusted.

Reads loop_labels.csv from label_loops.py. Seed-independent: no split is applied
anywhere here, which is the point. Split-conditioned numbers belong in the policy
evaluation, not in a statement about the transform.

Sections
--------
  1. CATEGORY DISTRIBUTION   how often each action is the right one, over all
                             loops and over benchmarks
  2. LABEL CONFIDENCE        how many labels sit within measurement noise of
                             flipping, and what the distribution of margins is.
                             A category split is only as good as this section.
  3. MAGNITUDE               what the wins are actually worth when they exist —
                             a 60% win rate at +0.5% is a different paper from
                             a 60% win rate at +20%
  4. PER-BENCHMARK           which applications benefit, since the whole
                             reframing rests on this varying
  5. ACTION-SPACE COST       compile-failure density, i.e. how hostile the
                             search space is per loop

Usage:
  python3 analyze_labels.py loop_labels.csv
  python3 analyze_labels.py loop_labels.csv --min-fill 0.9 --csv-out per_bench.csv
"""

import argparse
import csv
import statistics as st
import sys
from pathlib import Path

CATEGORIES = ("noop", "unroll_only", "unmerge_unroll")
LABEL = {"noop": "no-op", "unroll_only": "unroll-only",
         "unmerge_unroll": "unmerge+unroll"}


def fnum(s):
    """CSV cell -> float or None (empty means 'not applicable', not zero)."""
    return None if s == "" or s is None else float(s)


def pct(x, n):
    return f"{100 * x / n:5.1f}%" if n else "    - "


def hist(vals, edges, width=34):
    """Text histogram — keeps this dependency-free and paste-able into notes."""
    if not vals:
        return
    buckets = [0] * (len(edges) + 1)
    for v in vals:
        for i, e in enumerate(edges):
            if v <= e:
                buckets[i] += 1
                break
        else:
            buckets[-1] += 1
    top = max(buckets) or 1
    labels = [f"<={e:g}" for e in edges] + [f">{edges[-1]:g}"]
    for lab, b in zip(labels, buckets):
        bar = "#" * int(width * b / top)
        print(f"    {lab:>8} {b:5d} {pct(b, len(vals))} {bar}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("labels", type=Path, help="loop_labels.csv")
    p.add_argument("--min-fill", type=float, default=0.0,
                   help="Restrict to loops with at least this cell coverage. A "
                        "partially-covered loop can be mislabelled toward no-op "
                        "(an unmeasured cell cannot win), so re-running with "
                        "0.95 shows whether the headline survives. (default: 0)")
    p.add_argument("--top", type=int, default=12,
                   help="Benchmarks to list at each end of section 4 (default: 12)")
    p.add_argument("--csv-out", type=Path, default=None,
                   help="Write the per-benchmark table to CSV")
    args = p.parse_args()

    rows = list(csv.DictReader(open(args.labels)))
    if not rows:
        sys.exit(f"{args.labels} is empty")

    unlabelled = [r for r in rows if r["labelable"] != "1"]
    lab = [r for r in rows if r["labelable"] == "1"]
    if args.min_fill > 0:
        before = len(lab)
        lab = [r for r in lab if float(r["fill"]) >= args.min_fill]
        print(f"[--min-fill {args.min_fill}] kept {len(lab)}/{before} labelled loops\n")
    if not lab:
        sys.exit("no labelled loops survive the filter")
    n = len(lab)

    print("=" * 72)
    print(f"  {len(rows)} eligible loops | {n} labelled | "
          f"{len(unlabelled)} unmeasured (excluded)")
    print("=" * 72)

    # --- 1. category distribution -------------------------------------------
    print("\n1. CATEGORY DISTRIBUTION — how often each action is correct\n")
    counts = {c: sum(1 for r in lab if r["category"] == c) for c in CATEGORIES}
    for c in CATEGORIES:
        print(f"    {LABEL[c]:<16} {counts[c]:4d}  {pct(counts[c], n)}")
    transform = n - counts["noop"]
    print(f"\n    any transform    {transform:4d}  {pct(transform, n)}"
          f"   <- the headline")
    # Benchmark-level: a benchmark counts as benefiting if ANY of its loops do.
    by_bench: dict = {}
    for r in lab:
        by_bench.setdefault(r["benchmark"], []).append(r)
    helped = sum(1 for v in by_bench.values()
                 if any(x["category"] != "noop" for x in v))
    print(f"    benchmarks with >=1 benefiting loop: {helped}/{len(by_bench)} "
          f"({100 * helped / len(by_bench):.0f}%)")

    # --- 2. label confidence -------------------------------------------------
    print("\n2. LABEL CONFIDENCE — how many labels are one measurement from flipping\n")
    amb = sum(1 for r in lab if r["ambiguous"] == "1")
    print(f"    flagged ambiguous : {amb}  {pct(amb, n)}")
    margins = [m for m in (fnum(r["category_margin"]) for r in lab) if m is not None]
    if margins:
        print(f"    category margin   : median {st.median(margins):+.4f}  "
              f"mean {st.mean(margins):+.4f}")
        print("    distribution of category margin (blank = runner-up unmeasured "
              "or below deadzone):")
        hist(margins, [0.005, 0.01, 0.02, 0.05, 0.10, 0.20])
    cmar = [m for m in (fnum(r["cell_margin"]) for r in lab) if m is not None]
    if cmar:
        print(f"\n    factor choice within the winning category — median "
              f"cell margin {st.median(cmar):+.4f}")
        print("    (small = the label is right but WHICH factor is arbitrary)")
        hist(cmar, [0.005, 0.01, 0.02, 0.05, 0.10])
    print("\n    Read: a large ambiguous fraction means per-category accuracy "
          "needs\n    that bucket reported separately — those errors cost "
          "nothing in reward.")

    # --- 3. magnitude --------------------------------------------------------
    print("\n3. MAGNITUDE — what the wins are worth\n")
    for c in ("unroll_only", "unmerge_unroll"):
        vals = [fnum(r["oracle_reward"]) for r in lab if r["category"] == c]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        vs = sorted(vals)
        print(f"    {LABEL[c]:<16} n={len(vs):4d}  median {st.median(vs):+.4f}  "
              f"mean {st.mean(vs):+.4f}  p90 {vs[int(0.9 * len(vs)) - 1]:+.4f}  "
              f"max {vs[-1]:+.4f}")
    allv = [fnum(r["oracle_reward"]) or 0.0 for r in lab]
    print(f"\n    oracle mean over ALL labelled loops: {st.mean(allv):+.4f}"
          f"   <- ceiling for a perfect policy")
    print("    distribution of oracle reward on benefiting loops:")
    hist([v for v in allv if v > 0], [0.01, 0.02, 0.05, 0.10, 0.20, 0.50])

    # --- 4. per benchmark ----------------------------------------------------
    print("\n4. PER-BENCHMARK — the variation the whole study rests on\n")
    table = []
    for b, v in by_bench.items():
        nb = len(v)
        ben = sum(1 for x in v if x["category"] != "noop")
        orc = [fnum(x["oracle_reward"]) or 0.0 for x in v]
        table.append({"benchmark": b, "loops": nb, "benefiting": ben,
                      "benefit_rate": round(ben / nb, 4),
                      "oracle_mean": round(st.mean(orc), 6),
                      "oracle_max": round(max(orc), 6),
                      "unmerge_best": sum(1 for x in v
                                          if x["category"] == "unmerge_unroll")})
    table.sort(key=lambda t: -t["oracle_mean"])
    hdr = f"    {'benchmark':<32}{'loops':>6}{'benefit':>9}{'oracle_mean':>13}{'max':>9}"
    print(hdr)
    for t in table[:args.top]:
        print(f"    {t['benchmark']:<32}{t['loops']:>6}"
              f"{t['benefit_rate'] * 100:>8.0f}%{t['oracle_mean']:>13.4f}"
              f"{t['oracle_max']:>9.4f}")
    print(f"    {'...':<32}")
    for t in table[-args.top:]:
        print(f"    {t['benchmark']:<32}{t['loops']:>6}"
              f"{t['benefit_rate'] * 100:>8.0f}%{t['oracle_mean']:>13.4f}"
              f"{t['oracle_max']:>9.4f}")
    rates = [t["benefit_rate"] for t in table]
    zero = sum(1 for t in table if t["benefit_rate"] == 0)
    print(f"\n    benefit rate across benchmarks: median "
          f"{st.median(rates) * 100:.0f}%, "
          f"sd {st.stdev(rates) * 100:.0f}pp" if len(rates) > 1 else "")
    print(f"    benchmarks where NO loop benefits: {zero}/{len(table)} "
          f"({100 * zero / len(table):.0f}%)")

    # --- 5. action-space cost ------------------------------------------------
    print("\n5. ACTION-SPACE COST — how hostile the search space is\n")
    meas = sum(int(r["n_measured"]) for r in lab)
    fail = sum(int(r["n_failed"]) for r in lab)
    wins = sum(int(r["n_wins"]) for r in lab)
    print(f"    measured cells        : {meas}")
    print(f"    compile failures      : {fail}  {pct(fail, meas)}")
    print(f"    cells that beat no-op : {wins}  {pct(wins, meas)}")
    print("\n    Read: a brute-force search pays for every cell; these are the "
          "odds\n    it faces per probe, and the baseline any learned search "
          "must beat.")

    if args.csv_out:
        with open(args.csv_out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(table[0].keys()))
            w.writeheader()
            w.writerows(table)
        print(f"\nPer-benchmark table: {args.csv_out}")


if __name__ == "__main__":
    main()
