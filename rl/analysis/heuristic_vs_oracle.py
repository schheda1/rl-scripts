"""
Score the published heuristic against measured ground truth, per loop.

Joins heuristic_decisions.csv (what it chose) with loop_labels.csv (what was
correct) and reward_cache.json (what its choice was actually worth). Pure table
lookup — no compiles, no GPU.

This is the comparison the original study could not make: it reported
whole-application speedups against -O3 with no notion of what was achievable, so
"fired and gained 4%" and "fired and gained 4% where 30% was available" were
indistinguishable, as were "declined correctly" and "declined and left 20% on the
table".

EVALUATED POPULATION
--------------------
The 446 labelled loops. Every one has a heuristic action — it either fired or
declined — so the comparison over this population is COMPLETE, not a sample.

Decisions on loops outside it are counted and excluded: the eligible population
was fixed at pre-flight, and the toolchain has since begun reporting more
eligible loops (see study_plan.md). Those loops have no measurements, so nothing
can be said about them either way. Note the bias direction: the excluded
decisions are ones the heuristic chose to ACT on, so their exclusion understates
its footprint rather than flattering it.

A STRUCTURAL LIMIT WORTH REPORTING
----------------------------------
--enable-uu-heuristic always unmerges AND unrolls, so the heuristic can express
only two of the three actions: no-op, or unmerge+unroll. It has no way to say
"unroll-only", which the oracle says is correct on ~14% of loops. Those are
unreachable by construction, however well tuned it is — the confusion matrix
keeps that row visible rather than folding it into a generic error rate.

Usage:
  python3 heuristic_vs_oracle.py RUN_DIR --deadzone 0.005
  python3 heuristic_vs_oracle.py RUN_DIR --deadzone 0.005 --csv-out per_bench.csv
"""

import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path

CATEGORIES = ("noop", "unroll_only", "unmerge_unroll")
LABEL = {"noop": "no-op", "unroll_only": "unroll-only",
         "unmerge_unroll": "unmerge+unroll"}


def fnum(s):
    return None if s in ("", None) else float(s)


def pct(x, n):
    return f"{100 * x / n:5.1f}%" if n else "    - "


def load_decisions(path: Path) -> tuple:
    """
    {(benchmark, loop_idx): factor}, plus counts of what was dropped.

    Deduplicates on (benchmark, loop_idx): a small scoping leak means a few
    loops are reported under more than one compile. Identical repeats collapse;
    genuine disagreements about the factor are returned separately so the caller
    can EXCLUDE those loops from scoring. They must not fall through as
    "declined" — the heuristic demonstrably fired, only the factor is unknown,
    and scoring them as declines would report a miss the heuristic never made.
    """
    rows = [r for r in csv.DictReader(open(path)) if r["loop_idx"] != ""]
    seen: dict = {}
    conflicts: set = set()
    for r in rows:
        key = (r["benchmark"], int(r["loop_idx"]))
        fac = int(r["factor"])
        if key in seen and seen[key] != fac:
            conflicts.add(key)
        seen.setdefault(key, fac)
    for k in conflicts:
        seen.pop(k, None)
    return seen, len(rows), conflicts


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--deadzone", type=float, required=True,
                   help="REQUIRED: must match the cache (0.005). A realized "
                        "reward below -deadzone counts as a regression; above "
                        "+deadzone as a gain.")
    p.add_argument("--decisions", type=Path, default=None,
                   help="default: RUN_DIR/heuristic_decisions.csv")
    p.add_argument("--labels", type=Path, default=None,
                   help="default: RUN_DIR/loop_labels.csv")
    p.add_argument("--csv-out", type=Path, default=None)
    args = p.parse_args()

    dec_f = args.decisions or (args.run_dir / "heuristic_decisions.csv")
    lab_f = args.labels or (args.run_dir / "loop_labels.csv")
    rc_f = args.run_dir / "reward_cache.json"
    for f in (dec_f, lab_f, rc_f):
        if not f.exists():
            sys.exit(f"missing {f}")

    fired, n_rows, conflicts = load_decisions(dec_f)
    labels = [r for r in csv.DictReader(open(lab_f)) if r["labelable"] == "1"]
    rewards = json.loads(rc_f.read_text()).get("rewards", {})
    dz = args.deadzone

    in_pop = {(r["benchmark"], int(r["loop_idx"])) for r in labels}
    outside = {k for k in fired if k not in in_pop}

    print("=" * 74)
    print("  COVERAGE")
    print("=" * 74)
    print(f"  decision rows                     : {n_rows}")
    print(f"  distinct (benchmark, loop)        : {len(fired) + len(conflicts)}")
    print(f"  conflicting factors               : {len(conflicts)}"
          f"   <- excluded, NOT counted as declines")
    print(f"  fired INSIDE the population       : {len(fired) - len(outside)}")
    print(f"  fired OUTSIDE it (no ground truth): {len(outside)}"
          f"   <- excluded; understates its footprint")
    print(f"  labelled loops evaluated          : {len(labels)}")

    # --- evaluate every labelled loop -----------------------------------
    rows, unmeasured, skipped_conflict = [], [], 0
    excluded_oracle = 0.0          # headroom removed from BOTH sides of the ratio
    for r in labels:
        key = (r["benchmark"], int(r["loop_idx"]))
        if key in conflicts:
            # It fired; which factor is unknown. Scoring it as a decline would
            # invent a miss.
            skipped_conflict += 1
            excluded_oracle += fnum(r["oracle_reward"]) or 0.0
            continue
        truth = r["category"]
        oracle = fnum(r["oracle_reward"]) or 0.0
        factor = fired.get(key)
        if factor is None:
            pred, realized = "noop", 0.0
        else:
            pred = "unmerge_unroll"
            cell = f"{key[0]}|{key[1]}|1|{factor}"
            if cell not in rewards:
                # It fired on a cell the sweep never measured (a factor outside
                # the trip-count mask, or a gap). Cannot be scored either way.
                unmeasured.append((key, factor))
                excluded_oracle += oracle
                continue
            realized = float(rewards[cell])
        rows.append({"benchmark": key[0], "loop_idx": key[1], "truth": truth,
                     "pred": pred, "factor": factor if factor is not None else "",
                     "oracle": oracle, "realized": realized})

    if skipped_conflict:
        print(f"  labelled loops with a conflict    : {skipped_conflict}"
              f"   <- excluded from scoring")
    if unmeasured:
        print(f"  fired on an UNMEASURED cell       : {len(unmeasured)}"
              f"   <- excluded from scoring")
    print(f"  scored                            : {len(rows)}")
    if excluded_oracle > 0:
        # Exclusions are not random w.r.t. headroom: every one is a loop the
        # heuristic ACTED on. State the mass so the capture ratio is read with
        # the right denominator in mind.
        print(f"  oracle headroom in excluded loops : {excluded_oracle:+.3f}"
              f"   <- absent from BOTH sides of the capture ratio")
    if not rows:
        sys.exit("nothing scoreable")

    # --- confusion matrix ------------------------------------------------
    print("\n" + "=" * 74)
    print("  CATEGORY CONFUSION — rows are ground truth, columns the heuristic")
    print("=" * 74)
    print(f"  {'':<18}{'declined':>10}{'unmerge+unroll':>16}{'':>6}{'accuracy':>10}")
    correct = 0
    for t in CATEGORIES:
        sub = [x for x in rows if x["truth"] == t]
        if not sub:
            continue
        d = sum(1 for x in sub if x["pred"] == "noop")
        u = len(sub) - d
        ok = d if t == "noop" else (u if t == "unmerge_unroll" else 0)
        correct += ok
        note = "  <- UNREACHABLE" if t == "unroll_only" else ""
        print(f"  {LABEL[t]:<18}{d:>10}{u:>16}{'':>6}{pct(ok, len(sub))}{note}")
    print(f"\n  overall accuracy: {pct(correct, len(rows))} ({correct}/{len(rows)})")
    maj = max(CATEGORIES, key=lambda c: sum(1 for x in rows if x["truth"] == c))
    n_maj = sum(1 for x in rows if x["truth"] == maj)
    print(f"  majority-class baseline ('always {LABEL[maj]}'): "
          f"{pct(n_maj, len(rows))}")
    print("  unroll-only is unreachable: the heuristic always unmerges when it "
          "fires,\n  so those loops cannot be got right however well it is tuned.")

    # --- the two error modes ---------------------------------------------
    print("\n" + "=" * 74)
    print("  ERROR MODES")
    print("=" * 74)
    harmful = [x for x in rows if x["truth"] == "noop"
               and x["pred"] != "noop" and x["realized"] < -dz]
    benign = [x for x in rows if x["truth"] == "noop"
              and x["pred"] != "noop" and x["realized"] >= -dz]
    missed = [x for x in rows if x["truth"] != "noop" and x["pred"] == "noop"]
    hit = [x for x in rows if x["truth"] != "noop" and x["pred"] != "noop"]

    print(f"\n  HARMFUL FIRINGS — transformed where declining was optimal")
    print(f"    count {len(harmful)}"
          f"   (a further {len(benign)} fired harmlessly on no-op loops)")
    if harmful:
        c = sorted(x["realized"] for x in harmful)
        print(f"    cost: total {sum(c):+.3f}   median {st.median(c):+.4f}   "
              f"worst {c[0]:+.4f}")
        for x in sorted(harmful, key=lambda z: z["realized"])[:5]:
            print(f"      {x['benchmark']:<30} loop {x['loop_idx']:<4} "
                  f"factor {x['factor']:<3} {x['realized']:+.4f}")

    print(f"\n  MISSES — declined where a transform was worth taking")
    print(f"    count {len(missed)}")
    if missed:
        f_ = sorted((x["oracle"] for x in missed), reverse=True)
        print(f"    forfeited: total {sum(f_):+.3f}   median {st.median(f_):+.4f}"
              f"   largest {f_[0]:+.4f}")
        for x in sorted(missed, key=lambda z: -z["oracle"])[:5]:
            print(f"      {x['benchmark']:<30} loop {x['loop_idx']:<4} "
                  f"{LABEL[x['truth']]:<15} left {x['oracle']:+.4f}")

    print(f"\n  FIRED ON A LOOP THAT DID BENEFIT — how much it captured")
    print(f"    count {len(hit)}")
    if hit:
        o, r_ = sum(x["oracle"] for x in hit), sum(x["realized"] for x in hit)
        print(f"    oracle {o:+.3f} -> realized {r_:+.3f}   "
              f"capture {100 * r_ / o if o else float('nan'):.1f}%")
        worse = [x for x in hit if x["realized"] < -dz]
        if worse:
            print(f"    of these, {len(worse)} ended up SLOWER despite headroom "
                  f"existing:")
            for x in sorted(worse, key=lambda z: z["realized"])[:5]:
                print(f"      {x['benchmark']:<30} loop {x['loop_idx']:<4} "
                      f"oracle {x['oracle']:+.4f} realized {x['realized']:+.4f}")

    # --- bottom line ------------------------------------------------------
    print("\n" + "=" * 74)
    print("  BOTTOM LINE")
    print("=" * 74)
    # label_loops only assigns a transform category when the reward clears the
    # deadzone, so (truth != noop) <=> (oracle > dz). This set is therefore the
    # same one the confusion matrix's non-noop rows cover — if the two ever
    # disagreed, one of them would be wrong.
    head = [x for x in rows if x["oracle"] > dz]
    assert len(head) == sum(1 for x in rows if x["truth"] != "noop"), \
        "headroom set and non-noop categories disagree — label invariant broken"
    o = sum(x["oracle"] for x in head)
    r_ = sum(x["realized"] for x in head)
    print(f"  loops with headroom      : {len(head)}")
    print(f"  oracle available         : {o:+.3f}")
    print(f"  heuristic realized       : {r_:+.3f}")
    print(f"  capture ratio            : "
          f"{100 * r_ / o if o else float('nan'):.1f}%")
    reg = [x for x in rows if x["realized"] < -dz]
    print(f"  loops made SLOWER        : {len(reg)}  {pct(len(reg), len(rows))}")
    if reg:
        print(f"  total cost of those      : "
              f"{sum(x['realized'] for x in reg):+.3f}")
    print(f"  mean realized, all loops : "
          f"{st.mean([x['realized'] for x in rows]):+.4f}"
          f"   (oracle {st.mean([x['oracle'] for x in rows]):+.4f})")

    if args.csv_out:
        by: dict = {}
        for x in rows:
            b = by.setdefault(x["benchmark"], {"benchmark": x["benchmark"],
                                               "loops": 0, "fired": 0,
                                               "harmful": 0, "missed": 0,
                                               "oracle": 0.0, "realized": 0.0})
            b["loops"] += 1
            b["fired"] += int(x["pred"] != "noop")
            b["harmful"] += int(x["truth"] == "noop" and x["pred"] != "noop"
                                and x["realized"] < -dz)
            b["missed"] += int(x["truth"] != "noop" and x["pred"] == "noop")
            b["oracle"] += x["oracle"]
            b["realized"] += x["realized"]
        out = sorted(by.values(), key=lambda e: e["realized"])
        for e in out:
            e["oracle"] = round(e["oracle"], 6)
            e["realized"] = round(e["realized"], 6)
        with open(args.csv_out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
            w.writeheader(); w.writerows(out)
        print(f"\n  per-benchmark: {args.csv_out}")


if __name__ == "__main__":
    main()
