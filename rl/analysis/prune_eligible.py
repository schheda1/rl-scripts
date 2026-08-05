"""
Drop benchmarks with no measurable baseline from eligible_benchmarks.json.

WHY A SCRIPT
------------
This file drives split_benchmarks, so a hand edit that misses one name — or
leaves a stale loop_counts entry — silently changes every seed's train/val/test
assignment, and nothing downstream notices.

The exclusion is DERIVED, never typed: a benchmark is pruned iff it has no entry
in baseline_cache.json.  That is exactly the condition train._worker_fn and
collect_cells already apply at runtime ("No baseline for X — skipping N loops"),
so this only makes the file agree with what the pipeline was doing anyway.  No
human decision enters, which is what keeps the exclusion defensible.

KEYS THIS MUST NOT TOUCH
------------------------
  normalizer           reward_cache.json's post_features are NORMALISED vectors,
                       validated against an md5 of this state dict.  Refit or
                       drop it and all of them are silently discarded on load.
  features_version     A mismatch makes precheck_benchmarks discard the cache
  eligibility_version  and re-run the ~40-minute LoopCount pass — which rewrites
                       this file with a REFIT normalizer, i.e. the same damage.

They are copied through untouched and verified byte-identical after writing.

Usage:
  python3 prune_eligible.py RUN_DIR --dry-run
  python3 prune_eligible.py RUN_DIR
  python3 prune_eligible.py RUN_DIR --names foo-cuda bar-cuda   # explicit override
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

PRESERVE = ("normalizer", "features_version", "eligibility_version")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--names", nargs="+", default=None,
                   help="Explicit benchmark names to drop, instead of deriving "
                        "them from the missing baselines.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    elig_f = args.run_dir / "eligible_benchmarks.json"
    bl_f = args.run_dir / "baseline_cache.json"
    if not elig_f.exists():
        sys.exit(f"missing {elig_f}")
    data = json.loads(elig_f.read_text())
    eligible = list(data.get("eligible", []))
    records = data.get("loop_records", {})
    counts = data.get("loop_counts", {})

    if args.names is not None:
        drop = sorted(set(args.names) & set(eligible))
        unknown = sorted(set(args.names) - set(eligible))
        if unknown:
            print(f"warning: not in eligible, ignored: {unknown}")
    else:
        if not bl_f.exists():
            sys.exit(f"missing {bl_f} — pass --names to prune explicitly")
        # The flat name->{...} map lives under "baselines", not at the top level.
        baselines = json.loads(bl_f.read_text()).get("baselines", {})
        if not baselines:
            sys.exit(f"{bl_f} has no 'baselines' key — refusing to prune "
                     f"everything")
        drop = sorted(n for n in eligible if n not in baselines)

    kept = [n for n in eligible if n not in set(drop)]
    dropped_loops = sum(len(records.get(n, [])) for n in drop)
    kept_loops = sum(len(records.get(n, [])) for n in kept)

    # Cross-check against the reward cache.  A benchmark with no baseline can
    # never have been measured, so measured cells for one means it was simply
    # never ATTEMPTED — e.g. a run whose splits did not cover it — and pruning
    # it would silently delete a perfectly good benchmark.  This is the one
    # failure mode the derived rule cannot see on its own.
    rc_f = args.run_dir / "reward_cache.json"
    if rc_f.exists() and drop:
        try:
            rewards = json.loads(rc_f.read_text()).get("rewards", {})
        except Exception:
            rewards = {}
        with_cells: dict = {}
        for k in rewards:
            name = k.split("|")[0]
            if name in set(drop):
                with_cells[name] = with_cells.get(name, 0) + 1
        if with_cells:
            print("\nREFUSING TO PRUNE — these have no baseline but DO have "
                  "measured cells, so they were never attempted rather than "
                  "unmeasurable:")
            for n, c in sorted(with_cells.items()):
                print(f"    {n}: {c} cells")
            sys.exit("Re-measure their baselines, or pass --names to override.")

    print(f"eligible benchmarks : {len(eligible)} -> {len(kept)}  "
          f"({len(drop)} dropped)")
    print(f"eligible loops      : {kept_loops + dropped_loops} -> {kept_loops}  "
          f"({dropped_loops} dropped)")
    print(f"dropping: {drop}")
    if not drop:
        print("\nNothing to prune — file already agrees with the baselines.")
        return
    if args.dry_run:
        print("\nDry run — nothing written.")
        return

    backup = elig_f.with_suffix(f".json.bak-{datetime.now():%Y%m%d-%H%M%S}")
    shutil.copy2(elig_f, backup)

    data["eligible"] = kept
    data["loop_records"] = {k: v for k, v in records.items() if k in set(kept)}
    if counts:
        data["loop_counts"] = {k: v for k, v in counts.items() if k in set(kept)}
    # Record WHY they went, in the file's own bookkeeping.  precheck never reads
    # "excluded", so this is safe, and it keeps the exclusion auditable from the
    # artifact rather than only from a log.
    already = {e.get("name") for e in data.get("excluded", [])
               if isinstance(e, dict)}
    data.setdefault("excluded", []).extend(
        {"name": n, "reason": "no measurable baseline (pruned)"}
        for n in drop if n not in already)
    # indent=2 matches how train.py writes this file — a compact rewrite would
    # collapse it to one line and make future inspection and diffs unreadable.
    # Atomic: a kill mid-write must not leave a truncated eligibility file, or
    # the next precheck silently re-runs and refits the normalizer.
    tmp = elig_f.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(elig_f)

    # Verify the untouchable keys survived exactly.
    before = json.loads(backup.read_text())
    after = json.loads(elig_f.read_text())
    for k in PRESERVE:
        if json.dumps(before.get(k), sort_keys=True) != \
                json.dumps(after.get(k), sort_keys=True):
            sys.exit(f"ABORT: '{k}' changed — restore from {backup}")
    orphans = set(after["loop_records"]) - set(after["eligible"])
    if orphans:
        # Impossible by construction, but an orphan would inflate the audit's
        # "eligible loops" total, so fail loudly rather than trust the comment.
        sys.exit(f"ABORT: loop_records has {len(orphans)} entries not in "
                 f"eligible — restore from {backup}")
    no_records = set(after["eligible"]) - set(after["loop_records"])
    if no_records:
        print(f"note: {len(no_records)} kept benchmark(s) have no loop_records "
              f"entry (harmless — they contribute no loops): {sorted(no_records)}")

    print(f"\nBackup : {backup}")
    print(f"Written: {elig_f}")
    print(f"normalizer / features_version / eligibility_version verified unchanged")
    print("NOTE: every seed's train/val/test assignment has now shifted — "
          "re-run the audit before using any earlier split numbers.")


if __name__ == "__main__":
    main()
