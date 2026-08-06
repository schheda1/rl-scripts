"""
Offline data plane for the UU study — features, rewards, folds and scoring.

Everything here is a table lookup. No compiles, no GPU, no HeCBench source tree,
no toolchain: the three cached artifacts hold the whole experiment.

    eligible_benchmarks.json   loop_records[bench] -> pre_features_raw (93-dim,
                               RAW), filename, triple, loop_idx; plus the fitted
                               normalizer.
    reward_cache.json          rewards["bench|loop|unmerge|factor"] -> float,
                               and post_features["bench|loop"] (ALREADY
                               NORMALISED — see make_state in adapt_eval.py).
    loop_labels.csv            ground-truth category per loop, from label_loops.py.

WHY THIS MODULE EXISTS SEPARATELY
---------------------------------
The live pipeline is not modified. This mirrors the parts of it that decide what
an action *is* and what it *earns*, so the offline runs stay comparable to the
online ones — but it reads those decisions from cache instead of measuring them.
Every mirrored rule is marked MIRROR with its source, because a silent drift here
would make offline numbers look valid while describing a different experiment.

EVALUATION — TWO LEVELS, ALWAYS REPORTED TOGETHER
-------------------------------------------------
  1. DECISION QUALITY   per-category accuracy against loop_labels.csv: did the
                        policy pick the action the oracle says is correct?
  2. PERFORMANCE        what the chosen cell actually earned — capture ratio,
                        regressions, realized vs oracle reward.

Accuracy alone misleads in both directions on this population: "always
unmerge+unroll" scores the best accuracy of any fixed rule and is reward-
catastrophic, while "always no-op" scores poorly and beats the published
heuristic outright. Neither number is reported without the other.
"""

import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from adapt_eval import NOOP, build_tables                      # noqa: E402
from train import _dedup_loop_records                          # noqa: E402

# IMPORTED, never re-derived. Every bug found in review here came from
# re-implementing one of label_loops' rules instead of reading it: the deadzone
# gate on the oracle, and the trip-count restriction on which cells exist at all.
# label_loops is the definition of ground truth, so anything that has to agree
# with it comes from it.
from label_loops import CATEGORIES, valid_factors               # noqa: E402

LABEL = {"noop": "no-op", "unroll_only": "unroll-only",
         "unmerge_unroll": "unmerge+unroll"}
assert set(LABEL) == set(CATEGORIES), "LABEL and label_loops.CATEGORIES differ"

# MIRROR: agent.py:_IDX_TRIP_COUNT_KNOWN / _IDX_TRIP_COUNT (private, hence the
# copy). Copies drift, so load_run() asserts these positions against
# FEATURE_COLUMNS rather than trusting them: if the schema ever gains a column
# at the front, the mask silently starts reading a different feature and every
# factor decision is made against a fabricated trip count.
IDX_TRIP_KNOWN = 10
IDX_TRIP_COUNT = 11


def category_of(action) -> str:
    """
    MIRROR: the three-action space. (0,1) declines, (0,f>1) unrolls only,
    (1,f) unmerges and unrolls. Any factor with unmerge=1 is the unmerge
    category — including factor 1, which still pays for path duplication.
    """
    unmerge, factor = action
    if unmerge == 0:
        return "noop" if factor == 1 else "unroll_only"
    return "unmerge_unroll"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_run(run_dir: Path, deadzone: float,
             labels_file: "Path | None" = None) -> dict:
    """
    Everything the offline experiment needs, from cache alone.

    Returns keys:
      loops       list of assignment dicts (MIRROR: train.build_loop_assignments,
                  minus benchmark_path — nothing offline opens the source tree)
      tables      {(bench, loop_idx): {(unmerge, factor): reward}}, no-op at 0.0
      normalizer  the FITTED normalizer from the cache, used as-is
      postf       post-unmerge feature vectors, normalised
      labels      {(bench, loop_idx): category} from loop_labels.csv
      benchmarks  sorted benchmark names that have at least one loop
    """
    elig_f = run_dir / "eligible_benchmarks.json"
    rc_f = run_dir / "reward_cache.json"
    lab_f = labels_file or (run_dir / "loop_labels.csv")
    for f in (elig_f, rc_f, lab_f):
        if not f.exists():
            raise SystemExit(f"missing {f}")

    elig = json.loads(elig_f.read_text())
    records = elig.get("loop_records", {})
    if not records:
        raise SystemExit(f"{elig_f} has no loop_records")

    # MIRROR: train.precheck_benchmarks applies dedup at LOAD time, so a run
    # trained on the deduped set. Skipping it here would train offline on loops
    # the online pipeline never saw, and the two would stop being comparable.
    records, dropped = _dedup_loop_records(records)

    # Feature-space guard. The trip-count mask reads fixed indices out of the
    # raw vector; if the schema ever grows a column at the front, the mask
    # silently starts reading the wrong feature and every factor decision is
    # made against a fabricated trip count.
    from hecbench import FEATURE_COLUMNS
    assert FEATURE_COLUMNS[IDX_TRIP_KNOWN] == "tripCountKnown", \
        f"feature layout moved: index {IDX_TRIP_KNOWN} is not tripCountKnown"
    assert FEATURE_COLUMNS[IDX_TRIP_COUNT] == "tripCount", \
        f"feature layout moved: index {IDX_TRIP_COUNT} is not tripCount"

    loops = []
    for bench in sorted(records):
        for rec in records[bench]:
            loops.append({
                "benchmark_name":   bench,
                "loop_idx":         int(rec["loop_idx"]),
                "filename":         rec.get("filename", ""),
                "triple":           rec.get("triple", ""),
                "pre_features_raw": rec["pre_features_raw"],
            })

    rc = json.loads(rc_f.read_text())
    tables = build_tables(loops, rc.get("rewards", {}))
    postf = rc.get("post_features", {})

    # Restrict every table to the TRIP-COUNT-VALID cells, exactly as
    # label_loops.py:87-89 does before it computes anything. build_tables takes
    # the cache wholesale, so without this a table can hold a cell whose factor
    # exceeds a known trip count — one the oracle would happily pick and that
    # build_factor_mask makes it impossible for any policy to select. That
    # inflates the ceiling with unreachable reward and makes the derived oracle
    # disagree with the stored label.
    n_dropped_invalid = 0
    by_key = {(l["benchmark_name"], l["loop_idx"]): l for l in loops}
    for key, table in tables.items():
        allowed = {(u, f) for u in (0, 1)
                   for f in valid_factors(by_key[key]["pre_features_raw"])}
        kept = {a: r for a, r in table.items() if a in allowed}
        n_dropped_invalid += len(table) - len(kept)
        # The free no-op is injected by build_tables and is always legal
        # (factor 1 passes every mask), but re-assert rather than assume.
        kept[NOOP] = 0.0
        tables[key] = kept

    labels, oracle_mismatch, fill = {}, [], {}
    with open(lab_f) as fh:
        for r in csv.DictReader(fh):
            if r.get("labelable") != "1":
                continue
            key = (r["benchmark"], int(r["loop_idx"]))
            cat = r["category"]
            # Fail fast rather than KeyError deep inside a sweep's confusion
            # matrix. A category outside the three-action space means the labels
            # were produced by a different label_loops than this scorer expects.
            if cat not in CATEGORIES:
                raise SystemExit(
                    f"{lab_f}: loop {key} has category {cat!r}, which is not one "
                    f"of {CATEGORIES}. Labels and scorer disagree — regenerate "
                    f"loop_labels.csv against this cache.")
            labels[key] = cat
            # Measured-cell fraction, straight from label_loops. Consumers that
            # fine-tune on a loop need it: a half-measured loop is a half-known
            # table, not a known one.
            try:
                fill[key] = float(r.get("fill", 1.0) or 1.0)
            except ValueError:
                fill[key] = 1.0
            # Cross-check: label_loops' stored oracle and the oracle derived
            # from this cache must agree, or accuracy would describe one table
            # while performance describes another — invisible in the output.
            #
            # It must be compared DEADZONE-GATED. label_loops stores
            # scores[category] (label_loops.py:185), and scores sets every
            # transform category to -inf when it fails to clear the deadzone
            # (:165-167) — so a loop whose best transform is +0.001 against a
            # 0.005 deadzone is labelled no-op and stores 0.0, not 0.001.
            # Comparing against the raw cell maximum flags every such loop as a
            # cache mismatch that is not one.
            #
            # Gating the global max is equivalent to label_loops' per-category
            # gating: if the best cell clears the deadzone it is also its own
            # category's best, and if it does not, every category is gated away
            # and no-op's 0.0 wins either way.
            stored = r.get("oracle_reward", "")
            if stored not in ("", None) and key in tables:
                _, derived = oracle_of_gated(tables[key], deadzone)
                # Stored values are round(x, 6); 2e-6 clears that comfortably
                # while staying far below any real cache disagreement.
                if abs(float(stored) - derived) > 2e-6:
                    oracle_mismatch.append((key, float(stored), derived))
    if oracle_mismatch:
        head = "; ".join(f"{k}: labels {a:+.4f} vs cache {b:+.4f}"
                         for k, a, b in oracle_mismatch[:5])
        raise SystemExit(
            f"{len(oracle_mismatch)} loop(s) where loop_labels.csv disagrees "
            f"with reward_cache.json about the deadzone-gated oracle.\n"
            f"  Either the labels were built from a different cache, or they "
            f"were built with a different --deadzone than the {deadzone} passed "
            f"here. Regenerate loop_labels.csv against this cache, at this "
            f"deadzone.\n  Examples: {head}")

    from hecbench import FeatureNormalizer
    normalizer = FeatureNormalizer.from_state_dict(elig.get("normalizer", {}))

    # label_loops.py does NOT dedup — it walks loop_records directly — while
    # this loader does, mirroring train.precheck_benchmarks. In practice the
    # stored cache was already deduped when precheck wrote it, so both see the
    # same set and this is 0. If it ever is not, the scored population silently
    # stops matching the published label counts and every denominator moves.
    n_matched = sum(1 for l in loops
                    if (l["benchmark_name"], l["loop_idx"]) in labels)

    present = {l["benchmark_name"] for l in loops}
    return {
        "n_label_rows": len(labels),
        "n_labelled_loops": n_matched,
        # Raw cache, for train.build_warm_start_entries — it wants the flat
        # "bench|loop|u|f" map, not the per-loop tables.
        "rewards": rc.get("rewards", {}),
        "loops": loops,
        "tables": tables,
        "normalizer": normalizer,
        "postf": postf,
        "labels": labels,
        "fill": fill,
        "benchmarks": sorted(present),
        # UNSORTED, as stored. train.split_benchmarks shuffles whatever list it
        # is handed, so the permutation — and therefore which benchmarks landed
        # in test — depends on the INPUT ORDER, not just the seed. precheck
        # wrote this list in the order it processed them, so it is the only
        # order that reproduces a training run's split. Sorting here would
        # silently produce a different, plausible-looking split.
        "eligible_order": [b for b in elig.get("eligible", []) if b in present],
        "n_dropped_dedup": dropped,
        "n_dropped_invalid": n_dropped_invalid,
        "normalizer_fitted": normalizer._fitted,
    }


def oracle_of_gated(table: dict, deadzone: float) -> tuple:
    """
    (best_action, best_reward) under LABEL_LOOPS' rule, not the raw cell maximum.

    MIRROR: label_loops.py:165-167 — a transform must CLEAR the deadzone to beat
    declining; below it, no-op wins and the oracle is exactly 0.0. Chasing a
    +0.001 cell when the deadzone says that is noise is not optimal play, and
    declining costs nothing.

    This must be the single definition used by both the oracle reference policy
    and the label cross-check. When they disagreed, the "oracle" scored 80% against
    the very labels it is supposed to define — an incoherent ceiling.

    Ties between categories go to the SIMPLER one, as label_loops' max() over the
    ordered CATEGORIES tuple does. Exact ties are vanishingly unlikely in measured
    data, but a rule that is only right by luck is not a rule.
    """
    best_a, best_r = NOOP, 0.0
    for a, r in table.items():
        if a == NOOP or r <= deadzone:
            continue
        if r > best_r or (r == best_r and CATEGORIES.index(category_of(a))
                          < CATEGORIES.index(category_of(best_a))):
            best_a, best_r = a, r
    return best_a, best_r


def labelled_loops(data: dict) -> list:
    """
    Loops that carry a ground-truth category.

    label_loops.py marks a loop unlabelable when it has no measured cells, so
    these are exactly the loops on which BOTH accuracy and performance can be
    computed. Scoring the rest would mix "the policy was wrong" with "we do not
    know what was right".
    """
    return [l for l in data["loops"]
            if (l["benchmark_name"], l["loop_idx"]) in data["labels"]]


# ---------------------------------------------------------------------------
# Grouped K-fold
# ---------------------------------------------------------------------------

def grouped_kfold(benchmarks: list, k: int, seed: int) -> list:
    """
    [(train_benchmarks, test_benchmarks)] — grouped by BENCHMARK, never by loop.

    A benchmark's loops share kernels, source, and often the reward itself, so
    splitting one across folds leaks the answer. Grouping also matches how the
    result is used: the deployment question is "a new application", not "another
    loop in an application already measured".

    Every benchmark is in the test fold exactly once, so the union of held-out
    predictions covers the whole population, each loop predicted by a model that
    never saw it.
    """
    if k < 2:
        raise ValueError("k must be >= 2")
    shuffled = list(benchmarks)
    random.Random(seed).shuffle(shuffled)
    folds = [shuffled[i::k] for i in range(k)]
    out = []
    for i in range(k):
        test = folds[i]
        train = [b for j, f in enumerate(folds) if j != i for b in f]
        out.append((train, test))
    return out


def holdout_split(benchmarks: list, frac: float, seed: int) -> tuple:
    """(fit, holdout) inside a training fold — for early stopping only."""
    shuffled = list(benchmarks)
    random.Random(seed).shuffle(shuffled)
    n_hold = max(1, round(len(shuffled) * frac))
    return shuffled[n_hold:], shuffled[:n_hold]


def loops_for(loops: list, benchmarks) -> list:
    names = set(benchmarks)
    return [l for l in loops if l["benchmark_name"] in names]


# ---------------------------------------------------------------------------
# Reference policies
# ---------------------------------------------------------------------------
# Split into two groups that are NOT peers, and must never be tabulated as if
# they were:
#
#   DEPLOYABLE      a function of features alone — can be applied to an
#                   application nobody has measured. Learned policy,
#                   marginal-best, always-no-op.
#   REFERENCE       requires measuring the target first, so it cannot be
#                   shipped. Oracle, benchmark-dominant. Useful as a ceiling;
#                   beating a policy does NOT make it a better alternative.

def always_noop_picks(loops: list) -> list:
    return [(l["benchmark_name"], l["loop_idx"], NOOP) for l in loops]


def marginal_ranking(tables: dict, train_keys: list) -> list:
    """
    [(action, mean_reward, n_loops)], best mean first — the display companion to
    marginal_picks.

    Worth printing, because this baseline degenerating to the no-op is a result,
    not a bug: no-op is exactly 0.0 on every loop, so if it ranks first then
    EVERY transform arm has a negative mean over the population. Without seeing
    the means, "marginal-best is identical to always-no-op" just looks like a
    duplicated row.

    The ORDER comes from adapt_eval.marginal_policy, so the ranking shown and
    the ranking used cannot disagree; only the means are computed here.
    """
    from adapt_eval import marginal_policy
    tot: dict = {}
    cnt: dict = {}
    for k in train_keys:
        for a, r in tables[k].items():
            tot[a] = tot.get(a, 0.0) + r
            cnt[a] = cnt.get(a, 0) + 1
    return [(a, tot[a] / cnt[a], cnt[a]) for a in marginal_policy(tables, train_keys)]


def marginal_picks(loops: list, tables: dict, train_keys: list) -> list:
    """
    DEPLOYABLE. The single action with the best mean reward on TRAIN, applied to
    every loop regardless of features. A feature-conditioned policy that only
    matches this has learned a global prior and nothing about the loop.

    Falls back per loop to the best action that EXISTS in that loop's table:
    the global winner may be masked out by trip count or simply unmeasured, and
    scoring it as unmeasured would flatter the learned policy by comparison.
    """
    from adapt_eval import marginal_policy
    ranked = marginal_policy(tables, train_keys)
    picks = []
    for l in loops:
        key = (l["benchmark_name"], l["loop_idx"])
        table = tables[key]
        chosen = next((a for a in ranked if a in table), NOOP)
        picks.append((key[0], key[1], chosen))
    return picks


def oracle_picks(loops: list, tables: dict, deadzone: float) -> list:
    """
    REFERENCE — requires the target to be measured. The ceiling, not a rival.

    Gated, so it agrees with the labels by construction: this policy must score
    100% accuracy, and a drop below that means the ceiling and the ground truth
    have come apart.
    """
    picks = []
    for l in loops:
        key = (l["benchmark_name"], l["loop_idx"])
        best, _ = oracle_of_gated(tables[key], deadzone)
        picks.append((key[0], key[1], best))
    return picks


def benchmark_dominant_picks(loops: list, tables: dict, labels: dict) -> list:
    """
    REFERENCE — requires measuring the target application. Predicts each
    benchmark's most common ground-truth category for every loop in it, using
    the best available cell of that category.

    Included because benefit is bimodal per benchmark (median benefit rate 100%,
    26% of benchmarks at exactly 0%): if this matches the learned policy, the
    policy has learned the application and not the loop.
    """
    by_bench: dict = {}
    for l in loops:
        key = (l["benchmark_name"], l["loop_idx"])
        cat = labels.get(key)
        if cat:
            by_bench.setdefault(key[0], []).append(cat)
    # max() over a set() would depend on set iteration order, which for strings
    # varies with hash randomisation between processes — the same run would give
    # different baselines on different invocations. Ranking over the fixed
    # CATEGORIES tuple is reproducible, and ties go to the SIMPLER action, which
    # is label_loops' own tie rule.
    dominant = {b: max(CATEGORIES,
                       key=lambda c, cs=cs: (cs.count(c), -CATEGORIES.index(c)))
                for b, cs in by_bench.items()}
    picks = []
    for l in loops:
        key = (l["benchmark_name"], l["loop_idx"])
        table = tables[key]
        want = dominant.get(key[0], "noop")
        cands = [(r, a) for a, r in table.items() if category_of(a) == want]
        chosen = max(cands)[1] if cands else NOOP
        picks.append((key[0], key[1], chosen))
    return picks


# ---------------------------------------------------------------------------
# Scoring — decision quality AND performance
# ---------------------------------------------------------------------------

def score_decisions(picks: list, tables: dict, labels: dict,
                    deadzone: float, missing_reward: "float | None" = None) -> dict:
    """
    Both evaluation levels over one set of (bench, loop_idx, action) picks.

    A cell with no row in the cache always counts against ACCURACY — picking it
    is a real decision. What it earns is `missing_reward`:

      float (default in the runner: the failure penalty)
            After exhaustive collection an absent row means the cell FAILED, so
            charging it is the honest reading. It also removes a bias: the
            oracle, always-no-op and marginal-best all pick from cells that
            exist by construction, so excluding absences would discount only the
            learned policy's bad choices.
      None  Exclude from performance and report the count separately. Use to
            measure how much of a result rides on those cells.
    """
    conf = {t: {p: 0 for p in CATEGORIES} for t in CATEGORIES}
    realized_sum = oracle_sum = 0.0
    n = n_scored = n_unmeasured = n_regress = n_headroom = 0
    realized_all: list = []
    per_bench: dict = {}

    for bench, li, action in picks:
        key = (bench, li)
        truth = labels.get(key)
        if truth is None:
            continue
        n += 1
        conf[truth][category_of(action)] += 1

        table = tables[key]
        # Gated, so oracle_sum is exactly the sum of label_loops' stored
        # oracle_reward. (Behaviourally identical to the raw maximum here —
        # gating only zeroes values that already fail `orc > deadzone` below —
        # but going through one definition is what stops the two from drifting.)
        _, orc = oracle_of_gated(table, deadzone)
        if action not in table:
            n_unmeasured += 1
            if missing_reward is None:
                continue
            r = float(missing_reward)
        else:
            r = table[action]
        n_scored += 1
        realized_all.append(r)
        if r < -deadzone:
            n_regress += 1
        b = per_bench.setdefault(bench, {"realized": 0.0, "oracle": 0.0,
                                         "loops": 0, "regress": 0})
        b["loops"] += 1
        b["regress"] += int(r < -deadzone)
        # MIRROR: adapt_eval.score — only loops with headroom enter the capture
        # ratio. A loop whose oracle is 0 has nothing to capture, and including
        # it leaves the denominator reward-free while the numerator can still go
        # negative.
        if orc > deadzone:
            n_headroom += 1
            realized_sum += r
            oracle_sum += orc
            b["realized"] += r
            b["oracle"] += orc

    correct = sum(conf[t][t] for t in CATEGORIES)
    per_cat = {}
    for t in CATEGORIES:
        tot = sum(conf[t].values())
        per_cat[t] = {"n": tot, "correct": conf[t][t],
                      "acc": conf[t][t] / tot if tot else float("nan")}

    return {
        "loops": n,
        "loops_scored": n_scored,
        "loops_unmeasured": n_unmeasured,
        "loops_with_headroom": n_headroom,
        "accuracy": correct / n if n else float("nan"),
        "per_category": per_cat,
        "confusion": conf,
        "capture": realized_sum / oracle_sum if oracle_sum > 0 else float("nan"),
        "oracle_sum": oracle_sum,
        "realized_sum": realized_sum,
        # Mean over EVERY scored loop, not only those with headroom: this is the
        # number that must be compared against always-no-op's exact 0.0, and
        # restricting it to headroom loops would hide the cost of firing on
        # loops that had nothing to gain.
        "mean_realized": (sum(realized_all) / len(realized_all)
                          if realized_all else 0.0),
        "regression_rate": n_regress / n_scored if n_scored else 0.0,
        "n_regress": n_regress,
        "n_benchmarks": len(per_bench),
        "per_bench": per_bench,
    }


_HDR = (f"  {'':<26}{'all':>7}{'no-op':>8}{'unroll':>8}{'unmerge':>9}"
        f"  |{'capture':>9}{'mean':>9}{'slower':>8}{'unmeas':>8}")
_RULE = "  " + "-" * (len(_HDR) - 2)


def table_header() -> str:
    """
    One table instead of a block per policy. These numbers only mean anything
    against each other — 'capture 88.7%' says nothing until you can see the
    oracle's 100% and always-no-op's 0% on the lines above and below it.
    """
    return (f"  {'':<26}{'-- ACCURACY ------------------':^32}"
            f"  |{'-- PERFORMANCE ----------------':^34}\n"
            + _HDR + "\n" + _RULE)


def table_row(name: str, m: dict) -> str:
    def _a(t):
        c = m["per_category"][t]
        return "     -" if c["n"] == 0 else f"{100 * c['acc']:6.1f}%"
    return (f"  {name:<26}{100 * m['accuracy']:6.1f}%"
            f"{_a('noop'):>8}{_a('unroll_only'):>8}{_a('unmerge_unroll'):>9}"
            f"  |{100 * m['capture']:8.1f}%{m['mean_realized']:+9.4f}"
            f"{m['n_regress']:>8}{m['loops_unmeasured']:>8}")


def format_report(name: str, m: dict) -> str:
    """Verbose single-policy block. Kept for the checkpoint mode, where there is
    only one policy and the counts matter more than the comparison."""
    lines = [f"  {name}"]
    lines.append(f"    loops {m['loops']:<5} scored {m['loops_scored']:<5} "
                 f"unmeasured {m['loops_unmeasured']}")
    lines.append(f"    accuracy        {100 * m['accuracy']:5.1f}%")
    for t in CATEGORIES:
        c = m["per_category"][t]
        acc = "    - " if c["n"] == 0 else f"{100 * c['acc']:5.1f}%"
        lines.append(f"      {LABEL[t]:<16} {c['correct']:>4}/{c['n']:<5} {acc}")
    lines.append(f"    capture         {100 * m['capture']:5.1f}%"
                 f"   (realized {m['realized_sum']:+.3f}"
                 f" of {m['oracle_sum']:+.3f})")
    lines.append(f"    mean realized   {m['mean_realized']:+.4f}")
    lines.append(f"    regressions     {m['n_regress']:>4}"
                 f"   ({100 * m['regression_rate']:.1f}% of scored)")
    return "\n".join(lines)


def format_confusion(m: dict) -> str:
    corner = "truth \\ pred"
    head = f"  {corner:<18}" + "".join(f"{LABEL[p]:>16}" for p in CATEGORIES)
    lines = [head]
    for t in CATEGORIES:
        row = f"  {LABEL[t]:<18}" + "".join(
            f"{m['confusion'][t][p]:>16}" for p in CATEGORIES)
        lines.append(row)
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Run fingerprint
# ---------------------------------------------------------------------------

# Every flag that feeds the TRAINED agent. --adapt-steps / --adapt-lr /
# --adapt-unfreeze are deliberately absent: they act only on a deepcopy made
# after zero-shot is scored, so they cannot move a zero-shot number. If two runs
# disagree on a zero-shot row, the cause is in this list.
TRAINING_ARGS = (
    "agent", "min_fill", "deadzone", "folds", "fold_seed", "base_seed", "seeds",
    "holdout_frac", "epochs", "buffer_size", "batch_size", "lr", "K",
    "weight_decay", "max_grad_norm", "patience", "bandit_epsilon",
    "bandit_epsilon_final", "bandit_warm_epochs", "q_pessimism", "logit_cap",
    "entropy_coef", "entropy_coef_final", "entropy_coef_unmerge", "missing",
    "compile_failure_penalty", "score_missing", "threads",
    "supcon_coef", "supcon_warmup", "supcon_min_cells", "supcon_batch",
    "supcon_steps", "supcon_temp",
    # PPO-only, but they reach make_agent for every kind, so a run that varies
    # them is a different training run and must fingerprint differently.
    "clip_eps", "value_loss_coef",
)


def fingerprint(loops: list, args) -> list:
    """
    Lines identifying exactly what this run trains on, so two logs can be diffed.

    Zero-shot numbers are a deterministic function of (population, TRAINING_ARGS,
    seed) — `test_determinism` proves the seeded path is reproducible. So when
    two runs disagree on a zero-shot row, something in one of those two inputs
    differed, and reading it back off the logs beats guessing. Emitting this cost
    one confusing afternoon; not emitting it cost the same afternoon.
    """
    import hashlib
    keys = sorted((l["benchmark_name"], l["loop_idx"]) for l in loops)
    h = hashlib.sha1(repr(keys).encode()).hexdigest()[:12]
    benches = len({k[0] for k in keys})
    out = [f"  population {h}  ({len(keys)} loops / {benches} benchmarks)"]
    vals = " ".join(f"{n}={getattr(args, n, '-')}" for n in TRAINING_ARGS)
    for i in range(0, len(vals), 88):
        out.append(("  training args: " if i == 0 else "                 ")
                   + vals[i:i + 88])
    return out

def pairwise_accuracy(m: dict, a: str = "noop", b: str = "unmerge_unroll") -> tuple:
    """
    (accuracy, n) on the a-vs-b call alone, over loops whose truth is a or b AND
    whose prediction is a or b.

    Reported because it is the measured bottleneck, not a generic extra: those
    two categories are ~86% of the population, and on held-out applications the
    3-way accuracy hides that the model is at chance between exactly them.
    Loops predicted as the THIRD category are excluded — they are a different
    error, and folding them in turns a clean two-way call into a muddle.
    """
    conf = m["confusion"]
    correct = conf[a][a] + conf[b][b]
    crossed = conf[a][b] + conf[b][a]
    n = correct + crossed
    return (correct / n if n else float("nan")), n

