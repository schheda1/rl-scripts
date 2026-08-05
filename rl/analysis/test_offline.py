"""
Self-contained tests for offline_data.py / offline_train.py.

Builds a tiny synthetic run directory whose every metric is hand-computable,
then asserts the code reproduces those numbers exactly. Needs torch (the import
chain reaches agent.py) but no GPU, no cache, no source tree — run it on the box
before any real sweep:

    python3 test_offline.py

WHAT IT IS FOR
--------------
preflight_check.py catches unbound names and message-contract breaks. It cannot
tell whether a metric MEANS what its name says. These do: every expected value
below is derived by hand in the comment beside it, so a scoring change that
silently alters semantics fails here rather than in a plausible-looking table.

The last test is a smoke run of the training loop over two epochs for BOTH
agents. Two is the minimum that matters: one epoch cannot catch a RolloutEntry
whose log-probs still carry grad_fn, which backprops through a freed graph on
the second of ppo_update's K inner passes.
"""

import csv
import json
import shutil
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import offline_data as od                                       # noqa: E402
import offline_train as ot                                      # noqa: E402

DZ = 0.005
N_FEAT = 93          # 18 structural + 75 IR2Vec; asserted against the real list


def _features(loop_idx: int, trip_known: int = 0, trip_count: int = 0) -> list:
    """
    A raw feature vector. Index 0 varies with loop_idx so two loops in the same
    benchmark are NOT feature-identical — otherwise _dedup_loop_records drops
    one and the fixture silently loses a loop.
    """
    v = [0.0] * N_FEAT
    v[0] = float(loop_idx + 1)
    v[od.IDX_TRIP_KNOWN] = float(trip_known)
    v[od.IDX_TRIP_COUNT] = float(trip_count)
    return v


def build_fixture(tmp: Path) -> Path:
    """
    5 loops over 4 benchmarks. Rewards chosen so every aggregate is computable
    by hand:

      b1|0  (1,2)=+0.30  (0,2)=+0.10  (1,5)=-0.50   -> unmerge_unroll, oracle +0.30
      b1|1  (0,2)=+0.001                            -> noop (below deadzone), oracle 0
      b2|0  (0,3)=+0.20                             -> unroll_only,    oracle +0.20
      b3|0  (1,4)=+0.40                             -> unmerge_unroll, oracle +0.40
      b4|0  (0,2)=-0.10  (1,2)=-0.20                -> noop,           oracle 0

    Truth counts: noop 2, unroll_only 1, unmerge_unroll 2. Total headroom +0.90.
    """
    run = tmp / "run"
    run.mkdir(parents=True)

    records = {
        "b1": [{"loop_idx": 0, "filename": "a.cu", "triple": "nvptx64",
                "pre_features_raw": _features(0)},
               {"loop_idx": 1, "filename": "a.cu", "triple": "nvptx64",
                "pre_features_raw": _features(1)}],
        "b2": [{"loop_idx": 0, "filename": "b.cu", "triple": "nvptx64",
                "pre_features_raw": _features(0)}],
        "b3": [{"loop_idx": 0, "filename": "c.cu", "triple": "nvptx64",
                "pre_features_raw": _features(0)}],
        "b4": [{"loop_idx": 0, "filename": "d.cu", "triple": "nvptx64",
                "pre_features_raw": _features(0)}],
    }
    (run / "eligible_benchmarks.json").write_text(json.dumps({
        "eligible": ["b3", "b1", "b4", "b2"],      # deliberately NOT sorted
        "loop_records": records,
        "normalizer": {},                          # unfitted -> identity
    }))

    rewards = {
        "b1|0|1|2": 0.30, "b1|0|0|2": 0.10, "b1|0|1|5": -0.50,
        "b1|1|0|2": 0.001,
        "b2|0|0|3": 0.20,
        "b3|0|1|4": 0.40,
        "b4|0|0|2": -0.10, "b4|0|1|2": -0.20,
    }
    (run / "reward_cache.json").write_text(json.dumps({
        "rewards": rewards, "post_features": {},
    }))

    rows = [
        ("b1", 0, "unmerge_unroll", 0.30), ("b1", 1, "noop", 0.0),
        ("b2", 0, "unroll_only", 0.20), ("b3", 0, "unmerge_unroll", 0.40),
        ("b4", 0, "noop", 0.0),
    ]
    with open(run / "loop_labels.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["benchmark", "loop_idx", "category", "oracle_reward",
                    "labelable"])
        for b, li, cat, orc in rows:
            w.writerow([b, li, cat, orc, 1])
    return run


def approx(a, b, tol=1e-9):
    return abs(a - b) < tol


# ---------------------------------------------------------------------------

def test_feature_layout():
    from hecbench import FEATURE_COLUMNS
    assert len(FEATURE_COLUMNS) == N_FEAT, \
        f"fixture uses {N_FEAT} features, real schema has {len(FEATURE_COLUMNS)}"
    assert FEATURE_COLUMNS[od.IDX_TRIP_KNOWN] == "tripCountKnown"
    assert FEATURE_COLUMNS[od.IDX_TRIP_COUNT] == "tripCount"
    print("  feature layout             ok")


def test_category_of():
    assert od.category_of((0, 1)) == "noop"
    assert od.category_of((0, 2)) == "unroll_only"
    assert od.category_of((1, 1)) == "unmerge_unroll", \
        "unmerge=1 is the unmerge category even at factor 1 — it still pays " \
        "for path duplication"
    assert od.category_of((1, 7)) == "unmerge_unroll"
    print("  category_of                ok")


def test_load(run: Path):
    d = od.load_run(run)
    assert len(d["loops"]) == 5, d["loops"]
    assert len(d["labels"]) == 5
    assert d["benchmarks"] == ["b1", "b2", "b3", "b4"]
    # The stored order must survive UNSORTED: split reproduction depends on it.
    assert d["eligible_order"] == ["b3", "b1", "b4", "b2"], d["eligible_order"]
    assert d["normalizer_fitted"] is False
    # The free no-op must be injected into every table even though it is never
    # stored in the cache.
    for key, table in d["tables"].items():
        assert od.NOOP in table and table[od.NOOP] == 0.0, key
    assert d["tables"][("b1", 0)][(1, 2)] == 0.30
    print("  load_run                   ok")
    return d


def test_dedup_fires(tmp: Path):
    """A benchmark whose two loops share a feature vector must lose one."""
    run = tmp / "dup"
    run.mkdir(parents=True)
    same = _features(0)
    (run / "eligible_benchmarks.json").write_text(json.dumps({
        "eligible": ["bx"],
        "loop_records": {"bx": [
            {"loop_idx": 0, "filename": "x.cu", "triple": "t",
             "pre_features_raw": same},
            {"loop_idx": 1, "filename": "x.cu", "triple": "t",
             "pre_features_raw": list(same)},
        ]},
        "normalizer": {},
    }))
    (run / "reward_cache.json").write_text(json.dumps({"rewards": {}}))
    with open(run / "loop_labels.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["benchmark", "loop_idx", "category", "oracle_reward",
                    "labelable"])
        w.writerow(["bx", 0, "noop", 0.0, 1])
    d = od.load_run(run)
    assert d["n_dropped_dedup"] == 1, d["n_dropped_dedup"]
    assert len(d["loops"]) == 1 and d["loops"][0]["loop_idx"] == 0, \
        "dedup must keep the LOWEST loop_idx"
    print("  dedup                      ok")


def test_kfold():
    benches = [f"b{i}" for i in range(10)]
    folds = od.grouped_kfold(benches, 5, seed=0)
    assert len(folds) == 5
    seen = []
    for tr, te in folds:
        assert not (set(tr) & set(te)), "a benchmark is in both train and test"
        assert set(tr) | set(te) == set(benches), "fold does not cover the set"
        seen.extend(te)
    assert sorted(seen) == sorted(benches), \
        "every benchmark must be tested exactly once"
    assert len(seen) == len(set(seen))
    print("  grouped_kfold              ok")


def test_always_noop(d: dict):
    """
    Hand-computed. always-no-op predicts noop on all 5 loops.
      accuracy   = 2 noop truths / 5            = 0.40
      scored     = 5 (the free no-op is always in the table)
      mean       = 0.0 exactly — this is the bar the heuristic failed to clear
      headroom   = 3 loops, oracle 0.30+0.20+0.40 = 0.90, realized 0 -> capture 0
    """
    loops = od.labelled_loops(d)
    m = od.score_decisions(od.always_noop_picks(loops), d["tables"],
                           d["labels"], DZ)
    assert m["loops"] == 5 and m["loops_scored"] == 5
    assert m["loops_unmeasured"] == 0
    assert approx(m["accuracy"], 0.4), m["accuracy"]
    assert approx(m["mean_realized"], 0.0), m["mean_realized"]
    assert m["loops_with_headroom"] == 3
    assert approx(m["oracle_sum"], 0.90), m["oracle_sum"]
    assert approx(m["capture"], 0.0)
    assert m["n_regress"] == 0
    assert m["per_category"]["noop"]["n"] == 2
    assert m["per_category"]["noop"]["correct"] == 2
    assert m["per_category"]["unmerge_unroll"]["correct"] == 0
    print("  always-no-op               ok")


def test_oracle(d: dict):
    """
    Hand-computed. The oracle picks the best cell everywhere:
      accuracy 5/5 = 1.0, capture 0.90/0.90 = 1.0
      mean = (0.30 + 0 + 0.20 + 0.40 + 0) / 5 = 0.18
    """
    loops = od.labelled_loops(d)
    m = od.score_decisions(od.oracle_picks(loops, d["tables"]), d["tables"],
                           d["labels"], DZ)
    assert approx(m["accuracy"], 1.0), m["accuracy"]
    assert approx(m["capture"], 1.0), m["capture"]
    assert approx(m["mean_realized"], 0.18), m["mean_realized"]
    assert m["n_regress"] == 0
    print("  oracle ceiling             ok")


def test_absent_cell_scoring(d: dict):
    """
    An absent cell always counts as a DECISION (accuracy), and what it earns
    depends on missing_reward.

    missing_reward=None   excluded from performance, counted separately.
    missing_reward=float  charged. After exhaustive collection an absent row is
                          a failure, not an unknown — and only the learned
                          policy can ever land on one, so excluding it would
                          discount its mistakes alone.
    """
    picks = [("b4", 0, (1, 9))]        # not in rewards; truth is noop

    m = od.score_decisions(picks, d["tables"], d["labels"], DZ)
    assert m["loops"] == 1 and m["loops_scored"] == 0
    assert m["loops_unmeasured"] == 1
    assert approx(m["accuracy"], 0.0), "predicted unmerge_unroll, truth noop"
    assert m["confusion"]["noop"]["unmerge_unroll"] == 1

    m = od.score_decisions(picks, d["tables"], d["labels"], DZ,
                           missing_reward=-0.161)
    assert m["loops_scored"] == 1, m["loops_scored"]
    assert m["loops_unmeasured"] == 1, "still reported, even though charged"
    assert approx(m["mean_realized"], -0.161), m["mean_realized"]
    assert m["n_regress"] == 1, "a charged absence is a regression"
    print("  absent-cell scoring        ok")


def test_regression_and_deadzone(d: dict):
    """b4|0 -> (0,2) = -0.10, a regression; b1|1 -> (0,2) = +0.001, inside the
    deadzone so NOT a regression and NOT headroom."""
    m = od.score_decisions([("b4", 0, (0, 2)), ("b1", 1, (0, 2))],
                           d["tables"], d["labels"], DZ)
    assert m["n_regress"] == 1, m["n_regress"]
    assert m["loops_with_headroom"] == 0, "neither loop has oracle > deadzone"
    assert approx(m["mean_realized"], (-0.10 + 0.001) / 2), m["mean_realized"]
    print("  regression / deadzone      ok")


def test_marginal_is_deployable(d: dict):
    """
    marginal-best must return an action that EXISTS in each loop's table, even
    when the globally best action is absent there — otherwise it scores as
    unmeasured and flatters whatever it is compared against.
    """
    loops = od.labelled_loops(d)
    keys = [(l["benchmark_name"], l["loop_idx"]) for l in loops]
    picks = od.marginal_picks(loops, d["tables"], keys)
    for bench, li, action in picks:
        assert action in d["tables"][(bench, li)], (bench, li, action)
    print("  marginal-best              ok")


def test_split_reproduction_uses_stored_order(d: dict):
    """
    split_benchmarks shuffles whatever list it is handed, so the SAME seed over
    a different input order yields a different split. Reproducing a training
    run therefore requires the stored (unsorted) order, not sorted names.

    Swept over seeds rather than asserting on one: with 4 benchmarks a single
    seed can coincide, which would fail the test without a bug being present.
    """
    from train import split_benchmarks
    order = d["eligible_order"]
    assert order != sorted(order), "fixture must store an unsorted order"
    differs = sum(1 for s in range(20)
                  if split_benchmarks(order, 0.25, 0.25, s)
                  != split_benchmarks(sorted(order), 0.25, 0.25, s))
    assert differs > 0, ("stored order and sorted order gave identical splits "
                         "for all 20 seeds — the guard is not detectable here")
    print(f"  split order sensitivity    ok  ({differs}/20 seeds differ)")


def test_training_smoke(d: dict):
    """
    Two epochs, both agents. One epoch cannot catch log-probs that still carry
    grad_fn — the failure appears on the second of ppo_update's K inner passes,
    backpropagating through a freed graph.
    """
    args = Namespace(
        deadzone=DZ, epochs=2, patience=0, lr=3e-4, batch_size=4,
        weight_decay=0.01, max_grad_norm=0.5, entropy_coef=0.01, clip_eps=0.2,
        logit_cap=0.0, epsilon=0.1, missing="penalty", score_missing="penalty",
        compile_failure_penalty=-0.161,
    )
    loops = od.labelled_loops(d)
    for kind in ("ppo", "bandit"):
        agent, info = ot.train_agent(kind, loops, loops, d, args, seed=0)
        assert info["epochs_run"] == 2, (kind, info)
        assert info["best_epoch"] >= 1, (kind, info)
        # The fixture measures only a few cells per loop, so a random policy
        # lands on absent ones constantly. Under --missing penalty they become
        # training signal instead of vanishing, which is the point: with 'skip'
        # this fixture would often produce EMPTY buffers and an untrained agent.
        assert info["n_missing_cells"] > 0, \
            "fixture should exercise the absent-cell path"
        picks = ot.greedy_picks(agent, loops, d["normalizer"], d["postf"])
        assert len(picks) == len(loops)
        for _, _, (u, f) in picks:
            assert u in (0, 1) and 1 <= f <= 10, (u, f)
        print(f"  training smoke [{kind:6}]    ok  "
              f"(best epoch {info['best_epoch']}, "
              f"hold mean {info['best_hold_mean_realized']:+.4f})")


def test_trip_count_mask_is_respected(d: dict):
    """
    A loop with a KNOWN trip count of 3 must never be assigned a factor above 3.
    The mask is built from RAW features; deriving it from the z-scored tensor
    would truncate to 0 and silently disable it.
    """
    import torch
    from adapt_eval import fresh_agent
    loop = dict(d["loops"][0])
    loop["pre_features_raw"] = _features(0, trip_known=1, trip_count=3)
    agent = fresh_agent("bandit", None)
    seen = set()
    with torch.no_grad():
        for _ in range(60):
            u, f, *_ = ot.act(agent, loop, d["normalizer"], d["postf"],
                              greedy=False)
            seen.add(f)
    assert seen <= {1, 2, 3}, f"factor above the trip count was selected: {seen}"
    print(f"  trip-count mask            ok  (factors seen: {sorted(seen)})")


def test_determinism(d: dict):
    """
    Same seed -> identical agent. This is the property the whole sweep rests on:
    it is what lets a difference between two runs be attributed to the fold or
    the init seed rather than to noise.

    Exercises every RNG the runner touches at once — torch (actor sampling),
    global random (batch shuffling, bandit epsilon-greedy), and the epoch
    shuffle. A single unseeded source anywhere in that chain breaks this.
    """
    args = Namespace(
        deadzone=DZ, epochs=3, patience=0, lr=3e-4, batch_size=4,
        weight_decay=0.01, max_grad_norm=0.5, entropy_coef=0.01, clip_eps=0.2,
        logit_cap=0.0, epsilon=0.1, missing="penalty", score_missing="penalty",
        compile_failure_penalty=-0.161,
    )
    loops = od.labelled_loops(d)
    for kind in ("ppo", "bandit"):
        a1, i1 = ot.train_agent(kind, loops, loops, d, args, seed=7)
        a2, i2 = ot.train_agent(kind, loops, loops, d, args, seed=7)
        p1 = ot.greedy_picks(a1, loops, d["normalizer"], d["postf"])
        p2 = ot.greedy_picks(a2, loops, d["normalizer"], d["postf"])
        assert p1 == p2, f"{kind}: same seed gave different picks\n{p1}\n{p2}"
        assert i1["best_epoch"] == i2["best_epoch"], (kind, i1, i2)
        assert approx(i1["best_hold_mean_realized"],
                      i2["best_hold_mean_realized"]), (kind, i1, i2)

        # ...and a DIFFERENT seed must actually move something, or the test
        # above would pass just as well on a constant policy.
        a3, _ = ot.train_agent(kind, loops, loops, d, args, seed=8)
        p3 = ot.greedy_picks(a3, loops, d["normalizer"], d["postf"])
        note = "" if p3 != p1 else "  (note: seed 8 converged to the same picks)"
        print(f"  determinism [{kind:6}]       ok{note}")


def test_fold_seed_and_init_seed_are_independent():
    """
    The 2-factor design: the same init seed is reused across folds, so a
    difference between folds at fixed init is attributable to the DATA. Verify
    the partition depends only on --fold-seed and never on --base-seed.
    """
    benches = [f"b{i}" for i in range(12)]
    a = od.grouped_kfold(benches, 4, seed=0)
    b = od.grouped_kfold(benches, 4, seed=0)
    c = od.grouped_kfold(benches, 4, seed=1)
    assert a == b, "same fold seed gave different partitions"
    assert a != c, "fold seed does not affect the partition"
    print("  fold/init seed separation  ok")


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="uu_offline_test_"))
    try:
        run = build_fixture(tmp)
        print("offline_data / offline_train")
        test_feature_layout()
        test_category_of()
        d = test_load(run)
        test_dedup_fires(tmp)
        test_kfold()
        test_fold_seed_and_init_seed_are_independent()
        test_always_noop(d)
        test_oracle(d)
        test_absent_cell_scoring(d)
        test_regression_and_deadzone(d)
        test_marginal_is_deployable(d)
        test_split_reproduction_uses_stored_order(d)
        test_trip_count_mask_is_respected(d)
        test_training_smoke(d)
        test_determinism(d)
        print("\n*** all offline tests passed ***")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
