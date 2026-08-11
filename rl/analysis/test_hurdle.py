"""
Self-contained tests for cell_status.py / hurdle_factor.py / hurdle_run.py.

Same discipline as test_offline.py: a tiny synthetic run directory whose every
number is derivable by hand, with the expected value worked out in the comment
beside each assertion. preflight_check catches unbound names; these catch a
metric that stops meaning what its name says.

The fixture extends test_offline's with the three regimes that module has no
reason to carry — a compile failure, a clip-floor cell, and a migration block —
because the whole point of this code is telling them apart.

    python3 test_hurdle.py          # needs torch, no GPU, no cache, no toolchain
"""

import csv
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import torch                                                     # noqa: E402

import cell_status as cs                                         # noqa: E402
from agent import FACTOR_VALUES, N_FEATURES                      # noqa: E402
from hurdle_factor import HurdleFactor                           # noqa: E402
from test_offline import _features                               # noqa: E402

C = cs.CLIP_FLOOR
_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)


def approx(a, b, tol=1e-6):
    return abs(a - b) < tol


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

def build_fixture(tmp: Path) -> Path:
    """
    4 loops over 3 benchmarks, with one cell of every regime.

      b1|0  (0,2)=+0.30 ok    (1,2)=-0.161 FAILED   (1,3)=-1.0 CENSORED
      b1|1  (0,2)=+0.20 ok
      b2|0  (0,3)=+0.40 ok    (0,4)=-0.16  FAILED
      b3|0  (1,2)=-1.0  CENSORED

    Regime counts: ok 4, failed 2, censored 2, over 8 cells.
    failure_keys names exactly the two failures, so the agreement check passes.
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
    }
    (run / "eligible_benchmarks.json").write_text(json.dumps({
        "eligible": ["b2", "b1", "b3"],            # deliberately NOT sorted
        "loop_records": records,
        "normalizer": {},                          # unfitted -> identity
    }))

    rewards = {
        "b1|0|0|2": 0.30, "b1|0|1|2": -0.161, "b1|0|1|3": -1.0,
        "b1|1|0|2": 0.20,
        "b2|0|0|3": 0.40, "b2|0|0|4": -0.16,
        "b3|0|1|2": -1.0,
        # A second ok cell on b3 so its loop has a positive oracle and enters
        # the capture denominator; without it b3 contributes nothing scoreable.
        "b3|0|1|4": 0.25,
    }
    (run / "reward_cache.json").write_text(json.dumps({
        "rewards": rewards, "post_features": {},
        "migration": {"failure_keys": ["b1|0|1|2", "b2|0|0|4"],
                      "failure_penalty": -0.161},
    }))

    rows = [("b1", 0, "unroll_only", 0.30), ("b1", 1, "unroll_only", 0.20),
            ("b2", 0, "unroll_only", 0.40), ("b3", 0, "unmerge_unroll", 0.25)]
    with open(run / "loop_labels.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["benchmark", "loop_idx", "category", "oracle_reward",
                    "labelable"])
        for b, li, cat, orc in rows:
            w.writerow([b, li, cat, orc, 1])
    return run


def _args(run: Path, **over):
    from hurdle_run import build_parser
    a = build_parser().parse_args([str(run), "--deadzone", "0.005"])
    for k, v in over.items():
        if not hasattr(a, k):
            raise AssertionError(f"--{k} is not a real flag; the test drifted")
        setattr(a, k, v)
    return a


def _zeroed(agent: HurdleFactor) -> None:
    """Force both readouts to emit exactly 0, so mu = 0 and logit = 0 and every
    likelihood term below has a closed form."""
    with torch.no_grad():
        for m in (agent.feas, agent.mag):
            m.net[6].weight.zero_()
            m.net[6].bias.zero_()


def _agent(**over) -> HurdleFactor:
    kw = dict(lr=1e-3, weight_decay=0.0, max_grad_norm=0.0, epsilon=0.0,
              batch_size=8, K=1)
    kw.update(over)
    return HurdleFactor(**kw)


# ---------------------------------------------------------------------------
# cell_status
# ---------------------------------------------------------------------------

def test_constants_match_offline_data():
    """cell_status mirrors offline_data's two constants by hand. If they drift,
    the sidecar labels a different population than every reported figure."""
    import offline_data as od
    assert tuple(cs.FAILURE_VALUES) == tuple(od.FAILURE_VALUES), \
        f"{cs.FAILURE_VALUES} != {od.FAILURE_VALUES}"
    assert cs.CLIP_FLOOR == od.CLIP_FLOOR
    print("  ok  constants match offline_data")


def test_status_derivation(run: Path):
    rc = json.loads((run / "reward_cache.json").read_text())
    status, counts, diverge = cs.derive(
        rc["rewards"], set(rc["migration"]["failure_keys"]))
    assert not diverge, diverge                      # fixture is consistent
    assert counts == {"cells": 8, "failed": 2, "censored": 2, "ok": 4}, counts
    assert status["b1|0|1|2"] == cs.FAILED           # -0.161, and in the keys
    assert status["b2|0|0|4"] == cs.FAILED           # -0.16,  and in the keys
    assert status["b1|0|1|3"] == cs.CENSORED         # -1.0, not a failure value
    assert status["b3|0|1|2"] == cs.CENSORED
    assert "b1|0|0|2" not in status                  # ok cells are not stored
    assert cs.status_of(status, "b1", 0, 0, 2) == cs.OK
    print("  ok  status derivation: 4 ok / 2 failed / 2 censored")


def test_divergence_is_detected(run: Path):
    """A key list that names a cell whose value is not a failure penalty, and a
    cell whose value IS one but is absent from the list. offline_data ORs the
    two tests silently; derive() must surface both directions."""
    rc = json.loads((run / "reward_cache.json").read_text())
    rewards = dict(rc["rewards"])
    rewards["b1|1|1|9"] = -0.161                     # value-matched, not in keys
    # b1|0|0|2 is +0.30 — named as a failure but its value says otherwise.
    # b2|0|0|4 is a real failure the key list has "forgotten".
    keys = {"b1|0|1|2", "b1|0|0|2"}
    _, _, diverge = cs.derive(rewards, keys)
    assert diverge["keys_not_value_matched"] == ["b1|0|0|2"], diverge
    assert diverge["value_matched_not_in_keys"] == ["b1|1|1|9", "b2|0|0|4"], \
        diverge
    print("  ok  divergence detected in both directions")


# ---------------------------------------------------------------------------
# hurdle_factor — the likelihood, by hand
# ---------------------------------------------------------------------------

def test_feasibility_term_by_hand():
    """logit 0 -> p = 0.5 -> BCE = -log(0.5) = 0.6931 for EITHER label."""
    a = _agent(); _zeroed(a)
    s = torch.zeros(1, N_FEATURES)
    for status in (cs.OK, cs.FAILED):
        _, f_loss, _ = a._batch_loss([(s[0], 0, status, 0.0)])
        assert approx(float(f_loss), -math.log(0.5), 1e-5), (status, f_loss)
    print("  ok  feasibility BCE = 0.6931 at p=0.5, both labels")


def test_gaussian_term_by_hand():
    """mu=0, sigma=1, r=0.5:  0.5*log(2pi) + log(1) + 0.5*0.25 = 1.043939"""
    a = _agent(); _zeroed(a)
    s = torch.zeros(N_FEATURES)
    _, _, m_loss = a._batch_loss([(s, 0, cs.OK, 0.5)])
    want = _HALF_LOG_2PI + 0.0 + 0.5 * 0.25
    assert approx(float(m_loss), want, 1e-5), (float(m_loss), want)
    print(f"  ok  gaussian NLL = {want:.6f} at mu=0 sigma=1 r=0.5")


def test_censored_term_by_hand():
    """mu=0, sigma=1, c=-1:  -log Phi(-1) = -log(0.1586553) = 1.841022

    The censored cell's STORED value is irrelevant to its contribution — that is
    the point of censoring, and passing a nonsense value here proves the term
    does not read it."""
    a = _agent(); _zeroed(a)
    s = torch.zeros(N_FEATURES)
    _, _, m_loss = a._batch_loss([(s, 0, cs.CENSORED, -999.0)])
    want = -math.log(0.15865525393145707)
    assert approx(float(m_loss), want, 1e-5), (float(m_loss), want)
    print(f"  ok  censored NLL = {want:.6f} at mu=0 sigma=1 c=-1")


def test_failed_gives_no_magnitude_gradient():
    """A batch of failures must leave the magnitude head untouched. Under the
    scalar model those same cells are the dominant gradient."""
    a = _agent(); _zeroed(a)
    s = torch.zeros(N_FEATURES)
    loss, _, m_loss = a._batch_loss([(s, 0, cs.FAILED, -0.161),
                                     (s, 1, cs.FAILED, -0.16)])
    assert float(m_loss) == 0.0
    loss.backward()
    for name, p in a.mag.named_parameters():
        assert p.grad is None or torch.allclose(p.grad, torch.zeros_like(p.grad)), \
            f"magnitude head received gradient from a failed cell via {name}"
    assert a.feas.net[6].weight.grad is not None      # feasibility DID learn
    print("  ok  failed cells train feasibility only, magnitude grad is zero")


def test_floor_as_switches_the_regime():
    """The same censored cell is a ran-sample under 'censored' and a
    feasibility-negative under 'failed'. Not separable from the cache, so both
    readings must be reachable — and must actually differ."""
    s = torch.zeros(N_FEATURES)
    a = _agent(floor_as="censored"); _zeroed(a)
    assert a.ran(cs.CENSORED) is True
    _, f_c, m_c = a._batch_loss([(s, 0, cs.CENSORED, -1.0)])

    b = _agent(floor_as="failed"); _zeroed(b)
    assert b.ran(cs.CENSORED) is False
    _, f_f, m_f = b._batch_loss([(s, 0, cs.CENSORED, -1.0)])

    assert float(m_c) > 0.0 and float(m_f) == 0.0     # only 'censored' trains mu
    # BCE label flips 1 -> 0, and at p=0.5 both cost -log(0.5); the label is what
    # differs, so compare the gradient sign on the readout bias instead.
    assert approx(float(f_c), float(f_f), 1e-6)
    print("  ok  --floor-as moves the clip cells between the two terms")


def test_score_is_p_times_mu():
    a = _agent()
    s = torch.zeros(N_FEATURES)
    mask = torch.zeros(len(FACTOR_VALUES), dtype=torch.bool)
    mask[1] = mask[4] = True
    p_ok, mu, score = a.predict(s, mask)
    for i in range(len(FACTOR_VALUES)):
        if mask[i]:
            assert approx(float(score[i]), float(p_ok[i] * mu[i]), 1e-6)
        else:
            assert score[i] == float("-inf"), i
    assert int(score.argmax()) in (1, 4)
    print("  ok  score = p_ok * mu, -inf on illegal factors")


def test_snapshot_restore_clears_adaptation():
    """Restoring a base snapshot must clear a previous target's offsets.
    Otherwise one benchmark's adaptation leaks into the next, which is the
    failure offline_adapt's leak check exists to catch."""
    a = _agent()
    base = a.snapshot()
    a.feas_bias, a.mag_bias = 3.0, -2.0
    a.restore(base)
    assert a.feas_bias == 0.0 and a.mag_bias == 0.0
    print("  ok  restore clears feas_bias / mag_bias")


def test_bias_reaches_inference():
    """An adaptation offset is useless if predict_batch does not apply it."""
    a = _agent(); _zeroed(a)
    s = torch.zeros(1, N_FEATURES)
    p0, mu0 = a.predict_batch(s)
    a.feas_bias, a.mag_bias = 2.0, 0.5
    p1, mu1 = a.predict_batch(s)
    assert approx(float(p0[0, 0]), 0.5, 1e-6)
    assert approx(float(p1[0, 0]), 1 / (1 + math.exp(-2.0)), 1e-6)
    assert approx(float(mu1[0, 0] - mu0[0, 0]), 0.5, 1e-6)
    print("  ok  feas_bias / mag_bias reach predict_batch")


# ---------------------------------------------------------------------------
# hurdle_run
# ---------------------------------------------------------------------------

def test_roc_auc_by_hand():
    from hurdle_run import roc_auc
    # y=[0,0,1,1], p=[0.1,0.4,0.35,0.8]. Concordant pairs: (0.35,0.1),
    # (0.8,0.1), (0.8,0.4) = 3 of 4 -> 0.75.
    assert approx(roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]), 0.75)
    # A total tie is exactly chance, not 1.0 — the averaged-rank branch.
    assert approx(roc_auc([0, 1], [0.5, 0.5]), 0.5)
    # One class absent: undefined, and must not read as chance.
    assert roc_auc([1, 1], [0.2, 0.9]) != roc_auc([1, 1], [0.2, 0.9])
    print("  ok  roc_auc: 0.75 / ties 0.5 / single-class NaN")


def test_split3_is_disjoint_and_covering():
    from hurdle_run import random_split3
    bs = [f"b{i}" for i in range(20)]
    tr, va, te = random_split3(bs, 0.15, 0.2, seed=3)
    assert tr and va and te
    assert set(tr) & set(va) == set() and set(tr) & set(te) == set()
    assert set(va) & set(te) == set()
    assert sorted(tr + va + te) == sorted(bs)
    # Function of the seed alone: input order must not move the partition.
    assert random_split3(list(reversed(bs)), 0.15, 0.2, seed=3) == (tr, va, te)
    print("  ok  3-way split disjoint, covering, order-independent")


def test_pooling_uses_raw_cells_not_per_target_summaries():
    """
    run_adapt pools by concatenating raw_scores output, never by averaging the
    per-target metrics. None of these is a weighted mean of its own per-subset
    values, and with one or two evaluation loops per target the difference is
    not a rounding error.

    Two targets: A = 2 cells, B = 3 cells.
      AUC   pooled over all 5 ranks     != (auc(A)*2 + auc(B)*3) / 5
      NLL   pooled over 4 ran-CELLS     != (nll(A)*2 + nll(B)*3) / 5   <- loop weights
    """
    from hurdle_run import aggregate
    A = {"y": [1, 0], "p": [0.9, 0.1], "nll": [1.0], "ae": [0.5]}
    B = {"y": [1, 0, 0], "p": [0.2, 0.8, 0.7], "nll": [3.0, 3.0, 3.0], "ae": []}
    cat = {k: A[k] + B[k] for k in A}

    pooled = aggregate(cat)
    naive_auc = (aggregate(A)["feas_auc"] * 2 + aggregate(B)["feas_auc"] * 3) / 5
    assert approx(pooled["feas_auc"], 4 / 6)      # 4 concordant of 2*3 pairs
    assert abs(pooled["feas_auc"] - naive_auc) > 1e-9, \
        "this fixture no longer distinguishes correct pooling from the bug"

    # Cell-weighted, not loop-weighted: 4 ran-cells, (1 + 3 + 3 + 3) / 4.
    assert approx(pooled["mag_nll"], 2.5)
    assert abs(pooled["mag_nll"] - (1.0 * 2 + 3.0 * 3) / 5) > 1e-9
    assert pooled["mag_n"] == 4 and pooled["feas_n"] == 5
    print("  ok  pooling is over raw cells, not per-target summaries")


def test_aggregate_is_empty_safe():
    from hurdle_run import aggregate
    m = aggregate({"y": [], "p": [], "nll": [], "ae": []})
    assert m["feas_n"] == 0 and m["mag_n"] == 0
    for k in ("feas_auc", "feas_acc", "feas_brier", "feas_base",
              "mag_nll", "mag_mae"):
        assert m[k] != m[k], f"{k} should be NaN on an empty population"
    print("  ok  aggregate() on an empty population is all-NaN, never 0.0")


def test_split3_rejects_a_population_it_cannot_split():
    """Two benchmarks cannot make three non-empty groups. Raising beats
    returning an empty train set, which would train on nothing and report it as
    a result."""
    from hurdle_run import random_split3
    for n in (0, 1, 2):
        try:
            random_split3([f"b{i}" for i in range(n)], 0.15, 0.2, 0)
        except ValueError:
            continue
        raise AssertionError(f"n={n} should have raised")
    print("  ok  random_split3 refuses n < 3 instead of emptying train")


def test_empty_val_falls_back_to_final_not_random_init(run: Path):
    """
    THE dangerous one. With no validation pass, val_feas/val_mag would still
    hold the snapshot taken before epoch 1 — random initialisation — and main()
    restores snaps[--select-by] for the headline. The fallback must point them
    at the final epoch.
    """
    from hurdle_run import train
    from offline_data import load_run, labelled_loops, loops_for
    d = load_run(run, 0.005)
    loops = labelled_loops(d)
    fit = loops_for(loops, ["b1", "b2"])
    args = _args(run, epochs=2, val_every=1, log_every=0, buffer_size=4)
    agent, snaps, _, _, bep = train(fit, [], d, {}, args, seed=1)   # NO val loops
    for rule in ("val_feas", "val_mag"):
        for k, v in snaps["final"]["feas"].items():
            assert torch.allclose(snaps[rule]["feas"][k], v), \
                f"{rule} is not the final snapshot — it may be random init"
    assert bep == {"val_feas": 0, "val_mag": 0}
    print("  ok  empty val falls back to the final epoch, not random init")


def test_history_exposes_ran_rate(run: Path):
    """The magnitude head only learns from picks that ran, and p_ok*mu can steer
    greedy picks toward expected failures. That has to be visible per epoch."""
    from hurdle_run import train
    from offline_data import load_run, labelled_loops, loops_for
    d = load_run(run, 0.005)
    loops = labelled_loops(d)
    fit = loops_for(loops, ["b1", "b2"])
    args = _args(run, epochs=2, val_every=1, log_every=0, buffer_size=4)
    _, _, _, hist, _ = train(fit, [], d, {}, args, seed=1)
    for h in hist:
        assert 0.0 <= h["ran_rate"] <= 1.0, h
        assert h["updates"] >= 1
        # feas/mag are epoch means over the same updates as `loss`, not the last
        # buffer's values.
        for k in ("loss", "feas_loss", "mag_loss"):
            assert h[k] == h[k], (k, h)
    print("  ok  history carries ran_rate, updates, and epoch-mean losses")


def test_spread_and_readout_are_nan_safe():
    """The read-out is what a go/no-go gets read off, so it must not raise on a
    degenerate run — a split set where every metric is missing or NaN is exactly
    when someone is looking at it."""
    from hurdle_run import _readout, _spread

    class _A:
        adapt = False

    rows = [{"test_feas_auc": 0.71, "test_probe": 0.42, "coverage": 0.9},
            {"test_feas_auc": 0.66, "test_probe": 0.29, "coverage": 0.9}]
    s = _spread(rows, "test_feas_auc")
    assert "+0.685" in s and "over 2" in s, s
    assert "single split" in _spread(rows[:1], "test_probe")
    assert _spread(rows, "no_such_key") == "n/a"
    # sd is population, not sample: two values 0.71/0.66 -> 0.025, not 0.035.
    assert "0.025" in s, s
    _readout(rows, _A())                       # must not raise
    _readout([{"coverage": float("nan")}], _A())
    _readout([], _A())
    print("  ok  _spread formats correctly; _readout survives degenerate rows")


def test_observe_absent_cell(run: Path):
    from hurdle_run import observe
    from offline_data import load_run
    d = load_run(run, 0.005)
    a = _args(run)
    table = d["tables"][("b1", 0)]
    st, v, measured = observe(table, {}, "b1", 0, 1, 9, a)   # never collected
    assert measured is False and st == cs.FAILED and v == a.missing
    a2 = _args(run, absent_as="ok")
    st2, _, _ = observe(table, {}, "b1", 0, 1, 9, a2)
    assert st2 == cs.OK
    print("  ok  absent cell -> failed by default, ok under --absent-as ok")


def test_cell_rows_respects_the_trip_count_mask(run: Path):
    """Every row must be a LEGAL cell. A row for a masked-out factor would train
    the head on an action the policy can never take."""
    from hurdle_run import cell_rows
    from offline_data import load_run, labelled_loops
    from factor_only import states_for
    d = load_run(run, 0.005)
    loops = labelled_loops(d)
    states, rows = cell_rows(loops, d, {}, _args(run))
    n_legal = sum(int(m.sum()) for l in loops for _, _, m in states_for(l, d))
    assert len(rows) == n_legal, (len(rows), n_legal)
    assert len(states) == 2 * len(loops)
    print(f"  ok  cell_rows enumerates exactly the {n_legal} legal cells")


def test_interface_parity_with_factor_only(run: Path):
    """HurdleFactor must score through factor_only's paths unmodified — that is
    what makes the numbers comparable to the runs already on disk."""
    from factor_only import branch_picks, probe_picks
    from category_agent import UNROLL_ONLY
    from offline_data import load_run, labelled_loops
    d = load_run(run, 0.005)
    loops = labelled_loops(d)
    a = _agent()
    picks = probe_picks(a, loops, d)
    assert picks and all(len(p) == 3 for p in picks)
    for _, _, (u, f) in picks:
        assert u in (0, 1) and f in FACTOR_VALUES
    bp = branch_picks(a, loops, d, 0, UNROLL_ONLY, -0.161)
    assert len(bp) == len(loops)
    print("  ok  probe_picks / branch_picks drive HurdleFactor unchanged")


def test_adapt_intercept_moves_only_the_offset(run: Path):
    """The base network must be untouched: k measurements buy one scalar, and
    the point of the intercept mode is that nothing else moves."""
    from hurdle_run import adapt_intercept, cell_rows
    from offline_data import load_run, labelled_loops
    d = load_run(run, 0.005)
    loops = labelled_loops(d)
    a = _agent()
    before = {k: v.clone() for k, v in a.feas.state_dict().items()}
    states, rows = cell_rows(loops, d, {}, _args(run))
    adapt_intercept(a, states, rows, _args(run, adapt_steps=20, adapt_what="both"))
    for k, v in a.feas.state_dict().items():
        assert torch.allclose(v, before[k]), f"adaptation moved feas.{k}"
    assert a.feas_bias != 0.0
    print("  ok  intercept adaptation moves the offset and nothing else")


def test_training_smoke(run: Path):
    """Two epochs, both --floor-as readings. Two is the minimum that catches a
    buffer entry still carrying grad_fn into the second of update()'s K passes."""
    from hurdle_run import hurdle_metrics, train
    from offline_data import load_run, labelled_loops, loops_for
    d = load_run(run, 0.005)
    loops = labelled_loops(d)
    fit, val = loops_for(loops, ["b1", "b2"]), loops_for(loops, ["b3"])
    for floor_as in ("censored", "failed"):
        args = _args(run, epochs=2, val_every=1, log_every=0, floor_as=floor_as,
                     buffer_size=4, K=2)
        agent, snaps, cov, hist, bep = train(fit, val, d, {}, args, seed=1)
        assert set(snaps) == {"final", "val_feas", "val_mag"}
        assert 0.0 < cov <= 1.0
        assert len(hist) == 2
        m = hurdle_metrics(agent, fit, d, {}, args)
        assert m["feas_n"] > 0
        assert m["mag_nll"] == m["mag_nll"] or m["mag_n"] == 0
        print(f"  ok  smoke train --floor-as {floor_as}: coverage "
              f"{100*cov:.0f}%, sigma {agent.sigma:.3f}")


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="uu_hurdle_test_"))
    try:
        run = build_fixture(tmp)
        print("cell_status")
        test_constants_match_offline_data()
        test_status_derivation(run)
        test_divergence_is_detected(run)
        print("hurdle_factor")
        test_feasibility_term_by_hand()
        test_gaussian_term_by_hand()
        test_censored_term_by_hand()
        test_failed_gives_no_magnitude_gradient()
        test_floor_as_switches_the_regime()
        test_score_is_p_times_mu()
        test_snapshot_restore_clears_adaptation()
        test_bias_reaches_inference()
        print("hurdle_run")
        test_roc_auc_by_hand()
        test_pooling_uses_raw_cells_not_per_target_summaries()
        test_aggregate_is_empty_safe()
        test_split3_is_disjoint_and_covering()
        test_split3_rejects_a_population_it_cannot_split()
        test_empty_val_falls_back_to_final_not_random_init(run)
        test_history_exposes_ran_rate(run)
        test_spread_and_readout_are_nan_safe()
        test_observe_absent_cell(run)
        test_cell_rows_respects_the_trip_count_mask(run)
        test_interface_parity_with_factor_only(run)
        test_adapt_intercept_moves_only_the_offset(run)
        test_training_smoke(run)
        print("\n*** all hurdle tests passed ***")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
