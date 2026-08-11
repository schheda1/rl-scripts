"""
Runner for the hurdle + censored unroller. Train / zero-shot test / few-shot.

WHAT IT ANSWERS
---------------
The scalar factor agent reports ONE number, and that number averages two
questions that can succeed and fail independently:

    does the model know which cells will not BUILD?
    does it know which of the surviving factors is FASTEST?

This runner never collapses them. Feasibility and magnitude are trained by
different terms of the same likelihood (hurdle_factor.py), selected on different
validation statistics, and reported in separate blocks.

PROTOCOL
--------
Independent random three-way splits over BENCHMARKS -- train / val / test --
repeated `--splits` times. Grouped by benchmark for the reason factor_only
states at :144: loops inside a benchmark share source and kernels, so splitting
one across the boundary leaks the answer. Three-way rather than factor_only's
two, because model selection needs a slice that is neither trained on nor
reported.

Training is see-as-you-go, unchanged in shape from factor_only.train: sweep
every (loop, branch) cell each epoch, let the agent pick a factor, "measure" it
(a cache lookup standing in for compile+run), buffer, update. The ONLY
difference is the payload -- (state, idx, status, value) instead of
(state, idx, reward). One measurement, two labels; the measurement budget is
identical.

SELECTION
---------
Not `mean_realized`. That statistic is heavy-tailed over a small held-out slice
-- 744 cells at the -1.0 clip, plus failure penalties -- so its argmax over
epochs is close to a random draw, which is why best-val epochs land anywhere.
Val here is scored on two bounded statistics with many samples apiece:
feasibility AUC, and censored NLL on the cells that ran. Test is scored under
every rule (`final`, `val_feas`, `val_mag`) so the spread between them is
visible rather than hidden by whichever one was picked.

Reading true outcomes for val and test cells is evaluation, not training: the
see-as-you-go constraint binds what the AGENT may condition on, and
factor_only.evaluate already reads data["tables"] for held-out loops.

COMPARABILITY
-------------
The decision block comes from factor_only.evaluate, unmodified, so probe capture
and the constant-factor bar are computed by exactly the code that produced the
runs already on disk. HurdleFactor is interface-compatible, so probe_picks and
branch_picks score it without knowing what it is.

Usage:
  python3 cell_status.py RUN_DIR                     # once, writes the sidecar
  python3 hurdle_run.py RUN_DIR --deadzone 0.005
  python3 hurdle_run.py RUN_DIR --deadzone 0.005 --adapt --csv-out rows.csv
"""

import argparse
import json
import logging
import math
import random
import sys
# `fmean` by name, not `import statistics as st`: `st` is this directory's
# convention for a cell STATUS and is bound as a local in three functions here.
# The alias would be shadowed in all three, and would break the moment one of
# them grew a call to it.
from statistics import fmean, pstdev
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import torch                                                     # noqa: E402
import torch.nn.functional as F                                  # noqa: E402

from agent import FACTOR_VALUES                                  # noqa: E402
from cell_status import (CENSORED, CLIP_FLOOR, FAILED, OK,       # noqa: E402
                         derive, load_status, status_of)
from factor_only import evaluate as decision_eval, states_for    # noqa: E402
from offline_data import (labelled_loops, load_run, loops_for)   # noqa: E402
from offline_train import write_csv                              # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
log = logging.getLogger("hurdle")

DEVICE = torch.device("cpu")
SELECT_RULES = ("final", "val_feas", "val_mag")


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def random_split3(benchmarks: list, val_frac: float, test_frac: float,
                  seed: int) -> tuple:
    """
    (train, val, test) benchmark names.

    MIRROR: factor_only.random_split:144 -- sorted() before shuffling so the
    draw is a function of the seed alone and not of dict iteration order, and
    the grouping is by BENCHMARK because loops inside one share the answer.

    Sizes are floored at 1 for val and test and at 1 for train, then the split
    is taken by slicing a single permutation, so the three sets are disjoint by
    construction rather than by an intersection check.
    """
    bs = sorted(benchmarks)
    random.Random(seed).shuffle(bs)
    n = len(bs)
    if n < 3:
        raise ValueError(
            f"need at least 3 benchmarks for a train/val/test split, got {n}")
    n_test = max(1, round(test_frac * n))
    n_val = max(1, round(val_frac * n))
    # Train must keep at least one benchmark. Shrink test first, then val:
    # a thin val slice degrades model selection, an empty train degrades
    # everything, so they are not equally safe to give up.
    while n_test + n_val > n - 1:
        if n_test > 1:
            n_test -= 1
        elif n_val > 1:
            n_val -= 1
        else:                                    # n == 3: 1/1/1 is the floor
            break
    return bs[n_test + n_val:], bs[n_test:n_test + n_val], bs[:n_test]


# ---------------------------------------------------------------------------
# The measurement stand-in
# ---------------------------------------------------------------------------

def observe(table: dict, status_map: dict, bench: str, loop_idx: int,
            u: int, f: int, args) -> tuple:
    """
    (status, value, measured) -- one "measurement", offline.

    A cell present in the cache carries its stored value and its derived regime.
    A cell ABSENT is a different thing from a cell that failed, and the
    distinction only became expressible once status was stored: absent means
    never attempted. The default treats it as FAILED, matching the reasoning
    already in factor_only's --missing ("after exhaustive collection an absent
    row means it FAILED"), with --absent-as ok available to test how much of a
    result rides on that reading.
    """
    if (u, f) in table:
        return status_of(status_map, bench, loop_idx, u, f), table[(u, f)], True
    return (FAILED if args.absent_as == "failed" else OK), args.missing, False


def cell_rows(loops: list, data: dict, status_map: dict, args) -> tuple:
    """
    Every LEGAL (loop, branch, factor) cell of a population, with its true
    outcome. Evaluation only -- the agent never calls this.

    Returns (states, rows) where states is one tensor per (loop, branch) and
    each row is (state_index, factor_index, status, value). Structured this way
    so the whole population goes through predict_batch in one forward pass
    instead of one per cell.
    """
    states, rows = [], []
    for l in loops:
        key = (l["benchmark_name"], l["loop_idx"])
        table = data["tables"][key]
        for u, s, m in states_for(l, data):
            si = len(states)
            states.append(s)
            for fi, legal in enumerate(m.tolist()):
                if not legal:
                    continue
                st, v, _ = observe(table, status_map, key[0], key[1],
                                   u, FACTOR_VALUES[fi], args)
                rows.append((si, fi, st, v))
    return states, rows


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def roc_auc(y: list, p: list) -> float:
    """
    AUC by the rank identity, with ties averaged. No sklearn -- nothing else in
    this directory imports it, and a rank sum is four lines.

    NaN when one class is absent: AUC is undefined there, and returning 0.5
    would read as "chance" when the truth is "not measurable on this slice".
    """
    n_pos = sum(1 for v in y if v)
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = sorted(range(len(p)), key=lambda i: p[i])
    ranks = [0.0] * len(p)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and p[order[j + 1]] == p[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0            # 1-based, averaged over the tie run
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    s = sum(r for r, yy in zip(ranks, y) if yy)
    return (s - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def raw_scores(agent, states: list, rows: list) -> dict:
    """
    Per-cell ingredients, from ONE forward pass over the population.

    Returned rather than aggregated because several of these metrics cannot be
    pooled from their own summaries. An AUC over a union is not any weighted
    average of per-subset AUCs, and a mean NLL pools by CELL count, not by loop
    count. run_adapt concatenates these across targets and aggregates once;
    aggregating first and averaging after is the bug this shape prevents.

      y    did the cell run          (per legal cell)
      p    P(ok) the model assigned  (per legal cell)
      nll  censored NLL              (per cell that RAN)
      ae   |v - mu|                  (per cell that ran and is NOT censored)
    """
    empty = {"y": [], "p": [], "nll": [], "ae": []}
    if not rows:
        return empty
    p_ok, mu = agent.predict_batch(torch.stack(states))
    si = torch.tensor([r[0] for r in rows], dtype=torch.long)
    fi = torch.tensor([r[1] for r in rows], dtype=torch.long)
    out = {"y": [agent.ran(r[2]) for r in rows],
           "p": p_ok[si, fi].tolist(), "nll": [], "ae": []}

    ran = [r for r in rows if agent.ran(r[2])]
    if not ran:
        return out
    rsi = torch.tensor([r[0] for r in ran], dtype=torch.long)
    rfi = torch.tensor([r[1] for r in ran], dtype=torch.long)
    m = mu[rsi, rfi]
    v = torch.tensor([r[3] for r in ran], dtype=torch.float32)
    cens = torch.tensor([r[2] == CENSORED for r in ran], dtype=torch.bool)
    sigma = agent.sigma
    z = (v - m) / sigma
    gauss = 0.5 * math.log(2 * math.pi) + math.log(sigma) + 0.5 * z * z
    censored = -torch.special.log_ndtr((CLIP_FLOOR - m) / sigma)
    out["nll"] = torch.where(cens, censored, gauss).tolist()
    # Observed cells only: the residual of a censored cell is undefined, since
    # its true value is unknown below the clip.
    out["ae"] = (v[~cens] - m[~cens]).abs().tolist()
    return out


def aggregate(raw: dict) -> dict:
    """
    The reported metrics, from raw_scores output (or a concatenation of several).

    `feas_base` sits beside `feas_acc` deliberately: with most cells running, an
    accuracy equal to the base rate is the CONSTANT predictor, and the accuracy
    is uninterpretable without it on the same line.

    `mag_mae` sits beside `mag_nll` for the same reason in the other direction:
    an NLL improves when sigma widens, so the pair separates "fits better" from
    "admits it does not know".
    """
    y, p, nll, ae = raw["y"], raw["p"], raw["nll"], raw["ae"]
    n = len(y)
    return {
        "feas_auc": roc_auc(y, p) if n else float("nan"),
        "feas_acc": (sum(1 for yy, pp in zip(y, p) if (pp >= 0.5) == yy) / n
                     if n else float("nan")),
        "feas_brier": (sum((pp - float(yy)) ** 2 for yy, pp in zip(y, p)) / n
                       if n else float("nan")),
        "feas_base": sum(1 for yy in y if yy) / n if n else float("nan"),
        "feas_n": n,
        "mag_nll": fmean(nll) if nll else float("nan"),
        "mag_mae": fmean(ae) if ae else float("nan"),
        "mag_n": len(nll),
    }


def hurdle_metrics(agent, loops: list, data: dict, status_map: dict,
                   args) -> dict:
    """Convenience wrapper. Callers in a hot loop should hoist cell_rows out and
    use raw_scores/aggregate directly — cell_rows re-walks every loop and
    re-normalises every state, and nothing about it changes between epochs."""
    states, rows = cell_rows(loops, data, status_map, args)
    return aggregate(raw_scores(agent, states, rows))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(fit_loops: list, val_loops: list, data: dict, status_map: dict,
          args, seed: int) -> tuple:
    """(agent, snapshots, coverage, history, best_epoch).

    Shape is factor_only.train's, deliberately: same per-epoch sweep over every
    (loop, branch) cell, same epsilon schedule, same buffer cadence, same
    coverage denominator over LEGAL cells only. What differs is one field in the
    buffer payload and what the loss does with it.
    """
    from hurdle_factor import HurdleFactor

    torch.manual_seed(seed)
    random.seed(seed)
    agent = HurdleFactor(args.lr, args.weight_decay, args.max_grad_norm,
                         args.epsilon, args.batch_size, args.K,
                         floor_as=args.floor_as, mag_coef=args.mag_coef)

    cells = [(l, u, s, m) for l in fit_loops for u, s, m in states_for(l, data)]
    n_legal = sum(int(m.sum()) for _, _, _, m in cells)
    seen = set()

    # Hoisted: cell_rows re-walks every loop and re-normalises every state, and
    # its output cannot change between epochs. Inside the loop it would be
    # rebuilt on every validation pass for no reason.
    val_states, val_rows = ((None, None) if not val_loops
                            else cell_rows(val_loops, data, status_map, args))

    snaps = {r: agent.snapshot() for r in SELECT_RULES}
    best = {"val_feas": -float("inf"), "val_mag": float("inf")}
    best_epoch = {"val_feas": 0, "val_mag": 0}
    n_val_evals = 0

    buf, history = [], []
    for epoch in range(1, args.epochs + 1):
        frac = (epoch - 1) / (args.epochs - 1) if args.epochs > 1 else 0.0
        agent.epsilon = args.epsilon + frac * (args.epsilon_final - args.epsilon)
        order = list(range(len(cells)))
        random.shuffle(order)
        # All three accumulate per UPDATE, so the three numbers on one log line
        # are the same aggregate over the same set. Reading last_feas_loss
        # straight off the agent would report the final buffer's mean beside the
        # epoch's mean and label them alike.
        loss_sum = feas_sum = mag_sum = 0.0
        n_upd = n_ran = 0

        def _flush(b):
            nonlocal loss_sum, feas_sum, mag_sum, n_upd
            loss_sum += agent.update(b)
            feas_sum += agent.last_feas_loss
            mag_sum += agent.last_mag_loss
            n_upd += 1

        for i in order:
            l, u, s, m = cells[i]
            key = (l["benchmark_name"], l["loop_idx"])
            idx = agent.select(s, m)
            f = FACTOR_VALUES[idx]
            st_, v, _ = observe(data["tables"][key], status_map,
                                key[0], key[1], u, f, args)
            seen.add((key, u, f))
            n_ran += int(agent.ran(st_))
            buf.append((s, idx, st_, v))
            if len(buf) >= args.buffer_size:
                _flush(buf)
                buf = []
        if buf:
            _flush(buf)
            buf = []

        # NaN, not 0.0, when nothing was updated: a zero loss reads as "fit
        # perfectly" and is the opposite of what an empty epoch means.
        d = n_upd if n_upd else float("nan")
        row = {"epoch": epoch, "loss": loss_sum / d, "feas_loss": feas_sum / d,
               "mag_loss": mag_sum / d, "sigma": agent.sigma,
               "epsilon": agent.epsilon, "updates": n_upd,
               "coverage": len(seen) / n_legal if n_legal else float("nan"),
               # Fraction of this epoch's PICKS that produced a running program.
               # The magnitude head only learns from these, and the score rule
               # p_ok*mu can steer greedy picks toward expected failures once a
               # branch looks hopeless (see HurdleFactor.select). If this decays
               # toward zero while coverage stalls, the magnitude head starved
               # and its NLL is describing a shrinking sample, not a better fit.
               "ran_rate": n_ran / len(cells) if cells else float("nan")}

        # Val is scored on a schedule, not every epoch: it is a full forward
        # pass over every legal cell of the val benchmarks, which is the most
        # expensive thing in the loop.
        if val_rows and (epoch % args.val_every == 0 or epoch == args.epochs):
            n_val_evals += 1
            vm = aggregate(raw_scores(agent, val_states, val_rows))
            row.update({f"val_{k}": v for k, v in vm.items()})
            if vm["feas_auc"] == vm["feas_auc"] and vm["feas_auc"] > best["val_feas"]:
                best["val_feas"] = vm["feas_auc"]
                snaps["val_feas"] = agent.snapshot()
                best_epoch["val_feas"] = epoch
            if vm["mag_nll"] == vm["mag_nll"] and vm["mag_nll"] < best["val_mag"]:
                best["val_mag"] = vm["mag_nll"]
                snaps["val_mag"] = agent.snapshot()
                best_epoch["val_mag"] = epoch
        history.append(row)

        if args.log_every and (epoch % args.log_every == 0
                               or epoch in (1, args.epochs)):
            log.info("      epoch %4d | loss %.4f (feas %.4f mag %.4f over %d "
                     "update(s)) | sigma %.3f | eps %.3f | coverage %5.1f%% | "
                     "picks that ran %4.1f%%",
                     epoch, row["loss"], row["feas_loss"], row["mag_loss"],
                     n_upd, row["sigma"], agent.epsilon,
                     100 * row["coverage"], 100 * row["ran_rate"])

    snaps["final"] = agent.snapshot()

    # Without this, an empty val slice leaves val_feas and val_mag holding the
    # snapshot taken BEFORE epoch 1 — random initialisation. main() then
    # restores snaps[--select-by] and every headline number describes an
    # untrained network, with nothing in the output saying so. Fall back to the
    # final epoch and say it loudly.
    if n_val_evals == 0:
        for rule in ("val_feas", "val_mag"):
            snaps[rule] = snaps["final"]
        log.warning("      no validation pass ran (%d val loop(s), --val-every "
                    "%d, %d epoch(s)) — val_feas/val_mag fall back to the FINAL "
                    "epoch. Selection did not happen; read best_epoch_* = 0 as "
                    "'not selected', not 'epoch 0 was best'.",
                    len(val_loops), args.val_every, args.epochs)

    cov = len(seen) / n_legal if n_legal else float("nan")
    return agent, snaps, cov, history, best_epoch


# ---------------------------------------------------------------------------
# Few-shot adaptation
# ---------------------------------------------------------------------------

def adapt_cells(adapt_loops: list, data: dict, status_map: dict, args) -> tuple:
    """(states, rows) for the loops the target let us measure.

    Every legal cell of an adaptation loop is included. MIRROR:
    offline_adapt._fit_bandit:258-273 does the same -- "measuring a loop" means
    paying for its cells, and the few-shot cost accounting in that module is
    written against that reading.
    """
    return cell_rows(adapt_loops, data, status_map, args)


def adapt_intercept(agent, states: list, rows: list, args) -> None:
    """
    Fit a single output offset per head on the target's measured cells.

    The base network is frozen and evaluated ONCE under no_grad, then a scalar
    is optimised on top. Exact, and it makes the cost obvious: k measurements
    buy one number per head, which is the most they can support without
    variance swamping the estimate. factor_adapt's smallest setting fits 650
    parameters from ~38 cells; this fits one.
    """
    if not rows:
        return
    S = torch.stack(states)
    with torch.no_grad():
        base_logit, base_mu = agent.logits_and_mu(S)
    y = torch.tensor([float(agent.ran(st)) for _, _, st, _ in rows])
    lo = torch.tensor([float(base_logit[si, fi]) for si, fi, _, _ in rows])

    if args.adapt_what in ("feas", "both"):
        b = torch.zeros((), requires_grad=True)
        opt = torch.optim.Adam([b], lr=args.adapt_lr)
        for _ in range(args.adapt_steps):
            opt.zero_grad()
            F.binary_cross_entropy_with_logits(lo + b, y).backward()
            opt.step()
        agent.feas_bias = float(b.detach())

    if args.adapt_what in ("mag", "both"):
        ran = [(si, fi, st, v) for si, fi, st, v in rows if agent.ran(st)]
        if not ran:
            return
        mu0 = torch.tensor([float(base_mu[si, fi]) for si, fi, _, _ in ran])
        v = torch.tensor([float(x) for _, _, _, x in ran])
        cens = torch.tensor([st == CENSORED for _, _, st, _ in ran])
        sigma = agent.sigma
        b = torch.zeros((), requires_grad=True)
        opt = torch.optim.Adam([b], lr=args.adapt_lr)
        for _ in range(args.adapt_steps):
            opt.zero_grad()
            mu = mu0 + b
            z = (v - mu) / sigma
            g = 0.5 * z * z
            c = -torch.special.log_ndtr((CLIP_FLOOR - mu) / sigma)
            torch.where(cens, c, g).mean().backward()
            opt.step()
        agent.mag_bias = float(b.detach())


def adapt_finetune(agent, states: list, rows: list, args) -> None:
    """Gradient steps on the OUTPUT layer of the selected head(s).

    net[6] is the readout of a _MLP; everything below stays frozen. Offered for
    comparison against the intercept, since the prediction under test is that
    650 parameters is already too many for ~38 cells.
    """
    if not rows:
        return
    mods = []
    if args.adapt_what in ("feas", "both"):
        mods.append(agent.feas)
    if args.adapt_what in ("mag", "both"):
        mods.append(agent.mag)
    params = []
    for m in mods:
        for p in m.net[6].parameters():
            params.append(p)
    if not params:
        return
    S = torch.stack(states)
    buf = [(S[si], fi, st, v) for si, fi, st, v in rows]
    # A local optimizer over the readout only. agent.optimizer is left alone —
    # _batch_loss only does forward work, so nothing here needs it, and swapping
    # it would leave the base agent holding an optimizer over frozen tensors if
    # this raised partway through.
    opt = torch.optim.Adam(params, lr=args.adapt_lr)
    for _ in range(args.adapt_steps):
        loss, _, _ = agent._batch_loss(buf)
        opt.zero_grad()
        loss.backward()
        # Clipped like training. ~38 cells at --adapt-lr 1e-2 for 100 steps is
        # exactly the regime where one large gradient walks the readout
        # somewhere the base model never was, and the adapted number would then
        # be reporting the blow-up rather than the adaptation.
        if agent.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(params, agent.max_grad_norm)
        opt.step()


def run_adapt(agent, base_snap: dict, test_bench: list, loops: list,
              data: dict, status_map: dict, args, seed: int) -> dict:
    """
    Paired zero-shot vs adapted over the held-out benchmarks.

    A fresh restore of the base snapshot before every target: adapting the same
    agent twice in a row would carry one application's offset into the next,
    which is incremental training over the fold and not few-shot. The final
    restore is the leak guard -- everything after this call sees the base model.
    """
    from adapt_eval import split_adaptation_loops

    rng = random.Random(seed * 977 + 13)
    zero_l, adapt_l, n_ctrl = [], [], 0
    for b in test_bench:
        b_loops = [l for l in loops if l["benchmark_name"] == b]
        if not b_loops:
            continue
        a_loops, e_loops = split_adaptation_loops(b_loops, rng)
        if not e_loops:
            continue
        if not a_loops:                    # single-loop benchmark: control
            n_ctrl += 1
            continue
        agent.restore(base_snap)
        zero_l.extend(e_loops)
        a_states, a_rows = adapt_cells(a_loops, data, status_map, args)
        if args.adapt_mode == "intercept":
            adapt_intercept(agent, a_states, a_rows, args)
        else:
            adapt_finetune(agent, a_states, a_rows, args)
        adapt_l.append((e_loops, agent.snapshot()))

    if not zero_l:
        agent.restore(base_snap)
        return {"adapt_n_loops": 0, "adapt_n_targets": 0,
                "adapt_n_control": n_ctrl}

    agent.restore(base_snap)
    zero = decision_eval(agent, zero_l, data, args)
    z_states, z_rows = cell_rows(zero_l, data, status_map, args)
    zero_h = aggregate(raw_scores(agent, z_states, z_rows))

    # Scored per target under that target's OWN adapted weights, then pooled
    # over the identical loop set the zero-shot row used.
    #
    # Pooled from raw ingredients, never by averaging the per-target summaries.
    # None of these three quantities is a weighted mean of its own per-subset
    # values: capture is a ratio of SUMS (factor_only.evaluate returns
    # probe_realized / probe_oracle precisely so callers can add them), AUC over
    # a union is not any average of per-subset AUCs, and a mean NLL pools by
    # CELL count rather than by loop count. With one or two evaluation loops per
    # target, getting this wrong is not a rounding difference — most per-target
    # AUCs are undefined outright.
    ad_real = ad_orc = 0.0
    ad_raw = {"y": [], "p": [], "nll": [], "ae": []}
    for e_loops, snap in adapt_l:
        agent.restore(snap)
        d = decision_eval(agent, e_loops, data, args)
        if d["probe_oracle"] == d["probe_oracle"]:
            ad_real += d["probe_realized"]
            ad_orc += d["probe_oracle"]
        e_states, e_rows = cell_rows(e_loops, data, status_map, args)
        for k, v in raw_scores(agent, e_states, e_rows).items():
            ad_raw[k].extend(v)
    agent.restore(base_snap)
    ad = aggregate(ad_raw)

    return {
        "adapt_n_loops": len(zero_l),
        "adapt_n_targets": len(adapt_l),
        "adapt_n_control": n_ctrl,
        "zero_probe": zero["probe"], "zero_feas_auc": zero_h["feas_auc"],
        "zero_mag_nll": zero_h["mag_nll"],
        "adapt_probe": ad_real / ad_orc if ad_orc > 0 else float("nan"),
        "adapt_feas_auc": ad["feas_auc"],
        "adapt_mag_nll": ad["mag_nll"],
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _vals(rows: list, key: str) -> list:
    return [r[key] for r in rows if key in r and r[key] == r[key]]


def _mean(rows: list, key: str) -> float:
    v = _vals(rows, key)
    return fmean(v) if v else float("nan")


def _spread(rows: list, key: str) -> str:
    """'mean +- sd [min, max] over N' — the only honest form for these.

    A mean across splits is not a result on its own here: the study already
    measured a +-14.2pp swing across splits, so an effect smaller than its own
    spread is not an effect. One split reports the value with '(single split)'
    rather than a fabricated sd of 0.
    """
    v = _vals(rows, key)
    if not v:
        return "n/a"
    if len(v) < 2:
        return f"{v[0]:+.3f}  (single split)"
    return (f"{fmean(v):+.3f} +- {pstdev(v):.3f}  "
            f"[{min(v):+.3f}, {max(v):+.3f}]  over {len(v)}")


def _readout(rows: list, args) -> None:
    """
    The go / no-go criteria, stated as tests rather than left to the reader.

    Fixed in advance and written down here so the decision is not made by
    picking whichever column looks best after the fact — the discipline
    factor_rank.py sets with its own "WHAT WOULD COUNT AS SUCCESS" block.
    """
    auc = _vals(rows, "test_feas_auc")
    acc = _mean(rows, "test_feas_acc")
    base = _mean(rows, "test_feas_base")
    cap = _vals(rows, "test_probe")
    bar = _vals(rows, "test_probe_bar")
    rules = [_mean(rows, f"rule_{r}_probe") for r in SELECT_RULES]
    rules = [x for x in rules if x == x]

    log.info("\n" + "=" * 78)
    log.info("  READ-OUT — criteria fixed before the run")
    log.info("=" * 78)

    # 1. Is there ANY feasibility signal on unseen applications?
    if not auc:
        log.info("  1. FEASIBILITY TRANSFER : no test AUC — nothing to judge.")
    elif min(auc) > 0.5:
        log.info("  1. FEASIBILITY TRANSFER : YES. Test AUC %s, and the WORST "
                 "split\n     is above chance. Predicting what will not build "
                 "transfers across\n     applications — a signal the scalar "
                 "agent never separated out.", _spread(rows, "test_feas_auc"))
    elif fmean(auc) > 0.5:
        log.info("  1. FEASIBILITY TRANSFER : MARGINAL. Test AUC %s. The mean "
                 "clears chance\n     but at least one split does not, so the "
                 "effect is inside its own spread.", _spread(rows, "test_feas_auc"))
    else:
        log.info("  1. FEASIBILITY TRANSFER : NO. Test AUC %s is at or below "
                 "chance.\n     The features do not predict build failure on "
                 "unseen applications either.", _spread(rows, "test_feas_auc"))

    # 2. Does it beat the trivial predictor? AUC can clear chance while the
    #    thresholded decision still loses to "always predict the majority".
    if acc == acc and base == base:
        verdict = "YES" if acc > base else "NO"
        log.info("  2. BEATS CONSTANT      : %s. accuracy %.3f vs base rate "
                 "%.3f.\n     A constant predictor scores the base rate; below "
                 "it the head is worse\n     than declaring every cell the "
                 "majority class.", verdict, acc, base)

    # 3. The magnitude question — the one the scalar agent already failed.
    if cap and bar:
        d = fmean(cap) - fmean(bar)
        log.info("  3. MAGNITUDE vs BAR    : %s. capture %s\n     best constant "
                 "factor %s\n     difference %+.1fpp — compare against the "
                 "split spread above, not against 0.",
                 "ABOVE" if d > 0 else "AT OR BELOW",
                 _spread(rows, "test_probe"), _spread(rows, "test_probe_bar"),
                 100 * d)

    # 4. Did the epoch matter more than the model?
    if len(rules) > 1:
        rng = max(rules) - min(rules)
        log.info("  4. SELECTION STABILITY : rule spread %.1fpp across "
                 "final/val_feas/val_mag.\n     Large relative to the split "
                 "spread means the checkpoint, not the\n     method, is "
                 "producing the number.", 100 * rng)

    # 5. Did training actually happen? A good-looking metric on 20% coverage is
    #    a statement about 20% of the table.
    log.info("  5. TRAINING SANITY     : coverage %.1f%% of legal cells, "
             "picks that ran %.1f%%.\n     Low coverage makes every row above a "
             "claim about a fraction of the\n     table; a collapsing ran-rate "
             "means the magnitude head starved.",
             100 * _mean(rows, "coverage"), 100 * _mean(rows, "train_ran_rate"))

    log.info("\n  GO if 1 is YES and 5 is healthy — feasibility is a deliverable "
             "on its own\n  (fewer wasted compiles) whatever 3 says. NO-GO if 1 "
             "is NO: the hurdle\n  split bought nothing the scalar model was not "
             "already failing to find,\n  and the remaining explanation is the "
             "features or the measurement floor.")


def report(rows: list, args) -> None:
    if not rows:
        log.error("no split produced a result")
        return
    log.info("\n" + "=" * 78)
    log.info("  HURDLE UNROLLER — feasibility and magnitude, never averaged")
    log.info("=" * 78)
    log.info("  mean cell coverage %.1f%% | floor-as=%s | %d split(s)\n",
             100 * _mean(rows, "coverage"), args.floor_as, len(rows))

    log.info("  FEASIBILITY — will this cell build and run?")
    log.info("    %-8s %8s %8s %8s %8s", "split", "AUC", "acc", "base", "brier")
    for side in ("fit", "val", "test"):
        log.info("    %-8s %8.3f %8.3f %8.3f %8.3f", side,
                 _mean(rows, f"{side}_feas_auc"), _mean(rows, f"{side}_feas_acc"),
                 _mean(rows, f"{side}_feas_base"),
                 _mean(rows, f"{side}_feas_brier"))
    log.info("    AUC 0.5 is chance. Compare `acc` against `base`, not against "
             "0.5 —\n    a constant predictor already scores `base`.\n")

    log.info("  MAGNITUDE — how fast, given it ran (censored NLL, lower better)")
    log.info("    %-8s %8s %8s %8s", "split", "NLL", "MAE", "n")
    for side in ("fit", "val", "test"):
        log.info("    %-8s %8.3f %8.3f %8.0f", side,
                 _mean(rows, f"{side}_mag_nll"), _mean(rows, f"{side}_mag_mae"),
                 _mean(rows, f"{side}_mag_n"))
    log.info("")

    log.info("  DECISION — probe capture on the true branch, vs the best "
             "constant factor")
    log.info("    %-10s %10s %10s %8s", "split", "learned", "bar", "n")
    for side in ("fit", "test"):
        log.info("    %-10s %9.1f%% %9.1f%% %8.0f", side,
                 100 * _mean(rows, f"{side}_probe"),
                 100 * _mean(rows, f"{side}_probe_bar"),
                 _mean(rows, f"{side}_probe_n"))
    log.info("    failed picks %.0f | at clip floor %.0f",
             _mean(rows, "test_probe_failed"), _mean(rows, "test_probe_clipped"))

    log.info("\n  SELECTION RULE SPREAD on test (probe capture)")
    for rule in SELECT_RULES:
        log.info("    %-10s %6.1f%%   feas AUC %.3f",
                 rule, 100 * _mean(rows, f"rule_{rule}_probe"),
                 _mean(rows, f"rule_{rule}_feas_auc"))
    log.info("    A large spread here means the epoch mattered more than the "
             "model did.")

    if args.adapt and any("adapt_probe" in r for r in rows):
        log.info("\n  FEW-SHOT (%s / %s, %d steps) — paired on identical loops",
                 args.adapt_what, args.adapt_mode, args.adapt_steps)
        log.info("    %-10s %10s %10s %10s", "", "probe", "feas AUC", "mag NLL")
        log.info("    %-10s %9.1f%% %10.3f %10.3f", "zero-shot",
                 100 * _mean(rows, "zero_probe"), _mean(rows, "zero_feas_auc"),
                 _mean(rows, "zero_mag_nll"))
        log.info("    %-10s %9.1f%% %10.3f %10.3f", "adapted",
                 100 * _mean(rows, "adapt_probe"), _mean(rows, "adapt_feas_auc"),
                 _mean(rows, "adapt_mag_nll"))
        # %.0f, not %d: _mean returns NaN when a key is missing from every row,
        # and "%d" % nan raises inside logging — the line is then lost behind a
        # "--- Logging error ---" traceback instead of printing "nan".
        log.info("    %.0f target(s), %.0f eval loops, %.0f single-loop "
                 "controls excluded", _mean(rows, "adapt_n_targets"),
                 _mean(rows, "adapt_n_loops"), _mean(rows, "adapt_n_control"))
        log.info("    Prediction under test: few-shot lifts FEASIBILITY and not "
                 "magnitude.\n    If so it explains factor_adapt ~ factor_only "
                 "rather than leaving it open.")

    log.info("\n  Adaptation is NOT deployable-without-measurement: every "
             "adapted number\n  costs a compile+run per adaptation loop. Never "
             "tabulate it beside zero-shot\n  as if it were a peer.")

    # Spread, not just the mean. The headline three carry their own variability
    # because an effect smaller than the split spread is not an effect.
    log.info("\n  ACROSS SPLITS (mean +- sd [min, max])")
    for label, key in (("test feas AUC", "test_feas_auc"),
                       ("test probe cap", "test_probe"),
                       ("  its constant bar", "test_probe_bar"),
                       ("test mag NLL", "test_mag_nll"),
                       ("train loss (final)", "train_loss_final")):
        log.info("    %-19s %s", label, _spread(rows, key))

    _readout(rows, args)


# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Exposed rather than built inline in main() so the tests construct their
    args from THESE defaults. A hand-written Namespace in the test file drifts
    the moment a flag is added here, and it drifts silently toward whatever the
    old default was — the trap test_offline._args exists to avoid."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--deadzone", type=float, required=True)
    p.add_argument("--labels", type=Path, default=None)
    p.add_argument("--status", type=Path, default=None,
                   help="cell_status.json sidecar. Default RUN_DIR/cell_status.json; "
                        "derived in memory if absent.")

    g = p.add_argument_group("splits")
    g.add_argument("--splits", type=int, default=5)
    g.add_argument("--split-seed", type=int, default=0)
    g.add_argument("--val-frac", type=float, default=0.15)
    g.add_argument("--test-frac", type=float, default=0.2)
    g.add_argument("--base-seed", type=int, default=100)
    g.add_argument("--seeds", type=int, default=1)

    g = p.add_argument_group("model — defaults MIRROR factor_only")
    g.add_argument("--epochs", type=int, default=300)
    g.add_argument("--lr", type=float, default=3e-4)
    g.add_argument("--weight-decay", type=float, default=0.01)
    g.add_argument("--max-grad-norm", type=float, default=0.5)
    g.add_argument("--batch-size", type=int, default=8)
    g.add_argument("--buffer-size", type=int, default=128)
    g.add_argument("--K", type=int, default=2)
    g.add_argument("--epsilon", type=float, default=0.3)
    g.add_argument("--epsilon-final", type=float, default=0.05)
    g.add_argument("--mag-coef", type=float, default=1.0,
                   help="Weight on the magnitude term. The two terms are on "
                        "different scales (BCE in nats per cell vs a Gaussian "
                        "NLL), so this is the one knob that trades them off.")
    g.add_argument("--floor-as", choices=("censored", "failed"),
                   default="censored",
                   help="The 744 cells at the -1.0 clip. 'censored' reads them "
                        "as ran-and-at-least-this-bad; 'failed' reads them as "
                        "timeouts that shipped nothing. Not separable from the "
                        "cache, so both are reachable.")
    g.add_argument("--absent-as", choices=("failed", "ok"), default="failed")
    g.add_argument("--missing", type=float, default=-0.161,
                   help="Value for an absent cell. Only reaches the magnitude "
                        "head when --absent-as ok, and is passed through to "
                        "factor_only.evaluate for comparability.")

    g = p.add_argument_group("few-shot")
    g.add_argument("--adapt", action="store_true")
    g.add_argument("--adapt-what", choices=("feas", "mag", "both"),
                   default="feas")
    g.add_argument("--adapt-mode", choices=("intercept", "finetune"),
                   default="intercept")
    g.add_argument("--adapt-lr", type=float, default=1e-2)
    g.add_argument("--adapt-steps", type=int, default=100)

    p.add_argument("--val-every", type=int, default=10)
    p.add_argument("--select-by", choices=SELECT_RULES, default="val_feas")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--csv-out", type=str, default=None)
    p.add_argument("--curve-out", type=str, default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()

    torch.set_num_threads(max(1, args.threads))
    data = load_run(args.run_dir, args.deadzone, args.labels)
    loops = labelled_loops(data)

    sidecar = args.status or (args.run_dir / "cell_status.json")
    if sidecar.exists():
        status_map = load_status(sidecar)
        log.info("Status sidecar: %s (%d non-ok cells)", sidecar, len(status_map))
    else:
        rc = json.loads((args.run_dir / "reward_cache.json").read_text())
        mig = rc.get("migration") or {}
        status_map, counts, diverge = derive(
            rc.get("rewards", {}),
            set(mig.get("failure_keys", [])) if mig else None)
        if diverge:
            log.warning("failure_keys and the value test DISAGREE on %d cell(s) "
                        "— taking the union. Run cell_status.py for the detail.",
                        sum(len(v) for v in diverge.values()))
        log.info("No sidecar at %s — derived in memory: %d failed, %d censored",
                 sidecar, counts[FAILED], counts[CENSORED])

    log.info("Loaded %d loops / %d benchmarks | %d labelled | %d states\n",
             len(data["loops"]), len(data["benchmarks"]),
             data["n_labelled_loops"], 2 * len(loops))

    rows, curve = [], []
    for sp in range(args.splits):
        sseed = args.split_seed + sp
        tr_b, va_b, te_b = random_split3(data["benchmarks"], args.val_frac,
                                         args.test_frac, sseed)
        fit_l = loops_for(loops, tr_b)
        val_l = loops_for(loops, va_b)
        test_l = loops_for(loops, te_b)
        if not fit_l or not test_l:
            log.warning("  split %d: empty side (%d fit / %d val / %d test) — "
                        "skipped", sp + 1, len(fit_l), len(val_l), len(test_l))
            continue
        for si in range(args.seeds):
            seed = args.base_seed + si
            agent, snaps, cov, hist, bep = train(
                fit_l, val_l, data, status_map, args, seed)

            row = {"split": sp + 1, "split_seed": sseed, "seed": seed,
                   "coverage": cov, "sigma": agent.sigma,
                   "n_fit_loops": len(fit_l), "n_val_loops": len(val_l),
                   "n_test_loops": len(test_l), "n_test_benchmarks": len(te_b),
                   "best_epoch_feas": bep["val_feas"],
                   "best_epoch_mag": bep["val_mag"],
                   # FINAL epoch, not the mean: the concern is collapse, and a
                   # mean over 300 epochs hides a rate that fell to zero at 200.
                   "train_ran_rate": hist[-1]["ran_rate"] if hist else float("nan"),
                   "train_loss_first": hist[0]["loss"] if hist else float("nan"),
                   "train_loss_final": hist[-1]["loss"] if hist else float("nan")}

            # Built once per side and reused across every rule below. cell_rows
            # re-walks each loop and re-normalises each state, and it is
            # invariant to the weights — recomputing it per rule would repeat
            # that work four times over for an identical answer.
            cells_by_side = {
                side: cell_rows(ls, data, status_map, args)
                for side, ls in (("fit", fit_l), ("val", val_l),
                                 ("test", test_l)) if ls}

            # Every rule scored on test, so the spread between them is on the
            # page. Reporting only the selected one would hide exactly the
            # instability that motivated moving off mean_realized.
            t_states, t_rows = cells_by_side["test"]
            for rule in SELECT_RULES:
                agent.restore(snaps[rule])
                row[f"rule_{rule}_probe"] = decision_eval(
                    agent, test_l, data, args)["probe"]
                row[f"rule_{rule}_feas_auc"] = aggregate(
                    raw_scores(agent, t_states, t_rows))["feas_auc"]

            agent.restore(snaps[args.select_by])
            for side, (s_states, s_rows) in cells_by_side.items():
                row.update({f"{side}_{k}": v for k, v in
                            aggregate(raw_scores(agent, s_states,
                                                 s_rows)).items()})
            for side, ls in (("fit", fit_l), ("test", test_l)):
                row.update({f"{side}_{k}": v for k, v in
                            decision_eval(agent, ls, data, args).items()})

            if args.adapt:
                row.update(run_adapt(agent, snaps[args.select_by], te_b, loops,
                                     data, status_map, args, seed))

            rows.append(row)
            curve.extend(dict(split=sp + 1, seed=seed, **ep) for ep in hist)
            log.info("  split %d/%d seed %d | %3d fit / %3d val / %3d test | "
                     "cov %5.1f%% | feas AUC %.3f | probe %5.1f%% (bar %5.1f%%)",
                     sp + 1, args.splits, seed, len(fit_l), len(val_l),
                     len(test_l), 100 * cov, row.get("test_feas_auc", float("nan")),
                     100 * row.get("test_probe", float("nan")),
                     100 * row.get("test_probe_bar", float("nan")))

    report(rows, args)
    if args.csv_out and rows:
        write_csv(Path(args.csv_out), rows)
        log.info("\n  per-split rows: %s", args.csv_out)
    if args.curve_out and curve:
        write_csv(Path(args.curve_out), curve)
        log.info("  training curve: %s", args.curve_out)


if __name__ == "__main__":
    main()
