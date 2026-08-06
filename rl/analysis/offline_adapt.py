"""
Few-shot adaptation, offline: measure a couple of a target's loops, then predict
the rest of that target.

WHAT QUESTION THIS ANSWERS
--------------------------
Zero-shot CV asks "does a function over loop features transfer to an unseen
application". The answer so far is no: fit is comfortably positive while test is
negative. That is the PREMISE for this experiment, not an argument against it —
if zero-shot worked there would be nothing to adapt.

Few-shot asks a different question: given a handful of MEASURED loops from the
target itself, can a few gradient steps recover the rest of that target?

There is direct evidence the headroom is real. The benchmark-dominant reference
— "predict this benchmark's most common category for every loop in it" — scores
~81% accuracy and ~89% capture against the oracle, because benefit is
all-or-nothing per application (100/138 benchmarks uniform; 43/81 among
multi-loop ones). That reference is effectively this experiment's ceiling, and
it sits far above anything zero-shot achieved.

WHAT IT COSTS
-------------
Adaptation requires MEASURING loops of the target: compile + run, per loop. So
the adapted policy is NOT deployable-without-measurement and must never be
tabulated beside always-no-op or the zero-shot policy as if it were a peer. It
is its own category: "deployable given k measurements of the target".

PROTOCOL
--------
Per fold, per HELD-OUT benchmark, using that fold's zero-shot agent:
    >=3 loops -> adapt on 2, evaluate on the rest
       2 loops -> adapt on 1, evaluate on 1
       1 loop  -> adapt on 0; the loop is a CONTROL, scored zero-shot
The split is at LOOP granularity within one benchmark. Adapting on some cells of
a loop and testing on other cells of the SAME loop would measure action-space
interpolation, which is a far weaker claim.

Every benchmark is scored zero-shot AND adapted on the SAME evaluation loops, so
the comparison is paired: the difference is adaptation, not loop composition.
The single-loop benchmarks are not waste — they are the control group, run
through the identical machinery with k=0.

Usage:
  python3 offline_adapt.py RUN_DIR --agent bandit --deadzone 0.005
"""

import argparse
import copy
import csv
import logging
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import torch                                                     # noqa: E402

from offline_data import (grouped_kfold, holdout_split,          # noqa: E402
                          labelled_loops, load_run, loops_for,
                          oracle_of_gated, score_decisions, table_header,
                          table_row)
from offline_train import (_mr, build_parser, greedy_picks,      # noqa: E402
                           train_agent)

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
log = logging.getLogger("adapt")
logging.getLogger("agent").setLevel(logging.WARNING)


def adapt_in_place(agent, loops: list, data: dict, kind: str, lr: float,
                   steps: int):
    """
    Fine-tune `agent` on the measured cells of `loops`. Mutates and returns it.

    MIRROR: adapt_eval.adapt, with ONE deliberate change — the PPO
    behaviour-cloning target uses the DEADZONE-GATED oracle. adapt_eval.py:302
    calls the raw oracle_of, which on a loop whose best transform is +0.001
    against a 0.005 deadzone clones toward that transform while label_loops
    calls the loop a no-op. The policy would be taught to fire exactly where
    declining is correct, and then scored against labels that say so.
    (adapt_eval.py is left untouched; it is the pinned checkpoint-driven study.)

    freeze_trunk is imported rather than reimplemented: two adaptation loops
    give ~20-40 examples against ~60k parameters, so only each actor head's
    final projection is trainable. The critic stays frozen — it plays no part in
    greedy selection.

    The signal matches each agent's native one, so both consume the same cells:
      ppo    — cross-entropy toward the gated oracle-best action.
      bandit — MSE of Q(s,a) against the measured reward over every cell.
    """
    import torch.nn.functional as F

    from adapt_eval import freeze_trunk, make_state
    from agent import (FACTOR_VALUES, _IDX_TRIP_COUNT, _IDX_TRIP_COUNT_KNOWN,
                       build_factor_mask)

    if hasattr(agent, "select_category"):
        # The sample builder below stores the unmerge BIT as the first-head
        # action. A 3-way head would read that as a category index, training
        # "unmerge=1" as "unroll_only". Refuse rather than fine-tune on
        # mislabelled targets.
        raise NotImplementedError(
            "few-shot adaptation is not wired for the 3-way category head — "
            "the cloning target uses the 2-head (unmerge, factor) encoding")

    trainable = freeze_trunk(agent)
    if not loops or not trainable:
        return agent

    if kind == "ppo":
        _fit_ppo(agent, loops, data, trainable, lr, steps)
    else:
        _fit_bandit(agent, loops, data, trainable, lr, steps)
    return agent


def _fit_ppo(agent, loops, data, trainable, lr, steps) -> None:
    """Behaviour cloning toward the DEADZONE-GATED oracle action."""
    import torch.nn.functional as F

    from adapt_eval import make_state
    from agent import (FACTOR_VALUES, _IDX_TRIP_COUNT, _IDX_TRIP_COUNT_KNOWN,
                       build_factor_mask)

    rows = []
    for l in loops:
        table = data["tables"][(l["benchmark_name"], l["loop_idx"])]
        raw = l["pre_features_raw"]
        (u, f), _ = oracle_of_gated(table, data["deadzone"])
        s1, s2 = make_state(l, data["normalizer"], data["postf"], u)
        rows.append((s1, s2, u, FACTOR_VALUES.index(f),
                     raw[_IDX_TRIP_COUNT_KNOWN] > 0.5, int(raw[_IDX_TRIP_COUNT])))
    if not rows:
        return
    s1b = torch.stack([x[0] for x in rows])
    s2b = torch.stack([x[1] for x in rows])
    a1b = torch.tensor([x[2] for x in rows], dtype=torch.long)
    a2b = torch.tensor([x[3] for x in rows], dtype=torch.long)
    # The same trip-count mask the policy uses at selection time; without it,
    # cloning spends probability mass on factors masked out at evaluation.
    m2b = torch.stack([build_factor_mask(x[4], x[5]) for x in rows])

    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
    for _ in range(steps):
        opt.zero_grad()
        f_logits = agent.factor_actor.forward(s2b).masked_fill(
            ~m2b, float("-inf"))
        loss = (F.cross_entropy(agent.unmerge_actor.forward(s1b), a1b)
                + F.cross_entropy(f_logits, a2b))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 0.5)
        opt.step()


def _fit_bandit(agent, loops, data, trainable, lr, steps) -> None:
    """
    Q-regression, with Q1's target the MAX over the branch's valid cells.

    adapt_eval.adapt regresses Q1 against every cell on the branch, so it
    converges to the branch MEAN. agent.BanditAgent.ppo_update deliberately does
    the opposite — a max backup — and its docstring calls the mean target "the
    extinction bug, in the target itself": the unmerge branch has the fattest
    negative tail, so its mean sinks below unroll and argmax_u stops choosing
    it. Fine-tuning toward the mean pulls the policy at exactly the target
    training was fixed to avoid.

    Training must BOOTSTRAP that max through Q2, having no table online.
    Adaptation has one — the adaptation loops were measured, which is the whole
    premise — so the max is taken over MEASURED rewards: same quantity, without
    the max-of-noise inflation.
    """
    import torch.nn.functional as F

    from adapt_eval import make_state
    from agent import (FACTOR_VALUES, _IDX_TRIP_COUNT, _IDX_TRIP_COUNT_KNOWN,
                       build_factor_mask)

    q1_rows, q2_rows = [], []
    for l in loops:
        table = data["tables"][(l["benchmark_name"], l["loop_idx"])]
        raw = l["pre_features_raw"]
        valid = build_factor_mask(raw[_IDX_TRIP_COUNT_KNOWN] > 0.5,
                                  int(raw[_IDX_TRIP_COUNT]))
        for u in (0, 1):
            cells = [(f, r) for (uu, f), r in table.items()
                     if uu == u and f in FACTOR_VALUES
                     and bool(valid[FACTOR_VALUES.index(f)])]
            if not cells:
                continue
            s1, s2 = make_state(l, data["normalizer"], data["postf"], u)
            # u=0 includes the free no-op at 0.0 (build_tables injects it), so
            # Q1(s,0) is "decline or unroll, whichever is better" — exactly what
            # argmax_u compares at inference.
            q1_rows.append((s1, u, max(r for _, r in cells)))
            for f, r in cells:
                q2_rows.append((s2, FACTOR_VALUES.index(f), r))
    if not q1_rows:
        return

    s1b = torch.stack([x[0] for x in q1_rows])
    u_b = torch.tensor([x[1] for x in q1_rows], dtype=torch.long)
    t1b = torch.tensor([x[2] for x in q1_rows], dtype=torch.float32)
    s2b = torch.stack([x[0] for x in q2_rows])
    f_b = torch.tensor([x[1] for x in q2_rows], dtype=torch.long)
    r_b = torch.tensor([x[2] for x in q2_rows], dtype=torch.float32)

    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
    for _ in range(steps):
        opt.zero_grad()
        q1 = agent.unmerge_actor.forward(s1b).gather(
            1, u_b.unsqueeze(1)).squeeze(1)
        q2 = agent.factor_actor.forward(s2b).gather(
            1, f_b.unsqueeze(1)).squeeze(1)
        loss = F.mse_loss(q1, t1b) + F.mse_loss(q2, r_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 0.5)
        opt.step()


def split_for_adaptation(loops: list, rng: random.Random) -> tuple:
    """
    (adaptation_loops, evaluation_loops) for ONE benchmark.

    MIRROR: adapt_eval.split_adaptation_loops — >=3 loops give 2 adaptation
    loops, 2 give 1, and a single-loop benchmark gives 0 and becomes a control.
    """
    from adapt_eval import split_adaptation_loops
    return split_adaptation_loops(loops, rng)


def run(data: dict, args) -> None:
    loops = labelled_loops(data)
    if args.min_fill > 0:
        keep = {(l["benchmark_name"], l["loop_idx"]) for l in loops
                if data["fill"].get((l["benchmark_name"], l["loop_idx"]), 1.0)
                >= args.min_fill}
        dropped = len(loops) - len(keep)
        loops = [l for l in loops if (l["benchmark_name"], l["loop_idx"]) in keep]
        log.info("--min-fill %.2f dropped %d loop(s); a partially measured "
                 "adaptation loop teaches the model a table it does not have",
                 args.min_fill, dropped)
    benches = sorted({l["benchmark_name"] for l in loops})
    folds = grouped_kfold(benches, args.folds, args.fold_seed)

    log.info("Few-shot adaptation | %d-fold x %d init seed(s) | agent=%s | "
             "adapt lr=%g steps=%d", args.folds, args.seeds, args.agent,
             args.adapt_lr, args.adapt_steps)
    log.info("Adaptation loops per benchmark: >=3 loops -> 2, 2 -> 1, 1 -> 0 "
             "(control)")
    log.info("Each row below is one fold x seed: %d epochs of training, then "
             "one\nadaptation per held-out benchmark. The first row takes as "
             "long as a full\ntraining run — nothing is wrong if it sits there "
             "for a while.\n", args.epochs)
    log.info("               benchmarks | eval  |   ZERO-SHOT   |   FEW-SHOT    "
             "|   DELTA")
    log.info("  fold  seed |  k>0   k=0 | loops |    acc    mean |    acc    "
             "mean |    acc    mean")
    log.info("  " + "-" * 84)

    # Keyed BY SEED, not pooled. Pooling all seeds reports 3x the population as
    # one number with no spread — and the whole quantity of interest here is a
    # DELTA, which is uninterpretable without knowing the seed noise it has to
    # clear.
    zero: dict = {}
    adapt: dict = {}
    ctrl: dict = {}
    rows, leaks = [], 0

    def _sc(picks):
        return score_decisions(picks, data["tables"], data["labels"],
                               args.deadzone, _mr(args))

    for k, (tr_b, te_b) in enumerate(folds):
        fit_b, hold_b = holdout_split(tr_b, args.holdout_frac, args.fold_seed + k)
        fit_l, hold_l = loops_for(loops, fit_b), loops_for(loops, hold_b)
        for si in range(args.seeds):
            seed = args.base_seed + si
            base_agent, _ = train_agent(args.agent, fit_l, hold_l, data, args,
                                        seed)
            rng = random.Random(seed * 1000 + k)
            first = None
            fz, fa, n_k0 = [], [], 0
            for bench in te_b:
                b_loops = [l for l in loops if l["benchmark_name"] == bench]
                if not b_loops:
                    continue
                adapt_l, eval_l = split_for_adaptation(b_loops, rng)
                if not eval_l:
                    continue
                # Zero-shot on the SAME evaluation loops — paired, so the
                # difference is adaptation and not loop composition.
                zs = greedy_picks(base_agent, eval_l, data["normalizer"],
                                  data["postf"])
                if first is None:
                    first = (eval_l, zs)
                # A FRESH copy per benchmark: adapting the shared agent would
                # carry one target's fine-tuning into the next, which is not
                # few-shot adaptation but incremental training over the fold.
                tuned = copy.deepcopy(base_agent)
                if adapt_l:
                    adapt_in_place(tuned, adapt_l, data, args.agent,
                                   args.adapt_lr, args.adapt_steps)
                ad = greedy_picks(tuned, eval_l, data["normalizer"],
                                  data["postf"])
                if adapt_l:
                    zero.setdefault(seed, []).extend(zs)
                    adapt.setdefault(seed, []).extend(ad)
                    fz.extend(zs); fa.extend(ad)
                else:
                    ctrl.setdefault(seed, []).extend(zs)
                    n_k0 += 1
                mz, ma = _sc(zs), _sc(ad)
                rows.append({
                    "fold": k + 1, "seed": seed, "benchmark": bench,
                    "n_loops": len(b_loops), "n_adapt": len(adapt_l),
                    "n_eval": len(eval_l),
                    # Per-benchmark outcome. Without these the CSV cannot answer
                    # "which applications did adaptation help", which is the
                    # interesting question when benefit is bimodal per app.
                    "zero_accuracy": round(mz["accuracy"], 6),
                    "adapt_accuracy": round(ma["accuracy"], 6),
                    "zero_mean_realized": round(mz["mean_realized"], 6),
                    "adapt_mean_realized": round(ma["mean_realized"], 6),
                })
            # REAL leak check: re-score the first benchmark with the shared
            # agent AFTER every adaptation in this fold. Comparing the control
            # against itself (as the first version did) can never fail, because
            # control entries are never adapted.
            if first is not None:
                again = greedy_picks(base_agent, first[0], data["normalizer"],
                                     data["postf"])
                if again != first[1]:
                    leaks += 1

            # Progress, one line per fold x seed. Without it the whole run is
            # silent from the banner to the final table — 15 trainings and ~400
            # adaptations with no indication it is alive.
            if fz:
                mz, ma = _sc(fz), _sc(fa)
                log.info("  %2d/%-2d %4d | %4d  %4d | %5d | %6.1f%% %+7.4f | "
                         "%6.1f%% %+7.4f | %+5.1fpp %+7.4f",
                         k + 1, args.folds, seed, len(te_b) - n_k0, n_k0,
                         len(fz), 100 * mz["accuracy"], mz["mean_realized"],
                         100 * ma["accuracy"], ma["mean_realized"],
                         100 * (ma["accuracy"] - mz["accuracy"]),
                         ma["mean_realized"] - mz["mean_realized"])
            else:
                log.info("  %2d/%-2d %4d | %4d  %4d | %5d | (no benchmark in "
                         "this fold had >1 loop)",
                         k + 1, args.folds, seed, 0, n_k0, 0)

    if not rows:
        log.error("no benchmark produced an evaluation set — nothing to report")
        return

    log.info("=" * 92)
    log.info("  ZERO-SHOT vs FEW-SHOT, on identical evaluation loops")
    log.info("=" * 92)
    log.info(table_header())
    deltas_acc, deltas_mean = [], []
    for seed in sorted(zero):
        z, a = _sc(zero[seed]), _sc(adapt[seed])
        log.info(table_row(f"seed {seed} zero-shot", z))
        log.info(table_row(f"seed {seed} + {args.adapt_steps} adapt steps", a))
        deltas_acc.append(a["accuracy"] - z["accuracy"])
        deltas_mean.append(a["mean_realized"] - z["mean_realized"])
    if ctrl:
        log.info("")
        log.info("  CONTROL — single-loop benchmarks, 0 adaptation loops "
                 "(scored zero-shot)")
        log.info(table_row("control (k=0)", _sc(sum(ctrl.values(), []))))

    def _spread(vals, label, pct):
        f = (lambda v: f"{100 * v:+.1f}pp") if pct else (lambda v: f"{v:+.4f}")
        if len(vals) < 2:
            log.info("    %-22s %s   (single seed)", label, f(vals[0]))
        else:
            log.info("    %-22s %s +- %s   [%s, %s]", label, f(st.mean(vals)),
                     f(st.stdev(vals)).lstrip("+"), f(min(vals)), f(max(vals)))

    log.info("\n  DELTA FROM ADAPTATION, across %d init seed(s)", len(deltas_acc))
    _spread(deltas_acc, "accuracy", True)
    _spread(deltas_mean, "mean realized", False)
    log.info("    A delta smaller than its own spread is not a result.")

    log.info("\n  state leakage between benchmarks: %s",
             "none" if leaks == 0 else f"{leaks} FOLD(S) LEAKED — results void")

    log.info("\n  NOT deployable-without-measurement: every adapted number "
             "costs a compile+run\n  per adaptation loop. Report it as its own "
             "category, never beside always-no-op\n  or the zero-shot policy "
             "as if it were a peer.")

    if args.csv_out:
        with open(args.csv_out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        log.info("\n  per-benchmark: %s", args.csv_out)


def main() -> None:
    p = build_parser()
    g = p.add_argument_group("few-shot adaptation")
    g.add_argument("--adapt-lr", type=float, default=1e-3)
    g.add_argument("--adapt-steps", type=int, default=50)
    g.add_argument("--min-fill", type=float, default=0.0,
                   help="Drop loops whose measured-cell fraction is below this. "
                        "An adaptation loop with partial coverage teaches the "
                        "model from a table that does not exist; the population "
                        "mean fill is 0.992, so 0.95 costs almost nothing.")
    args = p.parse_args()
    if args.agent.startswith("category"):
        # adapt_in_place raises for the 3-way head, but only after a whole fold
        # of training has already been spent.
        p.error("few-shot adaptation is not wired for the 3-way category head: "
                "the cloning target uses the 2-head (unmerge, factor) encoding")
    torch.set_num_threads(max(1, args.threads))

    data = load_run(args.run_dir, args.deadzone, args.labels)
    # adapt_in_place needs the deadzone to gate its cloning target; carrying it
    # on `data` keeps every call site from having to remember to pass it.
    data["deadzone"] = args.deadzone
    log.info("Loaded %d loops / %d benchmarks | %d labelled\n",
             len(data["loops"]), len(data["benchmarks"]),
             data["n_labelled_loops"])
    run(data, args)


if __name__ == "__main__":
    main()
