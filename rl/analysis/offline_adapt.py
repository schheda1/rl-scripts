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

from offline_data import (always_noop_picks, fingerprint,        # noqa: E402
                          grouped_kfold, holdout_split, labelled_loops,
                          load_run, loops_for, oracle_of_gated, oracle_picks,
                          score_decisions, table_header, table_row)
from offline_train import (_mr, build_parser, greedy_picks,      # noqa: E402
                           train_agent, warn_ignored_flags, write_csv)

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
log = logging.getLogger("adapt")
logging.getLogger("agent").setLevel(logging.WARNING)


def _unfreeze_spec(text: str) -> tuple:
    """`--adapt-unfreeze N` -> (N, N); `--adapt-unfreeze CAT,FACTOR` -> (CAT, FACTOR)."""
    parts = [p.strip() for p in str(text).split(",")]
    if len(parts) == 1:
        parts *= 2
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected N or CAT,FACTOR")
    try:
        vals = tuple(int(p) for p in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not integers: {text!r}")
    for v in vals:
        if v not in (1, 2, 3):
            raise argparse.ArgumentTypeError(f"layers must be 1, 2 or 3 (got {v})")
    return vals


def _freeze_for_adaptation(agent, n_cat: int, n_fac: int) -> list:
    """
    Make the last `n_cat` parameterised layers of the FIRST head and the last
    `n_fac` of the FACTOR head trainable, freeze the rest, and return the
    trainable parameters.

    The two are separate because the diagnosis says they are not symmetric: the
    category is what adaptation demonstrably learns (accuracy +21pp, unmerge
    recall +27pp) while the factor is what stays stuck (mean at or below zero
    every time). Giving the factor head more capacity than the category head
    targets the arm that is actually failing, instead of loosening both and
    inviting the category head to memorise the 20-40 adaptation examples.

    Local rather than adapt_eval.freeze_trunk, which hardcodes net[6]; that file
    stays untouched. The _MLP stack is
        0 Linear(93,128)  1 LayerNorm  2 ReLU
        3 Linear(128,64)  4 LayerNorm  5 ReLU
        6 Linear(64,out)
    so "layers" counts LINEARs and each one's LayerNorm travels with it. ReLU
    holds no parameters.

        n=1  -> net[6]                     780 / 845 params across both heads
        n=2  -> net[3], net[4], net[6]     ~17.6k
        n=3  -> the whole stack            ~42k
    (the first pair is 2-head / 3-way: the first head has 2 vs 3 outputs.)

    The tradeoff is sharp and worth watching: two adaptation loops supply
    roughly 20-40 examples. At n=1 that is ~800 parameters and the fit is
    constrained; at n=2 it is ~17.6k and the head can memorise the adaptation
    loops outright. If accuracy on the ADAPTATION loops climbs while the
    evaluation loops do not follow, that is what happened.

    The critic is never trained — it is absent from every adaptation loss and
    from the optimizer. (Its requires_grad is left alone rather than cleared;
    nothing reads it, and saying "frozen" would overstate what is done.)
    """
    groups = {1: [6], 2: [3, 4, 6], 3: [0, 1, 3, 4, 6]}
    for n in (n_cat, n_fac):
        if n not in groups:
            raise ValueError(f"--adapt-unfreeze layers must be 1, 2 or 3 (got {n})")
    idx = sorted(set(groups[n_cat]) | set(groups[n_fac]))
    # The indices above are a hardcoded read of agent._MLP's Sequential. If that
    # stack is ever reordered, this would silently fine-tune the wrong layers
    # and every adaptation number would be quietly wrong — so assert the shape
    # rather than trust it.
    expect = {0: torch.nn.Linear, 1: torch.nn.LayerNorm, 3: torch.nn.Linear,
              4: torch.nn.LayerNorm, 6: torch.nn.Linear}
    for i in idx:
        got = agent.unmerge_actor.net[i]
        if not isinstance(got, expect[i]):
            raise AssertionError(
                f"agent._MLP layout changed: net[{i}] is {type(got).__name__}, "
                f"expected {expect[i].__name__} — --adapt-unfreeze indices are stale")
    trainable = []
    for module, n in ((agent.unmerge_actor, n_cat), (agent.factor_actor, n_fac)):
        for prm in module.parameters():
            prm.requires_grad = False
        for i in groups[n]:
            for prm in module.net[i].parameters():
                prm.requires_grad = True
                trainable.append(prm)
    return trainable


def adapt_in_place(agent, loops: list, data: dict, kind: str, lr: float,
                   steps: int, unfreeze=(1, 1)):
    """
    Fine-tune `agent` on the measured cells of `loops`. Mutates and returns it.

    MIRROR: adapt_eval.adapt, with ONE deliberate change — the PPO
    behaviour-cloning target uses the DEADZONE-GATED oracle. adapt_eval.py:302
    calls the raw oracle_of, which on a loop whose best transform is +0.001
    against a 0.005 deadzone clones toward that transform while label_loops
    calls the loop a no-op. The policy would be taught to fire exactly where
    declining is correct, and then scored against labels that say so.
    (adapt_eval.py is left untouched; it is the pinned checkpoint-driven study.)

    How much of each head is tunable is set by --adapt-unfreeze; see
    _freeze_for_adaptation. The critic stays frozen throughout — it plays no
    part in greedy selection.

    The signal matches each agent's native one, so both consume the same cells:
      ppo    — cross-entropy toward the gated oracle-best action.
      bandit — MSE of Q(s,a) against the measured reward over every cell.
    """
    import torch.nn.functional as F

    from adapt_eval import make_state
    from agent import (FACTOR_VALUES, _IDX_TRIP_COUNT, _IDX_TRIP_COUNT_KNOWN,
                       build_factor_mask)

    trainable = _freeze_for_adaptation(agent, *unfreeze)
    if not loops or not trainable:
        return agent

    # The unfreeze works unchanged on the 3-way head: CategoryActor is an _MLP
    # with the same net stack, so the same layer indices apply.
    fit = {"ppo": _fit_ppo, "bandit": _fit_bandit,
           "category": _fit_category_ppo,
           "category-bandit": _fit_category_bandit}.get(kind)
    if fit is None:
        raise ValueError(f"unknown agent kind for adaptation: {kind!r}")
    fit(agent, loops, data, trainable, lr, steps)
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


def _category_rows(loops, data):
    """
    Per-category Q1 targets and the factor cells that back them.

    ((s1, category, target), (s2, factor_idx, reward)) rows.

    The no-op's target is the constant 0.0 — its value is KNOWN, so unlike the
    two transform arms it is never estimated. That is the whole point of the
    3-way head: the arm that is correct on 40% of loops stops being inferred
    from a noisy max.

    The transform arms take the max over their own MEASURED cells, restricted to
    the category's legal factors — unroll_only excludes factor 1, because (0,1)
    IS the no-op and letting it in would give the two arms an identical target.
    """
    from adapt_eval import make_state
    from agent import (FACTOR_VALUES, _IDX_TRIP_COUNT, _IDX_TRIP_COUNT_KNOWN)
    from category_agent import (NOOP, UNMERGE_UNROLL, UNROLL_ONLY,
                                category_factor_mask)

    q1_rows, q2_rows = [], []
    for l in loops:
        table = data["tables"][(l["benchmark_name"], l["loop_idx"])]
        raw = l["pre_features_raw"]
        known = raw[_IDX_TRIP_COUNT_KNOWN] > 0.5
        trip = int(raw[_IDX_TRIP_COUNT])
        s1, _ = make_state(l, data["normalizer"], data["postf"], 0)
        # Known exactly; never estimated.
        q1_rows.append((s1, NOOP, 0.0))
        for cat, u in ((UNROLL_ONLY, 0), (UNMERGE_UNROLL, 1)):
            m = category_factor_mask(cat, known, trip)
            cells = [(f, r) for (uu, f), r in table.items()
                     if uu == u and f in FACTOR_VALUES
                     and bool(m[FACTOR_VALUES.index(f)])]
            if not cells:
                continue
            _, s2 = make_state(l, data["normalizer"], data["postf"], u)
            q1_rows.append((s1, cat, max(r for _, r in cells)))
            for f, r in cells:
                q2_rows.append((s2, FACTOR_VALUES.index(f), r))
    return q1_rows, q2_rows


def _fit_category_bandit(agent, loops, data, trainable, lr, steps) -> None:
    """Q-regression for the 3-way head: Q1 over categories, Q2 over factors."""
    import torch.nn.functional as F

    q1_rows, q2_rows = _category_rows(loops, data)
    if not q1_rows:
        return
    s1b = torch.stack([x[0] for x in q1_rows])
    c_b = torch.tensor([x[1] for x in q1_rows], dtype=torch.long)
    t1b = torch.tensor([x[2] for x in q1_rows], dtype=torch.float32)
    # q2_rows can be empty (a loop with no measured transform cell) while q1_rows
    # is not — _category_rows always emits the no-op target, whose value is known
    # exactly. Returning early here would throw that away, which is the one piece
    # of supervision the 3-way head exists to provide.
    if q2_rows:
        s2b = torch.stack([x[0] for x in q2_rows])
        f_b = torch.tensor([x[1] for x in q2_rows], dtype=torch.long)
        r_b = torch.tensor([x[2] for x in q2_rows], dtype=torch.float32)
    else:
        s2b = f_b = r_b = None

    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
    for _ in range(steps):
        opt.zero_grad()
        q1 = agent.unmerge_actor.forward(s1b).gather(
            1, c_b.unsqueeze(1)).squeeze(1)
        loss = F.mse_loss(q1, t1b)
        if s2b is not None:
            q2 = agent.factor_actor.forward(s2b).gather(
                1, f_b.unsqueeze(1)).squeeze(1)
            loss = loss + F.mse_loss(q2, r_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 0.5)
        opt.step()


def _fit_category_ppo(agent, loops, data, trainable, lr, steps) -> None:
    """
    Behaviour cloning for the 3-way head.

    The category head is cloned on every loop. The factor head is cloned ONLY on
    loops whose gated oracle is a transform — a no-op loop has no factor to
    clone toward, and including it would train the factor head on an action the
    policy never takes there.
    """
    import torch.nn.functional as F

    from adapt_eval import make_state
    from agent import (FACTOR_VALUES, _IDX_TRIP_COUNT, _IDX_TRIP_COUNT_KNOWN)
    from category_agent import (NOOP, UNMERGE_UNROLL, UNROLL_ONLY,
                                category_factor_mask, category_mask)
    from offline_data import category_of

    _CAT = {"noop": NOOP, "unroll_only": UNROLL_ONLY,
            "unmerge_unroll": UNMERGE_UNROLL}
    cat_rows, fac_rows = [], []
    for l in loops:
        table = data["tables"][(l["benchmark_name"], l["loop_idx"])]
        raw = l["pre_features_raw"]
        known = raw[_IDX_TRIP_COUNT_KNOWN] > 0.5
        trip = int(raw[_IDX_TRIP_COUNT])
        (u, f), _ = oracle_of_gated(table, data["deadzone"])
        cat = _CAT[category_of((u, f))]
        s1, s2 = make_state(l, data["normalizer"], data["postf"], u)
        cat_rows.append((s1, cat, category_mask(known, trip)))
        if cat != NOOP:
            fac_rows.append((s2, FACTOR_VALUES.index(f),
                             category_factor_mask(cat, known, trip)))
    if not cat_rows:
        return
    s1b = torch.stack([x[0] for x in cat_rows])
    c_b = torch.tensor([x[1] for x in cat_rows], dtype=torch.long)
    m1b = torch.stack([x[2] for x in cat_rows])
    # Built ONCE. These are loop-invariant; stacking them inside the step loop
    # rebuilt identical tensors on every one of --adapt-steps iterations.
    if fac_rows:
        s2b = torch.stack([x[0] for x in fac_rows])
        f_b = torch.tensor([x[1] for x in fac_rows], dtype=torch.long)
        m2b = torch.stack([x[2] for x in fac_rows])
    else:
        s2b = f_b = m2b = None

    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
    for _ in range(steps):
        opt.zero_grad()
        c_logits = agent.unmerge_actor.forward(s1b).masked_fill(
            ~m1b, float("-inf"))
        loss = F.cross_entropy(c_logits, c_b)
        if s2b is not None:
            f_logits = agent.factor_actor.forward(s2b).masked_fill(
                ~m2b, float("-inf"))
            loss = loss + F.cross_entropy(f_logits, f_b)
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
    for _l in fingerprint(loops, args):
        log.info(_l)

    log.info("Few-shot adaptation | %d-fold x %d init seed(s) | agent=%s | "
             "adapt lr=%g steps=%d unfreeze cat=%d factor=%d",
             args.folds, args.seeds, args.agent, args.adapt_lr,
             args.adapt_steps, args.adapt_unfreeze[0], args.adapt_unfreeze[1])
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
    # The eval LOOPS per seed, kept alongside the picks so the oracle and
    # always-no-op can be scored on exactly the same set. The adapt/eval split
    # is drawn from a per-(fold,seed) rng, so these differ by seed and a single
    # global ceiling would be the wrong denominator.
    ev: dict = {}
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
                                   args.adapt_lr, args.adapt_steps,
                                   args.adapt_unfreeze)
                ad = greedy_picks(tuned, eval_l, data["normalizer"],
                                  data["postf"])
                if adapt_l:
                    zero.setdefault(seed, []).extend(zs)
                    adapt.setdefault(seed, []).extend(ad)
                    ev.setdefault(seed, []).extend(eval_l)
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
    heads = []
    for seed in sorted(zero):
        z, a = _sc(zero[seed]), _sc(adapt[seed])
        # The ceiling ON THESE LOOPS. Without it, "+0.0129" is unreadable: it
        # could be 5% of what was available or 60%. Scored on ev[seed], the same
        # loops the two policy rows above are scored on.
        o = _sc(oracle_picks(ev[seed], data["tables"], args.deadzone))
        n = _sc(always_noop_picks(ev[seed]))
        log.info(table_row(f"seed {seed} zero-shot", z))
        log.info(table_row(f"seed {seed} + {args.adapt_steps} adapt steps", a))
        log.info(table_row(f"seed {seed} oracle (ceiling)", o))
        log.info(table_row(f"seed {seed} always-no-op", n))
        log.info("")
        heads.append((o["mean_realized"], a["mean_realized"], z["mean_realized"]))
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

    om = st.mean([h[0] for h in heads])
    am = st.mean([h[1] for h in heads])
    zm2 = st.mean([h[2] for h in heads])
    log.info("  HEADROOM ON THE EVALUATION LOOPS")
    log.info("    oracle mean %+.4f   always-no-op +0.0000   zero-shot %+.4f"
             "   few-shot %+.4f", om, zm2, am)
    log.info("    few-shot recovers %.0f%% of the gap from no-op to the ceiling"
             "   (%.0f%% zero-shot)",
             100 * am / om if om else float("nan"),
             100 * zm2 / om if om else float("nan"))
    log.info("    Read the mean against the ORACLE on these loops, not against "
             "the\n    population's +0.0989 — this subset excludes every "
             "single-loop benchmark.")

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
        write_csv(args.csv_out, rows)
        log.info("\n  per-benchmark: %s", args.csv_out)


def main() -> None:
    p = build_parser()
    g = p.add_argument_group("few-shot adaptation")
    g.add_argument("--adapt-lr", type=float, default=1e-3)
    g.add_argument("--adapt-steps", type=int, default=50,
                   help="Gradient steps per target application.")
    g.add_argument("--adapt-unfreeze", type=_unfreeze_spec, default=(1, 1),
                   help="Trailing parameterised layers to fine-tune, as N or "
                        "CAT,FACTOR. 1 = output projection only; 2 adds "
                        "Linear(128,64)+LayerNorm; 3 is the whole stack. "
                        "'1,2' gives the factor head more capacity than the "
                        "category head, which is where the evidence says the "
                        "bottleneck is. Two adaptation loops give ~20-40 "
                        "examples, so loosening both invites memorisation.")
    g.add_argument("--min-fill", type=float, default=0.0,
                   help="Drop loops whose measured-cell fraction is below this. "
                        "An adaptation loop with partial coverage teaches the "
                        "model from a table that does not exist; the population "
                        "mean fill is 0.992, so 0.95 costs almost nothing.")
    args = p.parse_args()
    torch.set_num_threads(max(1, args.threads))
    warn_ignored_flags(args)

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
