"""
Offline CPU training and evaluation for the UU study — PPO and contextual bandit.

Trains the SAME agents the online pipeline trains (rl/agent.py, unmodified) and
takes the SAME decisions the online worker takes, but reads each action's reward
from the measured table instead of compiling and timing it. Pure CPU: no GPU, no
toolchain, no HeCBench source tree.

    python3 offline_train.py RUN_DIR --agent bandit --deadzone 0.005
    python3 offline_train.py RUN_DIR --agent ppo --folds 5 --seeds 3
    python3 offline_train.py RUN_DIR --agent bandit --mode score-ckpt \\
        --checkpoint ckpt.pt --split-seed 42

PROTOCOL (mode=cv, the default)
-------------------------------
Grouped K-fold over BENCHMARKS. For each fold, train from scratch on the other
folds and predict the held-out one; repeat under several init seeds. Nothing is
carried between folds — no warm start, no shared optimizer state.

The union of held-out predictions covers every labelled loop, each predicted by
a model that never saw it. That union is the headline; it is the only number
here that estimates deployment quality. Three spreads are reported and they mean
different things:

    pooled over the union      the estimate
    across folds               generalization difficulty / data heterogeneity
    across init seeds          optimization stability

Those last two are separated on purpose. Varying the partition and the
initialization together makes it impossible to tell whether spread came from the
data or from the optimizer, which is exactly the question behind "did the policy
collapse, or was that fold just hard".

WHY K-FOLD AND NOT REPEATED RANDOM SPLITS
-----------------------------------------
Effective sample size is ~138 benchmarks, not 444 loops: benefit is bimodal per
application (median benefit rate 100%, 26% of benchmarks at exactly 0%), and an
8-seed sweep put the test-split spread at +-14.2pp. A single held-out split is
close to noise at that N. K-fold tests every benchmark exactly once, wastes
nothing, and double-counts nothing.

WHAT THIS IS NOT
----------------
Not a search. With a near-complete table every arm is observable at zero cost, so
neither PPO's entropy bonus nor the bandit's epsilon is buying exploration — this
is closer to supervised learning of the argmax. The RL framing's live claim is
sample efficiency (how few measurements reach X% of oracle), which is a separate
experiment. Read these numbers as "can a function over these features generalize
to an unseen application", nothing more.
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

import torch                                                    # noqa: E402

from offline_data import (_RULE, IDX_TRIP_COUNT, IDX_TRIP_KNOWN,  # noqa: E402
                          NOOP, always_noop_picks, benchmark_dominant_picks,
                          format_confusion, format_report, grouped_kfold,
                          holdout_split, labelled_loops, load_run, loops_for,
                          marginal_picks, marginal_ranking, oracle_picks,
                          score_decisions, table_header, table_row)

# force=True is load-bearing: importing offline_data pulls in adapt_eval and
# train, both of which call basicConfig with a timestamped format at import
# time. basicConfig is a no-op once handlers exist, so without force the whole
# report comes out prefixed "HH:MM:SS INFO" — and only on the first line of each
# multi-line block, which is worse than useless.
logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
log = logging.getLogger("offline")
# The trip-count mask logs one INFO line per masked loop per epoch — thousands of
# lines in a sweep, and it says nothing the summary does not.
logging.getLogger("agent").setLevel(logging.WARNING)

DEVICE = torch.device("cpu")


def _mr(args):
    """
    Reward charged at SCORING time for a cell with no row in the cache
    (None = exclude it from performance instead).

    Kept identical to the training rule by default. Scoring these as absent
    while training charges them would report a policy against a standard it was
    never trained to, and would discount only the learned policy's mistakes:
    the oracle, always-no-op and marginal-best all pick cells that exist by
    construction and can never land here.
    """
    return (args.compile_failure_penalty
            if args.score_missing == "penalty" else None)


# ---------------------------------------------------------------------------
# Action selection and reward lookup — mirrors of the live worker
# ---------------------------------------------------------------------------

def act(agent, loop: dict, normalizer, postf: dict, greedy: bool):
    """
    One policy decision for one loop.

    MIRROR: train._worker_fn — unmerge is decided on s1; the factor is decided on
    the state matching THAT decision; the trip-count mask is built from RAW
    features, never from the z-scored tensor. Identical for Agent and
    BanditAgent: both expose the same two selectors.

    Returns (unmerge, factor, factor_idx, log_p1, log_p2, mask, s1, s2).
    """
    from adapt_eval import make_state
    from agent import FACTOR_VALUES

    raw = loop["pre_features_raw"]
    s1, _ = make_state(loop, normalizer, postf, 0)
    unmerge, log_p1 = agent.select_unmerge(s1, greedy=greedy)
    _, s2 = make_state(loop, normalizer, postf, unmerge)
    factor_idx, log_p2, mask = agent.select_factor(
        s2,
        trip_known=raw[IDX_TRIP_KNOWN] > 0.5,
        trip_count=int(raw[IDX_TRIP_COUNT]),
        loop_idx=loop["loop_idx"],
        greedy=greedy,
    )
    return (int(unmerge), FACTOR_VALUES[factor_idx], factor_idx,
            log_p1, log_p2, mask, s1, s2)


def reward_for(tables: dict, loop: dict, unmerge: int, factor: int):
    """
    (reward, factor_active, measured) for a chosen action.

    MIRROR: train._worker_fn — the pure no-op (unmerge==0, factor==1) earns
    exactly 0.0 with no compile and does NOT train the factor head
    (factor_active=False); there is no unroll decision in it to learn.

    measured=False means the cell has no row in the cache. After exhaustive
    collection that is NOT "we never looked": every cell that compiled and ran
    has a value, compile failures are stored at the failure penalty and timeouts
    at -1.0, so an absent row is a cell that failed in some way that produced no
    measurement (persistent measure failures, mostly). Factors the trip-count
    mask forbids are never selectable in the first place, so they cannot land
    here. The default therefore charges these the failure penalty rather than
    dropping them — the agent should learn to avoid the action, and dropping the
    sample teaches it nothing about a choice that would fail on real hardware.
    """
    key = (loop["benchmark_name"], loop["loop_idx"])
    if unmerge == 0 and factor == 1:
        return 0.0, False, True
    cell = tables[key].get((unmerge, factor))
    if cell is None:
        return None, True, False
    return float(cell), True, True


@torch.no_grad()
def greedy_picks(agent, loops: list, normalizer, postf: dict) -> list:
    """Deployment-mode argmax over a set of loops. Runs every epoch, so no_grad
    is worth it — and it keeps evaluation from ever touching the graph."""
    out = []
    for l in loops:
        u, f, *_ = act(agent, l, normalizer, postf, greedy=True)
        out.append((l["benchmark_name"], l["loop_idx"], (u, f)))
    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@torch.no_grad()
def rollout_one(agent, loop: dict, data: dict, args):
    """
    One on-policy sample. (RolloutEntry or None, absent_cell: bool)

    no_grad covers SELECTION only. The ppo_update call must stay outside it —
    running an update under no_grad disables every gradient and training
    silently does nothing while still printing losses.
    """
    from agent import RolloutEntry

    u, f, f_idx, lp1, lp2, mask, s1, s2 = act(
        agent, loop, data["normalizer"], data["postf"], greedy=False)
    r, factor_active, measured = reward_for(data["tables"], loop, u, f)
    if not measured:
        if args.missing == "skip":
            return None, True
        r = args.compile_failure_penalty
    # MIRROR: train._send_loop_result -> main's rebuild. Online, entries cross a
    # process queue as PLAIN PYTHON, so ppo_update sees graph-free constants.
    # Rebuilt here in the same shapes and dtypes: a log-prob still carrying
    # grad_fn would make the PPO ratio differentiable through the
    # COLLECTION-time policy, which is meant to be a fixed reference.
    return RolloutEntry(
        state1=s1.detach(), state2=s2.detach(),
        action1=int(u), action2=int(f_idx),
        log_prob1=torch.tensor(float(lp1), dtype=torch.float32),
        log_prob2=torch.tensor(float(lp2), dtype=torch.float32),
        reward=float(r),
        mask2=mask.detach().to(torch.bool),
        factor_active=factor_active), (not measured)


def run_epoch(agent, order: list, buf, data: dict, args) -> tuple:
    """
    One epoch of collection with MID-EPOCH updates, exactly as the pipeline
    does it (train.py:2088-2090 and 2838-2840): update the moment the buffer
    fills, then clear.

    This is not a detail. The pipeline sizes its buffer at 128 with K=2 and
    batch 8 to hit ~0.25 gradient updates per sample (its own comment,
    train.py:214-216). Collecting a whole epoch into one buffer and updating
    once at the end gives ~0.06 — four times fewer gradient steps on the same
    data, plus an advantage normalisation computed over a 300-sample batch
    instead of a 128-sample one. Same samples, different optimiser entirely.

    Returns (n_absent_cells, n_updates, sum_actor_loss).
    """
    n_missing = n_updates = 0
    loss_sum = 0.0
    for l in order:
        entry, missing = rollout_one(agent, l, data, args)
        n_missing += int(missing)
        if entry is None:
            continue
        buf.append(entry)
        if buf.full():
            stats = agent.ppo_update(buf)      # OUTSIDE no_grad — see above
            buf.clear()
            n_updates += 1
            loss_sum += stats.get("actor_loss", 0.0)
    # MIRROR: train.py:2146 — flush the partial buffer at end of epoch.
    if len(buf) > 0:
        stats = agent.ppo_update(buf)
        buf.clear()
        n_updates += 1
        loss_sum += stats.get("actor_loss", 0.0)
    return n_missing, n_updates, loss_sum


def _snapshot(agent) -> dict:
    return {n: copy.deepcopy(m.state_dict()) for n, m in
            (("u", agent.unmerge_actor), ("f", agent.factor_actor),
             ("c", agent.critic))}


def _restore(agent, snap: dict) -> None:
    agent.unmerge_actor.load_state_dict(snap["u"])
    agent.factor_actor.load_state_dict(snap["f"])
    agent.critic.load_state_dict(snap["c"])


def make_agent(kind: str, args):
    """
    MIRROR: train.py:2390-2412, key for key.

    _agent_common is shared by both agents; logit_cap and entropy_coef_unmerge
    are passed to PPO ONLY. That is not an oversight to tidy up later — the
    bandit's heads are Q-values, not softmax logits, so a tanh cap would distort
    the regression and there is no policy-entropy term for the unmerge
    coefficient to weight. train.py says so in a comment at :2402; asserted in
    test_agent_construction_matches_pipeline so it stays true.
    """
    from agent import Agent, BanditAgent

    common = dict(
        clip_eps=args.clip_eps,
        K=args.K,
        batch_size=args.batch_size,
        lr=args.lr,
        value_loss_coef=args.value_loss_coef,
        entropy_coef=args.entropy_coef,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        device=DEVICE,
    )
    if kind == "bandit":
        return BanditAgent(epsilon=args.bandit_epsilon, **common)
    return Agent(logit_cap=args.logit_cap,
                 entropy_coef_unmerge=args.entropy_coef_unmerge, **common)


def _schedules(agent, kind: str, epoch: int, args) -> None:
    """
    MIRROR: train.py:2580-2594. Linear schedules, recomputed per epoch.

    Factor-head entropy decays --entropy-coef -> --entropy-coef-final. The
    UNMERGE head's coefficient is deliberately NOT decayed: it protects the
    binary head from extinction and is set once at construction. Bandit epsilon
    decays on the same shape (train._bandit_epsilon).
    """
    n = args.epochs
    frac = (epoch - 1) / (n - 1) if n > 1 else 0.0
    if kind == "bandit":
        agent.epsilon = (args.bandit_epsilon
                         + frac * (args.bandit_epsilon_final - args.bandit_epsilon))
    else:
        agent.entropy_coef = (args.entropy_coef
                              + frac * (args.entropy_coef_final - args.entropy_coef))


def train_agent(kind: str, fit_loops: list, hold_loops: list, data: dict,
                args, seed: int):
    """
    Train from scratch on fit_loops; select the best epoch on hold_loops.

    Construction, update cadence, schedules and warm start all mirror train.py
    so an offline number is comparable to an online one. The ONLY thing removed
    is the compile-and-measure step, replaced by a table lookup.

    Selection metric is MEAN REALIZED REWARD over every scored held-out loop.
    Not capture, and not accuracy: capture ignores loops with no headroom, so a
    policy that fires destructively on them scores well on it; accuracy weights a
    +0.0001 loop the same as a +0.99 one. Mean realized is also the number that
    compares directly against always-no-op's exact 0.0.
    """
    from agent import RolloutBuffer

    torch.manual_seed(seed)
    random.seed(seed)

    agent = make_agent(kind, args)

    # MIRROR: train.py:2542 — the bandit warm-starts its Q-heads on cached
    # cells before any rollout. Restricted to FIT loops: build_warm_start_entries
    # takes the assignment list, so passing only this fold's training loops is
    # what keeps the held-out fold out of the warm start.
    n_ws_cells = 0
    if kind == "bandit" and args.bandit_warm_epochs > 0:
        from train import build_warm_start_entries
        ws, n_ws_cells, _ = build_warm_start_entries(
            fit_loops, data["rewards"], data["postf"], data["normalizer"])
        if n_ws_cells:
            agent.warm_start(ws, args.bandit_warm_epochs)

    best_score, best_snap, best_epoch, since = -float("inf"), _snapshot(agent), 0, 0
    n_missing = n_updates = 0
    history = []
    buf = RolloutBuffer(capacity=args.buffer_size)

    for epoch in range(1, args.epochs + 1):
        _schedules(agent, kind, epoch, args)
        order = list(fit_loops)
        random.shuffle(order)
        miss, ups, loss = run_epoch(agent, order, buf, data, args)
        n_missing += miss
        n_updates += ups
        if ups == 0:
            continue

        m = score_decisions(greedy_picks(agent, hold_loops, data["normalizer"],
                                         data["postf"]),
                            data["tables"], data["labels"], args.deadzone,
                            _mr(args))
        # FIT curve, sampled. Without it there is no way to tell whether fit had
        # plateaued by epoch 20 or was still climbing at 100 — and therefore no
        # way to answer "would more epochs fit better?" except by guessing.
        # Sampled rather than every epoch because it is a full pass over ~300
        # loops; always taken on the last epoch so the endpoint is exact.
        fit_a = fit_m = float("nan")
        if args.curve_every and (epoch % args.curve_every == 0
                                 or epoch == args.epochs):
            fm = score_decisions(greedy_picks(agent, fit_loops,
                                              data["normalizer"], data["postf"]),
                                 data["tables"], data["labels"], args.deadzone,
                                 _mr(args))
            fit_a, fit_m = fm["accuracy"], fm["mean_realized"]
        history.append({"epoch": epoch, "hold_mean_realized": m["mean_realized"],
                        "hold_accuracy": m["accuracy"],
                        "fit_accuracy": fit_a, "fit_mean_realized": fit_m,
                        "actor_loss": loss / max(ups, 1)})
        if m["mean_realized"] > best_score:
            best_score, best_snap, best_epoch, since = (
                m["mean_realized"], _snapshot(agent), epoch, 0)
        else:
            since += 1
            if args.patience and since >= args.patience:
                break

    if not history:
        log.warning("    WARNING: no training samples in any epoch — the agent "
                    "is UNTRAINED. Check --missing and the table's coverage.")

    # Score the FINAL agent on fit BEFORE restoring the val-selected snapshot.
    # The restored checkpoint is whichever epoch maximised VAL, so its fit is
    # not the best fit the run achieved. Asking "would more epochs fit better?"
    # requires the final-epoch fit, not the selected one — otherwise extra
    # epochs are discarded by selection and the answer looks like "no".
    fin = score_decisions(greedy_picks(agent, fit_loops, data["normalizer"],
                                       data["postf"]),
                          data["tables"], data["labels"], args.deadzone,
                          _mr(args))

    _restore(agent, best_snap)
    for mod in (agent.unmerge_actor, agent.factor_actor, agent.critic):
        mod.eval()
    return agent, {"best_epoch": best_epoch, "best_hold_mean_realized": best_score,
                   "epochs_run": len(history), "n_missing_cells": n_missing,
                   "n_updates": n_updates, "n_warm_start_cells": n_ws_cells,
                   "final_fit_accuracy": fin["accuracy"],
                   "final_fit_mean_realized": fin["mean_realized"],
                   "history": history}


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

# Header and data rows share ONE template. Two hand-aligned format strings drift
# apart the moment a column changes width; with a single template they cannot.
# Field widths must fit the HEADER LABEL as well as the data — a label wider
# than its field silently pushes every later column right, which is how the
# first hand-aligned version drifted.
_FOLD_ROW = ("  {fold:>5} {seed:>5} | {loops:>5} /{bench:>4} |"
             "{a_noop:>8}{a_fit:>8}{a_val:>8}{a_test:>8} |"
             "{m_fit:>9}{m_val:>9}{m_test:>9} | {cap:>7} | {ep:>4}")


def fold_header() -> list:
    """The two header lines and the rule, measured off _FOLD_ROW itself."""
    h2 = _FOLD_ROW.format(fold="fold", seed="seed", loops="loops", bench="bn",
                          a_noop="no-op", a_fit="fit", a_val="val",
                          a_test="TEST", m_fit="fit", m_val="val",
                          m_test="TEST", cap="capt", ep="ep")
    blank = _FOLD_ROW.format(fold="", seed="", loops="", bench="", a_noop="",
                             a_fit="", a_val="", a_test="", m_fit="", m_val="",
                             m_test="", cap="", ep="")
    # Centre the group labels over their own column blocks, located by the
    # separators in the blank row.
    bars = [i for i, c in enumerate(blank) if c == "|"]
    h1 = list(" " * len(blank))
    for label, lo, hi in (("held out", bars[0], bars[1]),
                          ("ACCURACY", bars[1], bars[2]),
                          ("MEAN REALIZED", bars[2], bars[3]),
                          ("test", bars[3], bars[4]),
                          ("best", bars[4], len(blank))):
        start = lo + 1 + max(0, (hi - lo - 1 - len(label)) // 2)
        h1[start:start + len(label)] = label
    for b in bars:
        h1[b] = "|"
    return ["".join(h1), h2, "  " + "-" * (len(h2) - 2)]


def run_cv(data: dict, args) -> None:
    loops = labelled_loops(data)
    benches = sorted({l["benchmark_name"] for l in loops})
    folds = grouped_kfold(benches, args.folds, args.fold_seed)

    log.info("Grouped %d-fold over %d benchmarks (%d labelled loops), "
             "%d init seed(s) — %d runs total",
             args.folds, len(benches), len(loops), args.seeds,
             args.folds * args.seeds)
    log.info("agent=%s  epochs=%d  patience=%d  buffer=%d  batch=%d  missing=%s",
             args.agent, args.epochs, args.patience, args.buffer_size,
             args.batch_size, args.missing)
    log.info("")
    for _h in fold_header():
        log.info(_h)

    union: dict = {}          # seed -> list of held-out picks
    curve_rows: list = []
    fold_rows, per_fold_scores = [], {s: [] for s in range(args.seeds)}

    for k, (tr_b, te_b) in enumerate(folds):
        fit_b, hold_b = holdout_split(tr_b, args.holdout_frac, args.fold_seed + k)
        fit_l = loops_for(loops, fit_b)
        hold_l = loops_for(loops, hold_b)
        test_l = loops_for(loops, te_b)
        for s in range(args.seeds):
            seed = args.base_seed + s
            agent, info = train_agent(args.agent, fit_l, hold_l, data, args, seed)
            picks = greedy_picks(agent, test_l, data["normalizer"], data["postf"])
            union.setdefault(seed, []).extend(picks)
            def _sc(ls):
                return score_decisions(
                    greedy_picks(agent, ls, data["normalizer"], data["postf"]),
                    data["tables"], data["labels"], args.deadzone, _mr(args))

            m = score_decisions(picks, data["tables"], data["labels"],
                                args.deadzone, _mr(args))
            per_fold_scores[s].append(m)
            # FIT and VAL are reported alongside TEST because they answer
            # different questions, and the remedy differs by which one fails:
            #   fit  — the loops it trained on, deliberately NOT held out. If
            #          the policy scores badly even here it cannot represent
            #          the mapping at all: a capacity/optimisation problem,
            #          not a generalization one.
            #   val  — the slice held out of training to pick the epoch. The
            #          reported checkpoint was SELECTED on it, so it is
            #          optimistically biased and is not a deployment estimate.
            #   test — the fold, unseen in every capacity. The only estimate.
            # fit >> test is a generalization gap; fit ~ test ~ bad is underfitting.
            m_fit, m_val = _sc(fit_l), _sc(hold_l)
            # always-no-op on THIS fold's test loops. Its mean is 0.0000 by
            # construction on every split, so each MEAN REALIZED column already
            # IS the margin over doing nothing; only its ACCURACY varies by fold.
            base = score_decisions(always_noop_picks(test_l), data["tables"],
                                   data["labels"], args.deadzone, _mr(args))
            def _p(x):
                return f"{100 * x:.1f}%"
            log.info(_FOLD_ROW.format(
                fold=f"{k + 1}/{args.folds}", seed=str(seed),
                loops=str(len(test_l)), bench=str(len(te_b)),
                # The displayed fit is the FINAL epoch's, not the val-selected
                # epoch's: fit answers "can this model represent the mapping at
                # all", which is a property of the training run. Scoring it on
                # whichever checkpoint happened to maximise val answers a
                # different question. The selected-epoch fit is kept in the CSV
                # (accuracy_fit / mean_realized_fit) because the fit-vs-test GAP
                # must compare one and the same model.
                a_noop=_p(base["accuracy"]),
                a_fit=_p(info["final_fit_accuracy"]),
                a_val=_p(m_val["accuracy"]), a_test=_p(m["accuracy"]),
                m_fit=f"{info['final_fit_mean_realized']:+.4f}",
                m_val=f"{m_val['mean_realized']:+.4f}",
                m_test=f"{m['mean_realized']:+.4f}",
                cap=_p(m["capture"]), ep=str(info["best_epoch"])))
            fold_rows.append({
                "fold": k + 1, "seed": seed, "n_test_benchmarks": len(te_b),
                "n_test_loops": len(test_l), "n_fit_loops": len(fit_l),
                "n_val_loops": len(hold_l), "best_epoch": info["best_epoch"],
                "epochs_run": info["epochs_run"], "n_updates": info["n_updates"],
                "n_warm_start_cells": info["n_warm_start_cells"],
                "accuracy": round(m["accuracy"], 6),
                "accuracy_fit": round(m_fit["accuracy"], 6),
                "accuracy_val": round(m_val["accuracy"], 6),
                "capture": round(m["capture"], 6),
                "capture_fit": round(m_fit["capture"], 6),
                "mean_realized_fit": round(m_fit["mean_realized"], 6),
                "mean_realized_val": round(m_val["mean_realized"], 6),
                "mean_realized": round(m["mean_realized"], 6),
                "noop_accuracy_test": round(base["accuracy"], 6),
                "regression_rate": round(m["regression_rate"], 6),
                "loops_unmeasured": m["loops_unmeasured"],
                "n_missing_cells_in_training": info["n_missing_cells"],
                # fit at the LAST epoch vs at the val-SELECTED epoch. If the
                # first keeps rising with --epochs and the second does not, the
                # limit is selection, not capacity.
                "accuracy_fit_final": round(info["final_fit_accuracy"], 6),
                "mean_realized_fit_final": round(
                    info["final_fit_mean_realized"], 6),
            })
            if args.curve_out:
                for h in info["history"]:
                    curve_rows.append(dict(fold=k + 1, seed=seed, **h))

    log.info(fold_header()[2])
    log.info("  always-no-op scores exactly +0.0000 on every split, so each "
             "MEAN REALIZED figure\n  IS its own margin over doing nothing: "
             "negative means the policy made that split\n  SLOWER. Negative "
             "capture means the picks on loops that HAD headroom netted a\n"
             "  slowdown, not merely a failure to capture the upside.")
    log.info("  fit = the training loops at the FINAL epoch — it answers "
             "'can the model represent\n  the mapping at all', so it must not "
             "be gated on a val-selected checkpoint.\n  val chose the epoch, so "
             "val is optimistically biased. Only TEST estimates\n  deployment "
             "quality. 'no-op' is always-no-op's accuracy on this fold's TEST "
             "loops\n  — the trivial classifier the policy has to beat.")

    _ff = [(r["accuracy_fit"], r["accuracy_fit_final"]) for r in fold_rows]
    log.info("\n  fit accuracy at the SELECTED epoch %.1f%% vs at the FINAL "
             "epoch %.1f%%.\n  If the final keeps rising with --epochs while "
             "the selected does not, the binding\n  constraint is val-based "
             "selection, not capacity or training length.",
             100 * st.mean([a for a, _ in _ff]),
             100 * st.mean([b for _, b in _ff]))

    _acc = [(r["accuracy_fit_final"], r["accuracy_val"], r["accuracy"],
             r["noop_accuracy_test"]) for r in fold_rows]
    log.info("\n  accuracy over %d runs: fit %.1f%% -> val %.1f%% -> test "
             "%.1f%%   (always-no-op on test: %.1f%%)",
             len(_acc), 100 * st.mean([a for a, _, _, _ in _acc]),
             100 * st.mean([b for _, b, _, _ in _acc]),
             100 * st.mean([c for _, _, c, _ in _acc]),
             100 * st.mean([d for _, _, _, d in _acc]))

    # The GAP uses the selected-epoch fit deliberately — it is the same model
    # that produced the test number, so the difference is generalization. The
    # table's fit column is the final epoch, which answers capacity instead.
    _gap = [(r["mean_realized_fit"], r["mean_realized"]) for r in fold_rows]
    log.info("\n  generalization gap over %d runs (same checkpoint both sides): "
             "fit %+.4f -> test %+.4f  (gap %+.4f)",
             len(_gap), st.mean([a for a, _ in _gap]),
             st.mean([b for _, b in _gap]),
             st.mean([a - b for a, b in _gap]))

    # --- the headline: one table, policy and baselines side by side ---
    pooled = [score_decisions(picks, data["tables"], data["labels"],
                              args.deadzone, _mr(args))
              for _, picks in sorted(union.items())]

    log.info("\n" + "=" * 92)
    log.info("  RESULTS — %d loops, every one predicted by a model that never "
             "saw it", pooled[0]["loops"])
    log.info("  (union of %d held-out folds; baselines scored on the same "
             "population)", args.folds)
    log.info("=" * 92)
    log.info(table_header())
    for s, m in zip(sorted(union), pooled):
        log.info(table_row(f"{args.agent} (init seed {s})", m))
    _baseline_rows(data, loops, folds, args)

    log.info("\n  Read across the row, not down the column: 'capture' means "
             "nothing without the")
    log.info("  oracle's 100%% and always-no-op's 0%% on the same table. "
             "'mean' is the one number")
    log.info("  directly comparable to doing nothing, which scores exactly "
             "+0.0000.")
    log.info("\n  confusion, %s init seed %d (rows = truth, cols = predicted)",
             args.agent, sorted(union)[0])
    log.info(format_confusion(pooled[0]))

    # --- the two spreads, which mean different things ---
    log.info("\n" + "=" * 74)
    log.info("  VARIANCE — separated by source")
    log.info("=" * 74)

    def _spread(vals, label, note):
        if len(vals) < 2:
            log.info("    %-28s %+.4f   (single value)", label, vals[0])
            return
        log.info("    %-28s %+.4f +- %.4f   [%+.4f, %+.4f]   %s",
                 label, st.mean(vals), st.stdev(vals), min(vals), max(vals), note)

    _spread([m["mean_realized"] for m in pooled], "across init seeds (pooled)",
            "optimization stability")
    fold_means = [st.mean([m["mean_realized"] for m in per_fold_scores[s]])
                  for s in range(args.seeds)]
    _spread(fold_means, "fold means, per init seed", "")
    for s in range(args.seeds):
        vals = [m["mean_realized"] for m in per_fold_scores[s]]
        _spread(vals, f"across folds (seed {args.base_seed + s})",
                "data heterogeneity")

    if args.csv_out:
        with open(args.csv_out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(fold_rows[0].keys()))
            w.writeheader()
            w.writerows(fold_rows)
        log.info("\n  per-fold: %s", args.csv_out)
    if args.curve_out and curve_rows:
        with open(args.curve_out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(curve_rows[0].keys()))
            w.writeheader(); w.writerows(curve_rows)
        log.info("  learning curves: %s", args.curve_out)


def _baseline_rows(data: dict, loops: list, folds: list, args) -> None:
    """
    Baseline rows for the results table, over the SAME labelled population.

    marginal-best is fitted per fold on that fold's training benchmarks and
    applied to its held-out ones, exactly like the policy — fitting it on
    everything would hand it test information the policy never had.
    """
    tables, labels = data["tables"], data["labels"]

    marg: list = []
    for tr_b, te_b in folds:
        tr_keys = [(l["benchmark_name"], l["loop_idx"])
                   for l in loops_for(loops, tr_b)]
        marg.extend(marginal_picks(loops_for(loops, te_b), tables, tr_keys))

    def _row(name, picks):
        log.info(table_row(name, score_decisions(picks, tables, labels,
                                                 args.deadzone, _mr(args))))

    log.info(_RULE)
    log.info("  DEPLOYABLE — a function of features; works on unmeasured code")
    _row("always-no-op", always_noop_picks(loops))
    _row("marginal-best", marg)
    log.info(_RULE)
    log.info("  REFERENCE — needs the target measured first; NOT deployable")
    _row("benchmark-dominant", benchmark_dominant_picks(loops, tables, labels))
    _row("oracle (ceiling)", oracle_picks(loops, tables, args.deadzone))
    log.info(_RULE)
    log.info("  A reference beating the policy does NOT mean the policy lost to "
             "a real\n  alternative: neither can be applied to unmeasured code.")

    # marginal-best collapsing onto the no-op is a RESULT, not a duplicated row,
    # and without the means printed it just looks like one. Show the ranking so
    # the reason is on the page.
    all_keys = [(l["benchmark_name"], l["loop_idx"]) for l in loops]
    ranking = marginal_ranking(tables, all_keys)
    if ranking and ranking[0][0] == NOOP:
        log.info("\n  NOTE: the best FEATURE-BLIND action is to decline — no-op "
                 "is exactly 0.0 on\n  every loop, so every transform arm has a "
                 "NEGATIVE mean over the population.\n  That is why "
                 "marginal-best and always-no-op are the same row. Top arms by "
                 "mean:")
        for a, mean, n in ranking[:5]:
            tag = "no-op" if a == NOOP else f"unmerge={a[0]} factor={a[1]}"
            log.info("    %-24s mean %+.4f over %d loops", tag, mean, n)


def run_score_ckpt(data: dict, args) -> None:
    """
    Score an existing checkpoint on ITS OWN held-out split.

    The checkpoint saw ~70% of the population in training, so scoring it over
    everything would be leakage. This reproduces the training split and reports
    the test slice only.

    Read it as ONE fold, not as the estimate: n is ~21 benchmarks, where the
    8-seed sweep measured a +-14.2pp spread. It answers "did this run learn
    anything", not "does the method work".
    """
    from adapt_eval import fresh_agent
    from train import split_benchmarks

    # Order matters as much as the seed — see load_run's eligible_order note.
    order = data["eligible_order"]
    tr, va, te = split_benchmarks(order, args.val_ratio, args.test_ratio,
                                 args.split_seed)
    loops = labelled_loops(data)
    test_l = loops_for(loops, te)

    log.info("Reproduced split seed=%d from the STORED eligible order: "
             "%d train / %d val / %d test benchmarks",
             args.split_seed, len(tr), len(va), len(te))
    log.info("Test benchmarks (verify against the training log before trusting "
             "this number):\n  %s", ", ".join(sorted(te)))
    log.info("Labelled test loops: %d\n", len(test_l))

    agent = fresh_agent(args.agent, args.checkpoint, logit_cap=args.logit_cap)
    m = score_decisions(greedy_picks(agent, test_l, data["normalizer"],
                                     data["postf"]),
                        data["tables"], data["labels"], args.deadzone,
                        _mr(args))
    log.info("=" * 74)
    log.info("  %s on its own test split — ONE FOLD, not the estimate",
             args.checkpoint.name)
    log.info("=" * 74)
    log.info(format_report(args.agent, m))
    log.info("\n  confusion")
    log.info(format_confusion(m))

    log.info("\n  same-split baselines")
    log.info(format_report("always-no-op", score_decisions(
        always_noop_picks(test_l), data["tables"], data["labels"],
        args.deadzone, _mr(args))))
    tr_keys = [(l["benchmark_name"], l["loop_idx"])
               for l in loops_for(loops, tr)]
    log.info(format_report("marginal-best (fit on train)", score_decisions(
        marginal_picks(test_l, data["tables"], tr_keys), data["tables"],
        data["labels"], args.deadzone, _mr(args))))
    log.info(format_report("oracle (reference ceiling)", score_decisions(
        oracle_picks(test_l, data["tables"], args.deadzone), data["tables"],
        data["labels"], args.deadzone, _mr(args))))


# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """
    Separate from main() so the tests can construct args from the REAL defaults
    instead of a hand-written Namespace that drifts the moment a flag changes.
    """
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--mode", choices=["cv", "score-ckpt"], default="cv")
    p.add_argument("--agent", choices=["ppo", "bandit"], default="bandit")
    p.add_argument("--deadzone", type=float, required=True,
                   help="REQUIRED: must match the cache (0.005).")
    p.add_argument("--labels", type=Path, default=None,
                   help="default: RUN_DIR/loop_labels.csv")
    p.add_argument("--csv-out", type=Path, default=None)
    p.add_argument("--curve-out", type=Path, default=None,
                   help="Per-epoch learning curves (fold, seed, epoch, fit and "
                        "holdout accuracy/mean). This is what answers 'would "
                        "more epochs fit better?' — a single final number "
                        "cannot.")
    p.add_argument("--curve-every", type=int, default=5,
                   help="Sample the FIT curve every N epochs (0 disables). Each "
                        "sample is a full pass over the training loops; the "
                        "last epoch is always sampled. (default: 5)")

    g = p.add_argument_group("cross-validation")
    g.add_argument("--folds", type=int, default=5)
    g.add_argument("--seeds", type=int, default=3, help="init seeds per fold")
    g.add_argument("--fold-seed", type=int, default=0,
                   help="controls the PARTITION only")
    g.add_argument("--base-seed", type=int, default=100,
                   help="first init seed; controls INITIALIZATION only")
    g.add_argument("--holdout-frac", type=float, default=0.15,
                   help="fraction of each fold's train benchmarks used for "
                        "early stopping (never the test fold)")

    g = p.add_argument_group("training — defaults MIRROR train.py's argparse")
    # Every default below is train.py's. They are not tuning knobs here: an
    # offline number is only comparable to an online one if the optimiser is the
    # same one. Changing any of these makes this a different experiment.
    g.add_argument("--epochs", type=int, default=100)
    g.add_argument("--buffer-size", type=int, default=128,
                   help="Rollout buffer capacity before an update is triggered. "
                        "With K=2 and batch 8 this gives ~0.25 gradient updates "
                        "per sample, which is what the pipeline runs. Raising it "
                        "to a whole epoch cuts that ~4x. (default: 128)")
    g.add_argument("--patience", type=int, default=0,
                   help="Early-stop after this many epochs with no holdout "
                        "improvement. 0 (default) = run all epochs and select "
                        "the best, which is what the pipeline does.")
    g.add_argument("--lr", type=float, default=3e-4)
    g.add_argument("--batch-size", type=int, default=8)
    g.add_argument("--K", type=int, default=2, dest="K",
                   help="Update epochs per rollout buffer. (default: 2)")
    g.add_argument("--value-loss-coef", type=float, default=0.5)
    g.add_argument("--weight-decay", type=float, default=0.01)
    g.add_argument("--max-grad-norm", type=float, default=0.5)
    g.add_argument("--clip-eps", type=float, default=0.2, help="PPO only")
    g.add_argument("--entropy-coef", type=float, default=0.01,
                   help="PPO factor-head entropy at epoch 1.")
    g.add_argument("--entropy-coef-final", type=float, default=0.001,
                   help="PPO factor-head entropy at the last epoch; decays "
                        "linearly from --entropy-coef.")
    g.add_argument("--entropy-coef-unmerge", type=float, default=0.05,
                   help="PPO binary-head entropy, held CONSTANT (no decay) — it "
                        "protects the unmerge head from extinction.")
    g.add_argument("--logit-cap", type=float, default=4.0,
                   help="PPO only. Bounds actor logits via C*tanh(logit/C). "
                        "This is the entropy-collapse guard; 0 disables it.")
    g.add_argument("--bandit-epsilon", type=float, default=0.3,
                   help="Bandit exploration at epoch 1; decays linearly.")
    g.add_argument("--bandit-epsilon-final", type=float, default=0.05)
    g.add_argument("--bandit-warm-epochs", type=int, default=10,
                   help="Q-head warm-start passes over the FIT fold's cached "
                        "cells before any rollout. 0 disables.")
    g.add_argument("--missing", choices=["penalty", "skip"], default="penalty",
                   help="what a policy earns during TRAINING for a cell with no "
                        "row in the cache. Default 'penalty' pays "
                        "--compile-failure-penalty: after exhaustive collection "
                        "an absent row means the cell failed, not that it was "
                        "never tried, so the agent should learn to avoid it. "
                        "'skip' drops the sample instead.")
    g.add_argument("--compile-failure-penalty", type=float, default=-0.161,
                   help="The ONE default here that is deliberately not "
                        "train.py's (-0.25). It must match the value the CACHE "
                        "was built with, not the current CLI default: "
                        "run_sweep_1 was measured at -0.161, and the cells "
                        "holding it are indistinguishable from real "
                        "measurements without the migration block.")
    g.add_argument("--score-missing", choices=["penalty", "exclude"],
                   default="penalty",
                   help="how an absent cell is scored at EVALUATION time. "
                        "Default matches --missing so the policy is scored "
                        "against the rule it was trained under.")

    g = p.add_argument_group("score-ckpt")
    g.add_argument("--checkpoint", type=Path, default=None)
    g.add_argument("--split-seed", type=int, default=42)
    g.add_argument("--val-ratio", type=float, default=0.15)
    g.add_argument("--test-ratio", type=float, default=0.15)

    p.add_argument("--threads", type=int, default=1,
                   help="torch intra-op threads (default 1). This runner is "
                        "single-process and single-threaded by construction; "
                        "the only parallelism torch can introduce is inside a "
                        "single op, and on 93-wide MLPs at batch 32 the thread "
                        "pool costs more than it saves. Pinning it to 1 also "
                        "makes results bit-reproducible across machines. If "
                        "wall-clock ever matters, parallelise across PROCESSES "
                        "(separate agents or fold ranges) — folds and seeds are "
                        "independent, so there is nothing to share and no race "
                        "to have.")

    return p


def main() -> None:
    p = build_parser()
    args = p.parse_args()
    if args.mode == "score-ckpt" and args.checkpoint is None:
        p.error("--mode score-ckpt requires --checkpoint")
    torch.set_num_threads(max(1, args.threads))

    data = load_run(args.run_dir, args.deadzone, args.labels)
    log.info("Loaded %d loops / %d benchmarks | %d labelled | "
             "normalizer %s | post_features %d",
             len(data["loops"]), len(data["benchmarks"]),
             data["n_labelled_loops"],
             "fitted" if data["normalizer_fitted"] else "IDENTITY (not fitted)",
             len(data["postf"]))
    if data["n_dropped_invalid"]:
        log.info("  dropped %d cell(s) whose factor exceeds a known trip count "
                 "(MIRROR: label_loops.valid_factors) — unreachable by any "
                 "policy, so they must not enter the ceiling",
                 data["n_dropped_invalid"])
    if data["n_labelled_loops"] != data["n_label_rows"]:
        log.warning("  WARNING: loop_labels.csv has %d labelable rows but only "
                    "%d match a loop here (dedup dropped %d). The scored "
                    "population is NOT the published one — every denominator "
                    "differs. Reconcile before using these numbers.",
                    data["n_label_rows"], data["n_labelled_loops"],
                    data["n_dropped_dedup"])
    # post_features are stored ALREADY NORMALISED under the normalizer that was
    # live at collection time. make_state falls back to s1 when one is absent,
    # which is legal but changes what the factor head sees on the unmerge
    # branch — so the coverage is stated, never left implicit.
    n_unmerge_states = len(data["loops"])
    if len(data["postf"]) < n_unmerge_states:
        log.info("  post_features cover %d/%d loops — the rest fall back to the "
                 "pre-unmerge state on the unmerge branch (MIRROR: "
                 "adapt_eval.make_state)", len(data["postf"]), n_unmerge_states)
    log.info("")

    if args.mode == "cv":
        run_cv(data, args)
    else:
        run_score_ckpt(data, args)


if __name__ == "__main__":
    main()
