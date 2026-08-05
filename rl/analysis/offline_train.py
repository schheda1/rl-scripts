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
def collect(agent, order: list, data: dict, args):
    """
    One epoch of on-policy rollouts, rewards read from the table.

    (buffer, n_absent_cells). Runs under no_grad: the stored log-probs are
    constants — the online pipeline gets that for free by shipping floats
    through a process queue — and every gradient comes from ppo_update's own
    forward passes.
    """
    from agent import RolloutBuffer, RolloutEntry

    normalizer, postf, tables = data["normalizer"], data["postf"], data["tables"]
    buf = RolloutBuffer(capacity=len(order) + 1)
    n_missing = 0
    for l in order:
        u, f, f_idx, lp1, lp2, mask, s1, s2 = act(
            agent, l, normalizer, postf, greedy=False)
        r, factor_active, measured = reward_for(tables, l, u, f)
        if not measured:
            n_missing += 1
            if args.missing == "skip":
                continue
            r = args.compile_failure_penalty
        # MIRROR: train._send_loop_result -> main's rebuild. Online, entries
        # cross a process queue as PLAIN PYTHON, so ppo_update sees graph-free
        # constants. Rebuilt here in the same shapes and dtypes: a log-prob that
        # still carried grad_fn would make the PPO ratio differentiable through
        # the COLLECTION-time policy, which is meant to be a fixed reference.
        buf.append(RolloutEntry(
            state1=s1.detach(), state2=s2.detach(),
            action1=int(u), action2=int(f_idx),
            log_prob1=torch.tensor(float(lp1), dtype=torch.float32),
            log_prob2=torch.tensor(float(lp2), dtype=torch.float32),
            reward=float(r),
            mask2=mask.detach().to(torch.bool),
            factor_active=factor_active))
    return buf, n_missing


def _snapshot(agent) -> dict:
    return {n: copy.deepcopy(m.state_dict()) for n, m in
            (("u", agent.unmerge_actor), ("f", agent.factor_actor),
             ("c", agent.critic))}


def _restore(agent, snap: dict) -> None:
    agent.unmerge_actor.load_state_dict(snap["u"])
    agent.factor_actor.load_state_dict(snap["f"])
    agent.critic.load_state_dict(snap["c"])


def train_agent(kind: str, fit_loops: list, hold_loops: list, data: dict,
                args, seed: int):
    """
    Train from scratch on fit_loops; early-stop on hold_loops.

    Selection metric is MEAN REALIZED REWARD over every scored held-out loop.
    Not capture, and not accuracy: capture ignores loops with no headroom, so a
    policy that fires destructively on them scores well on it; accuracy weights a
    +0.0001 loop the same as a +0.99 one. Mean realized is also the number that
    compares directly against always-no-op's exact 0.0, which is the bar that
    matters on this population.
    """
    from agent import Agent, BanditAgent

    torch.manual_seed(seed)
    random.seed(seed)

    kw = dict(device=DEVICE, lr=args.lr, batch_size=args.batch_size,
              weight_decay=args.weight_decay, max_grad_norm=args.max_grad_norm)
    if kind == "bandit":
        agent = BanditAgent(epsilon=args.epsilon, **kw)
    else:
        agent = Agent(entropy_coef=args.entropy_coef, logit_cap=args.logit_cap,
                      clip_eps=args.clip_eps, **kw)

    normalizer, postf, tables = data["normalizer"], data["postf"], data["tables"]
    best_score, best_snap, best_epoch, since = -float("inf"), _snapshot(agent), 0, 0
    n_missing = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        order = list(fit_loops)
        random.shuffle(order)
        buf, n_miss_epoch = collect(agent, order, data, args)
        n_missing += n_miss_epoch
        if len(buf) == 0:
            continue
        stats = agent.ppo_update(buf)

        m = score_decisions(greedy_picks(agent, hold_loops, normalizer, postf),
                            tables, data["labels"], args.deadzone, _mr(args))
        history.append({"epoch": epoch, "hold_mean_realized": m["mean_realized"],
                        "hold_accuracy": m["accuracy"],
                        "actor_loss": stats.get("actor_loss", float("nan"))})
        if m["mean_realized"] > best_score:
            best_score, best_snap, best_epoch, since = (
                m["mean_realized"], _snapshot(agent), epoch, 0)
        else:
            since += 1
            if args.patience and since >= args.patience:
                break

    if not history:
        # Every epoch produced an empty buffer, so the returned agent is the
        # random initialization. Silently reporting its score as a trained
        # result would be the worst outcome here.
        log.warning("    WARNING: no training samples in any epoch — the agent "
                    "is UNTRAINED. Check --missing and the table's coverage.")

    _restore(agent, best_snap)
    for mod in (agent.unmerge_actor, agent.factor_actor, agent.critic):
        mod.eval()
    return agent, {"best_epoch": best_epoch, "best_hold_mean_realized": best_score,
                   "epochs_run": len(history), "n_missing_cells": n_missing,
                   "history": history}


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def run_cv(data: dict, args) -> None:
    loops = labelled_loops(data)
    benches = sorted({l["benchmark_name"] for l in loops})
    folds = grouped_kfold(benches, args.folds, args.fold_seed)

    log.info("Grouped %d-fold over %d benchmarks (%d labelled loops), "
             "%d init seed(s) — %d runs total",
             args.folds, len(benches), len(loops), args.seeds,
             args.folds * args.seeds)
    log.info("agent=%s  epochs<=%d  patience=%d  missing=%s\n",
             args.agent, args.epochs, args.patience, args.missing)

    union: dict = {}          # seed -> list of held-out picks
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
            m = score_decisions(picks, data["tables"], data["labels"],
                                args.deadzone, _mr(args))
            per_fold_scores[s].append(m)
            log.info("  fold %d/%d seed %d | test %3d loops / %2d benches | "
                     "acc %5.1f%%  capture %6.1f%%  mean %+.4f  "
                     "(best epoch %d)",
                     k + 1, args.folds, seed, len(test_l), len(te_b),
                     100 * m["accuracy"], 100 * m["capture"],
                     m["mean_realized"], info["best_epoch"])
            fold_rows.append({
                "fold": k + 1, "seed": seed, "n_test_benchmarks": len(te_b),
                "n_test_loops": len(test_l), "best_epoch": info["best_epoch"],
                "epochs_run": info["epochs_run"],
                "accuracy": round(m["accuracy"], 6),
                "capture": round(m["capture"], 6),
                "mean_realized": round(m["mean_realized"], 6),
                "regression_rate": round(m["regression_rate"], 6),
                "loops_unmeasured": m["loops_unmeasured"],
                "n_missing_cells_in_training": info["n_missing_cells"],
            })

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

def main() -> None:
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

    g = p.add_argument_group("training")
    g.add_argument("--epochs", type=int, default=60)
    g.add_argument("--patience", type=int, default=12, help="0 disables")
    g.add_argument("--lr", type=float, default=3e-4)
    g.add_argument("--batch-size", type=int, default=32)
    g.add_argument("--weight-decay", type=float, default=0.01)
    g.add_argument("--max-grad-norm", type=float, default=0.5)
    g.add_argument("--entropy-coef", type=float, default=0.01, help="PPO only")
    g.add_argument("--clip-eps", type=float, default=0.2, help="PPO only")
    g.add_argument("--logit-cap", type=float, default=0.0,
                   help="PPO only; must match the run for a loaded checkpoint")
    g.add_argument("--epsilon", type=float, default=0.1, help="bandit only")
    g.add_argument("--missing", choices=["penalty", "skip"], default="penalty",
                   help="what a policy earns during TRAINING for a cell with no "
                        "row in the cache. Default 'penalty' pays "
                        "--compile-failure-penalty: after exhaustive collection "
                        "an absent row means the cell failed, not that it was "
                        "never tried, so the agent should learn to avoid it. "
                        "'skip' drops the sample instead — use only to measure "
                        "how much those cells are driving the policy.")
    g.add_argument("--compile-failure-penalty", type=float, default=-0.161)
    g.add_argument("--score-missing", choices=["penalty", "exclude"],
                   default="penalty",
                   help="how an absent cell is scored at EVALUATION time. "
                        "Default matches --missing so the policy is scored "
                        "against the rule it was trained under. 'exclude' drops "
                        "it from performance (still counted against accuracy) — "
                        "use to measure how much a result rides on those cells.")

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
