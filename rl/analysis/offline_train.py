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
import math
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import torch                                                    # noqa: E402

from offline_data import (_RULE, IDX_TRIP_COUNT, IDX_TRIP_KNOWN,  # noqa: E402
                          NOOP, always_noop_picks, benchmark_dominant_picks,
                          best_constant_factor,
                          fingerprint, format_confusion, format_report, grouped_kfold,
                          holdout_split, fingerprint, labelled_loops, load_run, loops_for,
                          marginal_picks, marginal_ranking, oracle_picks, pairwise_accuracy,
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


def _modal_factor(rows: list, key: str, n_key: str) -> int:
    """
    Most common constant factor across folds, skipping folds whose probe
    population was empty.

    Two things it must not do. It must not count the 0 placeholder a fold with
    no probe loops writes — 0 is not a factor, and one such fold would vote in a
    value FACTOR_VALUES does not contain. And it must not iterate a set: set
    order varies with hash randomisation, so a tie between two equally common
    factors would print differently run to run (the bug already fixed once in
    benchmark_dominant_picks). sorted() makes ties break to the lower factor,
    matching the tie convention used everywhere else.
    """
    vals = [r[key] for r in rows if r.get(n_key, 0) > 0 and r[key]]
    if not vals:
        return 0
    return max(sorted(set(vals)), key=vals.count)


def _nanmean(xs: list) -> float:
    """Mean over the finite entries. A fold whose held-out slice happens to hold
    no transform-labelled loop yields NaN, and st.mean would then poison the
    whole summary line — the same trap the curve's accuracy column hit."""
    ok = [x for x in xs if x == x]
    return st.mean(ok) if ok else float("nan")


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

    Returns (unmerge, factor, factor_idx, log_p1, log_p2, mask, s1, s2,
             head_action, head_mask).

    head_action is the index into the FIRST head — the unmerge bit for the
    2-head agents, the CATEGORY for the 3-way one. RolloutEntry.action1 must be
    that, not the unmerge bit: a 3-way head gathered at {0,1} could never be
    trained on unmerge_unroll at all.

    head_mask is the collection-time mask over that first head (None for the
    2-head agents, whose unmerge bit is never masked). PPO's ratio is only a
    ratio if the update reapplies it.
    """
    from adapt_eval import make_state
    from agent import FACTOR_VALUES

    raw = loop["pre_features_raw"]
    known, trip = raw[IDX_TRIP_KNOWN] > 0.5, int(raw[IDX_TRIP_COUNT])

    if hasattr(agent, "select_category"):
        # 3-way head (analysis/category_agent.py). action1 is a CATEGORY index,
        # not an unmerge bit — RolloutEntry stores it unchanged, which is what
        # lets this share every other piece of the runner.
        from category_agent import NOOP, to_pipeline_action
        s1, _ = make_state(loop, normalizer, postf, 0)
        cat, log_p1, cat_mask = agent.select_category(
            s1, trip_known=known, trip_count=trip, greedy=greedy)
        if cat == NOOP:
            # No factor decision exists. s2 = s1 and the factor index is a
            # placeholder the update masks out via factor_active=False.
            return 0, 1, 0, log_p1, torch.zeros(()), None, s1, s1, cat, cat_mask
        unmerge, _ = to_pipeline_action(cat, 1)
        _, s2 = make_state(loop, normalizer, postf, unmerge)
        f_idx, log_p2, mask = agent.select_factor(
            s2, cat, trip_known=known, trip_count=trip, greedy=greedy)
        u, f = to_pipeline_action(cat, FACTOR_VALUES[f_idx])
        return u, f, f_idx, log_p1, log_p2, mask, s1, s2, cat, cat_mask

    s1, _ = make_state(loop, normalizer, postf, 0)
    unmerge, log_p1 = agent.select_unmerge(s1, greedy=greedy)
    _, s2 = make_state(loop, normalizer, postf, unmerge)
    factor_idx, log_p2, mask = agent.select_factor(
        s2,
        trip_known=known,
        trip_count=trip,
        loop_idx=loop["loop_idx"],
        greedy=greedy,
    )
    return (int(unmerge), FACTOR_VALUES[factor_idx], factor_idx,
            log_p1, log_p2, mask, s1, s2, int(unmerge), None)


CLIP_FLOOR = -1.0


def reshape_floor(r: float, floor_penalty) -> float:
    """
    Rollout-time reward SHAPING at the clip floor. Training only.

    train.py:1443 clips every measured reward at -1.0 and also uses -1.0 as the
    compile-timeout penalty, so a cell sitting there is either a timeout or a
    slowdown of 100% or worse — 744 of 8,352 cells, and the cache keeps no
    record of which is which (`is_timeout` never reaches it; 0 of the 744 are
    in failure_keys). They are therefore not separable, and this remaps ALL of
    them.

    The hypothesis it exists to test: the floor is ~10x the size of the typical
    gain (+0.19 median win), so it may be suppressing firing far beyond what the
    real risk warrants.

    ONLY the rollout reward is remapped. Every reported number — capture,
    mean_realized, the oracle, the labels — is still computed from the true
    stored value, so a softer tail cannot manufacture performance. The oracle
    and labels are unaffected either way: both -1.0 and any softer value sit far
    below the deadzone, so a category whose best cell is at the floor is -inf
    regardless.
    """
    if floor_penalty is None or r > CLIP_FLOOR + 1e-9:
        return r
    return float(floor_penalty)


def reward_for(tables: dict, loop: dict, unmerge: int, factor: int,
               floor_penalty=None):
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
    return reshape_floor(float(cell), floor_penalty), True, True


@torch.no_grad()
@torch.no_grad()
def factor_probe_picks(agent, loops: list, normalizer, postf: dict,
                       labels: dict) -> list:
    """
    The FACTOR head alone: the category is FORCED to the truth and the policy
    chooses only the factor.

    Why this and not `capture_factor`. That one restricts to the loops the model
    happened to get category-right, so the population MOVES between runs — a head
    that shifts category accuracy is then scored on a different and
    differently-hard subset, and two runs are not comparable. This probe covers
    every loop whose true category is a transform, whatever the category head
    did, so the denominator is a property of the fold and not of the model.

    Category agents only. The 2-head path's factor mask does not exclude factor 1
    from the unroll branch, so a probe there could answer (0,1) — the no-op —
    which is a category decision leaking back into a factor measurement. Raising
    beats reporting a number that means something different per agent.
    """
    from adapt_eval import make_state
    from agent import FACTOR_VALUES

    if not hasattr(agent, "select_category"):
        raise NotImplementedError(
            "factor_probe_picks needs a 3-way head (--agent category or "
            "category-bandit); the 2-head factor mask admits (0,1).")
    from category_agent import UNMERGE_UNROLL, UNROLL_ONLY, to_pipeline_action

    out = []
    for l in loops:
        key = (l["benchmark_name"], l["loop_idx"])
        truth = labels.get(key)
        if truth is None or truth == "noop":
            continue
        raw = l["pre_features_raw"]
        known, trip = raw[IDX_TRIP_KNOWN] > 0.5, int(raw[IDX_TRIP_COUNT])
        cat = UNMERGE_UNROLL if truth == "unmerge_unroll" else UNROLL_ONLY
        unmerge, _ = to_pipeline_action(cat, 1)
        # s2 must be built from the FORCED branch, exactly as act() builds it
        # from the chosen one — the unmerge branch uses post-unmerge features.
        _, s2 = make_state(l, normalizer, postf, unmerge)
        f_idx, _, _ = agent.select_factor(s2, cat, trip_known=known,
                                          trip_count=trip, greedy=True)
        out.append((key[0], key[1], to_pipeline_action(cat, FACTOR_VALUES[f_idx])))
    return out


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

    u, f, f_idx, lp1, lp2, mask, s1, s2, head, head_mask = act(
        agent, loop, data["normalizer"], data["postf"], greedy=False)
    # floor_penalty applies to the ROLLOUT ONLY — scoring never sees it.
    r, factor_active, measured = reward_for(data["tables"], loop, u, f,
                                            args.train_floor_penalty)
    seen = (((loop["benchmark_name"], loop["loop_idx"]), (u, f), float(r))
            if measured else None)
    if not measured:
        if args.missing == "skip":
            return None, True, None
        r = args.compile_failure_penalty
    # MIRROR: train._send_loop_result -> main's rebuild. Online, entries cross a
    # process queue as PLAIN PYTHON, so ppo_update sees graph-free constants.
    # Rebuilt here in the same shapes and dtypes: a log-prob still carrying
    # grad_fn would make the PPO ratio differentiable through the
    # COLLECTION-time policy, which is meant to be a fixed reference.
    fields = dict(
        state1=s1.detach(), state2=s2.detach(),
        action1=int(head), action2=int(f_idx),
        log_prob1=torch.tensor(float(lp1), dtype=torch.float32),
        log_prob2=torch.tensor(float(lp2), dtype=torch.float32),
        reward=float(r),
        # None on the 3-way head's no-op branch: there is no factor decision.
        # RolloutEntry treats None as all-valid, which is what the update wants
        # for an entry whose factor head is masked out anyway.
        mask2=None if mask is None else mask.detach().to(torch.bool),
        factor_active=factor_active)
    if head_mask is None:
        return RolloutEntry(**fields), (not measured), seen
    # 3-way head: carry the category mask so ppo_update can reapply it.
    from category_agent import CategoryRolloutEntry
    return (CategoryRolloutEntry(mask1=head_mask.detach().to(torch.bool),
                                 **fields),
            (not measured), seen)


def run_epoch(agent, order: list, buf, data: dict, args,
              observed: dict) -> tuple:
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

    `observed` accumulates {(bench, loop): {(unmerge, factor): reward}} across
    epochs — the cells the agent has actually paid for. It is the label source
    for the contrastive term, and keeping it separate from data["tables"] is
    what makes that term online-legitimate rather than a table lookup.

    Returns (n_absent_cells, n_updates, sum_actor_loss).
    """
    n_missing = n_updates = 0
    loss_sum = 0.0
    for l in order:
        entry, missing, seen = rollout_one(agent, l, data, args)
        n_missing += int(missing)
        if seen is not None:
            observed.setdefault(seen[0], {})[seen[1]] = seen[2]
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
    if kind in ("category", "category-bandit"):
        from category_agent import CategoryAgent, CategoryBanditAgent
        common["q_pessimism"] = args.q_pessimism
        if kind == "category-bandit":
            return CategoryBanditAgent(epsilon=args.bandit_epsilon,
                                       logit_cap=args.logit_cap, **common)
        return CategoryAgent(logit_cap=args.logit_cap,
                             entropy_coef_category=args.entropy_coef_unmerge,
                             **common)
    common.pop("q_pessimism", None)
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
    if kind in ("bandit", "category-bandit"):
        agent.epsilon = (args.bandit_epsilon
                         + frac * (args.bandit_epsilon_final - args.bandit_epsilon))
    else:
        agent.entropy_coef = (args.entropy_coef
                              + frac * (args.entropy_coef_final - args.entropy_coef))


def supcon_step(agent, fit_loops: list, observed: dict, data: dict,
                epoch: int, args) -> tuple:
    """
    One supervised-contrastive update on the category head's embedding.

    A SEPARATE optimizer step, not a term folded into ppo_update. Two reasons:
    the agents in agent.py stay untouched, and the contrastive term wants a much
    larger batch than the rollout minibatch of 8 — with three classes, batch 8
    yields roughly three same-label pairs, far too few negatives for the loss to
    mean anything. It needs states and labels, not rollout structure, so it can
    be batched independently.

    Runs --supcon-steps times per epoch. This matters more than the
    coefficient: the rollout cadence performs ~76 Q updates per epoch, so ONE
    contrastive step is 1.3% of the optimizer traffic and would barely move the
    embedding no matter how it were weighted. The two knobs are not
    interchangeable — the coefficient scales one step's gradient, the count
    decides how much of training the term actually participates in.

    Returns (mean loss over the steps taken, n_labelled) — (0.0, n) when the
    gate is closed.
    """
    from adapt_eval import make_state
    from supcon import embed, is_active, provisional_labels, supcon_loss

    labels = provisional_labels(fit_loops, observed, args.deadzone,
                                args.supcon_min_cells)
    if not is_active(epoch, labels, args):
        return 0.0, len(labels)

    pool = [l for l in fit_loops
            if (l["benchmark_name"], l["loop_idx"]) in labels]
    # States for the WHOLE labelled pool, built once per epoch; each step then
    # indexes a fresh random subset. Sampling one batch and stepping on it
    # --supcon-steps times is not N steps of contrastive learning, it is N
    # steps of memorising one batch — and with a fixed batch the negatives
    # never change, which is the one thing the loss depends on.
    all_states = torch.stack([
        make_state(l, data["normalizer"], data["postf"], 0)[0] for l in pool])
    all_y = torch.tensor(
        [labels[(l["benchmark_name"], l["loop_idx"])] for l in pool],
        dtype=torch.long)
    n_pool = all_states.size(0)
    bs = min(args.supcon_batch, n_pool)

    total, taken = 0.0, 0
    for _ in range(max(1, args.supcon_steps)):
        idx = torch.randperm(n_pool)[:bs]
        states, y = all_states[idx], all_y[idx]
        loss = args.supcon_coef * supcon_loss(embed(agent, states), y,
                                              args.supcon_temp)
        if not math.isfinite(float(loss)):
            # A non-finite loss reaches the optimizer as non-finite gradients
            # and turns every weight it touches to NaN — after which the agent
            # still runs, still reports numbers, and every one of them is
            # garbage. `float(loss) == 0.0` does NOT catch NaN. Fail loudly.
            raise AssertionError(
                f"supcon loss is {float(loss)} at epoch {epoch} — refusing to "
                f"step. A NaN here silently corrupts the run.")
        if float(loss) == 0.0:
            # This draw had no same-label pair. Resample rather than abandon
            # the epoch's remaining steps.
            continue
        agent.optimizer.zero_grad()
        loss.backward()
        if agent.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(agent._all_params, agent.max_grad_norm)
        agent.optimizer.step()
        total += float(loss)
        taken += 1
    return (total / taken if taken else 0.0), len(labels)


def factor_rank_step(agent, kind: str, fit_loops: list, observed: dict,
                     data: dict, epoch: int, args) -> tuple:
    """
    Soft-target ranking updates on the FACTOR head. See analysis/factor_rank.py.

    A separate optimizer step, for the same reasons as supcon_step: agent.py
    stays untouched and the term wants a larger batch than the rollout's 8. The
    batch is resampled every step — a fixed batch stepped N times is N steps of
    memorising one draw, which is exactly what the first supcon implementation
    did.

    Returns (mean loss, n_rankable_rows, calibration). Calibration is mean
    |Q2 - r| and is meaningful for the BANDIT only, where Q1 backs up max_f Q2
    and so depends on Q2 keeping reward scale; it is NaN for PPO, whose head
    emits logits and is not supposed to be calibrated at all.
    """
    from factor_rank import branch_rows, factor_calibration, is_active, rank_loss

    states, masks, targets, rewards = branch_rows(
        fit_loops, observed, data, args.rank_temp, args.rank_min_cells)
    if not is_active(epoch, len(states), args):
        return 0.0, len(states), float("nan")

    S, M, T = torch.stack(states), torch.stack(masks), torch.stack(targets)
    n_rows = S.size(0)
    bs = min(args.rank_batch, n_rows)

    total, taken = 0.0, 0
    for _ in range(max(1, args.rank_steps)):
        idx = torch.randperm(n_rows)[:bs]
        loss = args.rank_coef * rank_loss(
            agent.factor_actor.forward(S[idx]), T[idx], M[idx])
        if not math.isfinite(float(loss)):
            # A non-finite loss reaches the optimizer as non-finite gradients
            # and turns every weight it touches to NaN — after which the agent
            # still runs, still reports numbers, and every one of them is
            # garbage. `float(loss) == 0.0` does NOT catch NaN. Fail loudly.
            raise AssertionError(
                f"factor-rank loss is {float(loss)} at epoch {epoch} — refusing to "
                f"step. A NaN here silently corrupts the run.")
        if float(loss) == 0.0:
            continue
        agent.optimizer.zero_grad()
        loss.backward()
        if agent.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(agent._all_params, agent.max_grad_norm)
        agent.optimizer.step()
        total += float(loss)
        taken += 1

    cal = (factor_calibration(agent, S, M, torch.stack(rewards))
           if kind.endswith("bandit") else float("nan"))
    return (total / taken if taken else 0.0), n_rows, cal


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
    if args.factor_head != "mlp":
        # AFTER make_agent and BEFORE anything reads agent.optimizer: the swap
        # rebuilds the optimizer, so any reference taken earlier would be stale.
        from factor_scorer import FactorScorer, swap_factor_head
        if args.factor_head == "scorer":
            def _factory(cap, _f=args.factor_feats):
                return FactorScorer(feats=_f, logit_cap=cap)
        else:
            from factor_attn import FactorAttnActor
            def _factory(cap, _t=args.attn_tokens, _h=args.attn_heads,
                         _b=args.attn_blocks):
                return FactorAttnActor(tokens=_t, heads=_h, blocks=_b,
                                       logit_cap=cap)
        swap_factor_head(agent, _factory)

    # MIRROR: train.py:2542 — the bandit warm-starts its Q-heads on cached
    # cells before any rollout. Restricted to FIT loops: build_warm_start_entries
    # takes the assignment list, so passing only this fold's training loops is
    # what keeps the held-out fold out of the warm start.
    n_ws_cells = 0
    # NOT applied to the category agents: build_warm_start_entries emits
    # action1 as an unmerge BIT, which a 3-way head would read as a category
    # index — silently training "unmerge=1" as "unroll_only".
    if kind == "bandit" and args.bandit_warm_epochs > 0:
        from train import build_warm_start_entries
        ws, n_ws_cells, _ = build_warm_start_entries(
            fit_loops, data["rewards"], data["postf"], data["normalizer"])
        if n_ws_cells:
            agent.warm_start(ws, args.bandit_warm_epochs)

    # Fit BEFORE any rollout — for the bandit this is the warm start alone.
    # Recorded as epoch 0 so the curve shows how much of the final fit was
    # already there before a single gradient step from experience. Without it,
    # "fit is 69%" cannot be attributed between the cached-cell regression and
    # the RL loop, and the two imply completely different next steps.
    pre = score_decisions(greedy_picks(agent, fit_loops, data["normalizer"],
                                       data["postf"]),
                          data["tables"], data["labels"], args.deadzone,
                          _mr(args))

    best_score, best_snap, best_epoch, since = -float("inf"), _snapshot(agent), 0, 0
    bestfit_score, bestfit_snap, bestfit_epoch = -float("inf"), _snapshot(agent), 0
    # Cells the agent has paid for, accumulated across epochs. Grows toward the
    # full table as epsilon decays, so the contrastive labels sharpen over the
    # run without ever reading a cell that was not sampled.
    observed: dict = {}
    n_supcon = n_labelled = n_rankstep = 0
    n_rank = 0
    rk_cal = float("nan")
    n_missing = n_updates = 0
    # Same keys as the per-epoch rows below. This row is built BEFORE the loop,
    # so any field added to the loop's history entry must be mirrored here or
    # the curve CSV takes its fieldnames from an incomplete first row and dies
    # at the very end of the run.
    history = [{"epoch": 0, "hold_mean_realized": float("nan"),
                "hold_accuracy": float("nan"),
                "fit_accuracy": pre["accuracy"],
                "fit_mean_realized": pre["mean_realized"],
                "actor_loss": float("nan"),
                "supcon_loss": float("nan"), "supcon_labelled": 0,
                "rank_loss": float("nan"), "rank_rows": 0,
                "factor_calibration": float("nan")}]
    buf = RolloutBuffer(capacity=args.buffer_size)

    for epoch in range(1, args.epochs + 1):
        _schedules(agent, kind, epoch, args)
        order = list(fit_loops)
        random.shuffle(order)
        miss, ups, loss = run_epoch(agent, order, buf, data, args,
                                    observed)
        n_missing += miss
        n_updates += ups
        sc_loss, n_lab = supcon_step(agent, fit_loops, observed, data, epoch, args)
        n_supcon += int(sc_loss != 0.0)
        rk_loss, n_rank, rk_cal = factor_rank_step(
            agent, kind, fit_loops, observed, data, epoch, args)
        n_rankstep += int(rk_loss != 0.0)
        # How many loops CLEARED --supcon-min-cells. If this stays near zero the
        # term never fires and the run silently reduces to the plain agent, so
        # it is reported rather than inferred from the absence of an effect.
        n_labelled = max(n_labelled, n_lab)
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
                        "actor_loss": loss / max(ups, 1),
                        # Without these the curve cannot answer the only
                        # question that matters for the term: is the embedding
                        # separating (loss falling) and is coverage growing
                        # enough for the labels to mean anything (n_labelled).
                        "supcon_loss": sc_loss, "supcon_labelled": n_lab,
                        # rank_rows is the count of (loop, branch) pairs with
                        # enough observed factors to rank; if it stays at 0 the
                        # term never fired and nothing below it means anything.
                        "rank_loss": rk_loss, "rank_rows": n_rank,
                        "factor_calibration": rk_cal})
        # Best-FIT checkpoint, tracked independently of val. Selecting on ~17
        # held-out benchmarks is a noisy criterion, and there is no reason to
        # assume the epoch that maximises a small val slice is the one that
        # transfers — that is an assumption worth measuring, not inheriting.
        if fit_a == fit_a and fit_a > bestfit_score:
            bestfit_score, bestfit_snap, bestfit_epoch = fit_a, _snapshot(agent), epoch
        if m["mean_realized"] > best_score:
            best_score, best_snap, best_epoch, since = (
                m["mean_realized"], _snapshot(agent), epoch, 0)
        else:
            since += 1
            if args.patience and since >= args.patience:
                break

    if n_updates == 0:
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

    # THE factor measurement. Category forced to the truth, so the population is
    # fixed by the fold and the category head cannot move it — unlike
    # capture_factor, whose subset is whatever the model got category-right.
    # `probe_const` is model-independent (it depends only on tables and labels),
    # so it is the same number for every run on this fold: the bar to beat.
    probe = fin_probe = None
    if hasattr(agent, "select_category"):
        probe = score_decisions(
            factor_probe_picks(agent, fit_loops, data["normalizer"],
                               data["postf"], data["labels"]),
            data["tables"], data["labels"], args.deadzone, _mr(args))
        fin_probe = best_constant_factor(fit_loops, data["tables"],
                                         data["labels"], args.deadzone,
                                         _mr(args))

    # Three checkpoints, three selection rules. The agent is returned on the
    # val rule (unchanged default); the others are handed back so the caller can
    # score the SAME held-out loops under each and see whether the rule matters.
    final_snap = _snapshot(agent)
    if bestfit_epoch == 0:
        # fit is only sampled every --curve-every epochs, so with 0 (sampling
        # off) or a --patience break before the first sample, bestfit_snap is
        # still the RANDOM INITIALISATION. Scoring the test fold with that and
        # labelling it "bestfit" would be a plausible-looking wrong number.
        bestfit_snap = final_snap
        log.warning("    fit was never sampled (--curve-every %d, ran %d "
                    "epochs) — 'bestfit' falls back to the final epoch",
                    args.curve_every, len(history) - 1)
    snapshots = {"val": best_snap, "final": final_snap,
                 "bestfit": bestfit_snap}

    _restore(agent, best_snap)
    for mod in (agent.unmerge_actor, agent.factor_actor, agent.critic):
        mod.eval()
    return agent, {"best_epoch": best_epoch, "best_hold_mean_realized": best_score,
                   "epochs_run": len(history) - 1,   # epoch 0 is the pre-rollout probe
                   "n_missing_cells": n_missing,
                   "warm_fit_accuracy": pre["accuracy"],
                   "warm_fit_mean_realized": pre["mean_realized"],
                   "warm_fit_capture": pre["capture"],
                   "warm_fit_capture_factor": pre["capture_factor"],
                   "n_updates": n_updates, "n_warm_start_cells": n_ws_cells,
                   "n_supcon_steps": n_supcon,
                   "n_supcon_labelled": n_labelled,
                   "n_rank_steps": n_rankstep, "n_rank_rows": n_rank,
                   "factor_calibration": rk_cal,
                   "snapshots": snapshots, "bestfit_epoch": bestfit_epoch,
                   "n_observed_cells": sum(len(v) for v in observed.values()),
                   "final_fit_accuracy": fin["accuracy"],
                   "final_fit_mean_realized": fin["mean_realized"],
                   "final_fit_capture": fin["capture"],
                   "final_fit_capture_factor": fin["capture_factor"],
                   "final_fit_n_factor": fin["n_factor_loops"],
                   # nan/0 for the 2-head agents, which the probe refuses.
                   # `is not None`, not truthiness: probe is a dict and
                   # fin_probe a tuple, and an empty one of either would read as
                   # "no probe ran" when it actually means "the probe ran and
                   # found nothing" — different things, different columns.
                   "probe_capture": (probe["capture"]
                                     if probe is not None else float("nan")),
                   "probe_n": (probe["loops_with_headroom"]
                               if probe is not None else 0),
                   "probe_regress": probe["n_regress"] if probe is not None else 0,
                   "probe_const_factor": fin_probe[0] if fin_probe is not None else 0,
                   "probe_const_capture": (fin_probe[1] if fin_probe is not None
                                           else float("nan")),
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
    for _l in fingerprint(loops, args):
        log.info(_l)

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
            # Score the SAME held-out loops under each selection rule, then
            # put the agent back on the val rule so everything below is
            # unchanged. Restoring is cheap; retraining would not be.
            for rule, snap in info["snapshots"].items():
                _restore(agent, snap)
                union.setdefault((rule, seed), []).extend(
                    greedy_picks(agent, test_l, data["normalizer"],
                                 data["postf"]))
            # Factor probe on the HELD-OUT loops, on the FINAL snapshot so it
            # matches probe_capture_fit — comparing a final-epoch fit number
            # against a val-selected test number would confound the
            # architecture with the selection rule. Restored to val below, so
            # nothing downstream sees the swap.
            probe_te = probe_te_const = None
            if hasattr(agent, "select_category"):
                _restore(agent, info["snapshots"]["final"])
                probe_te = score_decisions(
                    factor_probe_picks(agent, test_l, data["normalizer"],
                                       data["postf"], data["labels"]),
                    data["tables"], data["labels"], args.deadzone, _mr(args))
                probe_te_const = best_constant_factor(
                    test_l, data["tables"], data["labels"], args.deadzone,
                    _mr(args))

            _restore(agent, info["snapshots"]["val"])
            picks = greedy_picks(agent, test_l, data["normalizer"], data["postf"])
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
                # Factor-only capture: the same ratio restricted to loops whose
                # CATEGORY was already right. `n_factor_*` is its denominator
                # size — a ratio over a handful of loops is noise, so it is
                # never reported without the count beside it.
                "capture_factor": round(m["capture_factor"], 6),
                "n_factor_loops": m["n_factor_loops"],
                "capture_factor_val": round(m_val["capture_factor"], 6),
                "mean_realized_fit": round(m_fit["mean_realized"], 6),
                "mean_realized_val": round(m_val["mean_realized"], 6),
                "mean_realized": round(m["mean_realized"], 6),
                "noop_accuracy_test": round(base["accuracy"], 6),
                "regression_rate": round(m["regression_rate"], 6),
                "loops_unmeasured": m["loops_unmeasured"],
                "n_missing_cells_in_training": info["n_missing_cells"],
                "bestfit_epoch": info["bestfit_epoch"],
                "n_supcon_steps": info["n_supcon_steps"],
                "n_supcon_labelled": info["n_supcon_labelled"],
                "n_rank_steps": info["n_rank_steps"],
                "n_rank_rows": info["n_rank_rows"],
                "factor_calibration": round(info["factor_calibration"], 6),
                # fit at the LAST epoch vs at the val-SELECTED epoch. If the
                # first keeps rising with --epochs and the second does not, the
                # limit is selection, not capacity.
                # Fit BEFORE any rollout (bandit: the warm start alone).
                # final minus this is what the RL loop actually contributed.
                "accuracy_fit_prerollout": round(info["warm_fit_accuracy"], 6),
                "mean_realized_fit_prerollout": round(
                    info["warm_fit_mean_realized"], 6),
                "capture_fit_prerollout": round(info["warm_fit_capture"], 6),
                "accuracy_fit_final": round(info["final_fit_accuracy"], 6),
                "mean_realized_fit_final": round(
                    info["final_fit_mean_realized"], 6),
                # capture_fit above is scored on the VAL-SELECTED snapshot, so it
                # reports which epoch selection landed on as much as it reports
                # factor quality — it ranged 0.04 (best_epoch 1) to 0.80
                # (best_epoch 89) across one run's folds. THIS is the column the
                # factor-ranking criterion is pre-registered against.
                "capture_fit_final": round(info["final_fit_capture"], 6),
                # THE primary criterion for any factor-head change.
                "capture_factor_fit_final": round(
                    info["final_fit_capture_factor"], 6),
                "n_factor_loops_fit_final": info["final_fit_n_factor"],
                "capture_factor_fit_prerollout": round(
                    info["warm_fit_capture_factor"], 6),
                # THE decision numbers. probe_* is the factor head alone on a
                # population the category head cannot move; probe_const_* is the
                # best single fixed factor on that same population and is
                # identical for every run on this fold.
                "probe_capture_fit": round(info["probe_capture"], 6),
                "probe_n_fit": info["probe_n"],
                "probe_regress_fit": info["probe_regress"],
                "probe_const_capture_fit": round(info["probe_const_capture"], 6),
                "probe_const_factor_fit": info["probe_const_factor"],
                # Same probe on the HELD-OUT loops, final snapshot.
                "probe_capture_test": (round(probe_te["capture"], 6)
                                       if probe_te is not None else float("nan")),
                "probe_n_test": (probe_te["loops_with_headroom"]
                                 if probe_te is not None else 0),
                "probe_regress_test": (probe_te["n_regress"]
                                       if probe_te is not None else 0),
                "probe_const_capture_test": (round(probe_te_const[1], 6)
                                             if probe_te_const is not None
                                             else float("nan")),
                "probe_const_factor_test": (probe_te_const[0]
                                            if probe_te_const is not None else 0),
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

    _pr = [(r["accuracy_fit_prerollout"], r["accuracy_fit_final"])
           for r in fold_rows]
    log.info("\n  fit accuracy BEFORE any rollout %.1f%% -> after training "
             "%.1f%%  (RL contributed %+.1fpp).\n  For the bandit the first "
             "figure is the warm start alone; if the two are close, the\n"
             "  cached-cell regression is the model and the rollout loop is "
             "decorative.",
             100 * st.mean([a for a, _ in _pr]),
             100 * st.mean([b for _, b in _pr]),
             100 * st.mean([b - a for a, b in _pr]))

    _ff = [(r["accuracy_fit"], r["accuracy_fit_final"]) for r in fold_rows]
    log.info("\n  fit accuracy at the SELECTED epoch %.1f%% vs at the FINAL "
             "epoch %.1f%%.\n  If the final keeps rising with --epochs while "
             "the selected does not, the binding\n  constraint is val-based "
             "selection, not capacity or training length.",
             100 * st.mean([a for a, _ in _ff]),
             100 * st.mean([b for _, b in _ff]))

    # Factor-only capture on the FIT split at the final epoch. `capture_fit_final`
    # answers "how much headroom did the policy realise"; this answers "given the
    # category was right, how good was the factor" — the only one of the two that
    # a factor-head change can be judged on. nan when no fold had a
    # category-correct headroom loop, which would itself be the finding.
    _cf = [(r["capture_factor_fit_final"], r["capture_fit_final"],
            r["n_factor_loops_fit_final"]) for r in fold_rows
           if r["capture_factor_fit_final"] == r["capture_factor_fit_final"]]
    if _cf:
        log.info("\n  FIT capture %.1f%% overall vs %.1f%% on the loops whose "
                 "CATEGORY was right\n  (%.0f loops per fold on average). The "
                 "second number is the factor's own score:\n  category errors "
                 "are divided out, so a factor-head change moves it and a\n"
                 "  category-head change does not. The heuristic scores 5.9%% "
                 "here.",
                 100 * st.mean([c for _, c, _ in _cf]),
                 100 * st.mean([f for f, _, _ in _cf]),
                 st.mean([n for _, _, n in _cf]))

    _pb = [r for r in fold_rows if r["probe_n_fit"] > 0
           and r["probe_capture_fit"] == r["probe_capture_fit"]]
    if _pb:
        log.info(
            "\n==========================================================\n"
            "  FACTOR HEAD ALONE — category forced to the truth\n"
            "==========================================================\n"
            "  %.0f fit / %.0f held-out loops per fold, fixed by the fold and\n"
            "  NOT by the model — identical in every run on the same split.\n"
            "                              FIT       HELD-OUT\n"
            "    learned factor        %7.1f%%      %7.1f%%\n"
            "    best constant factor  %7.1f%%      %7.1f%%    (f=%d / f=%d)\n"
            "    oracle                  100.0%%        100.0%%\n"
            "    loops made slower     %8.0f     %8.0f\n"
            "  The constant row is the bar. A learned factor that does not clear\n"
            "  it is not a factor model, it is an expensive constant — the same\n"
            "  reading always-no-op forces on the category numbers. Both\n"
            "  constant rows are model-independent: if they differ between two\n"
            "  runs on the same split, the comparison is invalid.",
            st.mean([r["probe_n_fit"] for r in _pb]),
            st.mean([r["probe_n_test"] for r in _pb]),
            # _nanmean throughout, including the fit columns that _pb has
            # already filtered. Two different reducers over columns that must be
            # read side by side is how one of them silently becomes -inf or NaN
            # while the other looks fine.
            100 * _nanmean([r["probe_capture_fit"] for r in _pb]),
            100 * _nanmean([r["probe_capture_test"] for r in _pb]),
            # ORDER MATTERS and an arity check does not catch it: both captures
            # come first, then both factor indices, matching the row layout
            # "%.1f%% %.1f%% (f=%d / f=%d)".
            100 * _nanmean([r["probe_const_capture_fit"] for r in _pb]),
            100 * _nanmean([r["probe_const_capture_test"] for r in _pb]),
            _modal_factor(_pb, "probe_const_factor_fit", "probe_n_fit"),
            _modal_factor(_pb, "probe_const_factor_test", "probe_n_test"),
            st.mean([r["probe_regress_fit"] for r in _pb]),
            st.mean([r["probe_regress_test"] for r in _pb]))

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
    rules = ("val", "final", "bestfit")
    pooled_by_rule = {
        r: [score_decisions(union[(r, s_)], data["tables"], data["labels"],
                            args.deadzone, _mr(args))
            for s_ in sorted({k[1] for k in union})]
        for r in rules}
    pooled = pooled_by_rule["val"]

    log.info("\n" + "=" * 92)
    log.info("  RESULTS — %d loops, every one predicted by a model that never "
             "saw it", pooled[0]["loops"])
    log.info("  (union of %d held-out folds; baselines scored on the same "
             "population)", args.folds)
    log.info("=" * 92)
    log.info(table_header())
    seeds_sorted = sorted({k[1] for k in union})
    _RULE_NOTE = {"val": "epoch chosen on the val slice (~17 benchmarks)",
                  "final": "last epoch, no selection at all",
                  "bestfit": "epoch with the highest FIT accuracy"}
    for r in rules:
        for s_, m in zip(seeds_sorted, pooled_by_rule[r]):
            log.info(table_row(f"{r:<8} seed {s_}", m))
        log.info(_RULE)
    log.info("  Selection rules, same training runs and same held-out loops:")
    for r in rules:
        log.info("    %-8s %s", r, _RULE_NOTE[r])
    log.info("  If they agree, selection is not what limits transfer.")
    _baseline_rows(data, loops, folds, args)

    log.info("\n  Read across the row, not down the column: 'capture' means "
             "nothing without the")
    log.info("  oracle's 100%% and always-no-op's 0%% on the same table. "
             "'mean' is the one number")
    log.info("  directly comparable to doing nothing, which scores exactly "
             "+0.0000.")
    log.info("\n  confusion, %s VAL-selected, init seed %d "
             "(rows = truth, cols = predicted)", args.agent, seeds_sorted[0])
    log.info(format_confusion(pooled[0]))
    for s_, mm in zip(seeds_sorted, pooled):
        acc2, n2 = pairwise_accuracy(mm)
        log.info("  seed %d | no-op vs unmerge+unroll alone: %.1f%% of %d loops"
                 "  (chance = 50%%)", s_, 100 * acc2, n2)
    log.info("  Those two categories are ~86%% of the population. A 3-way "
             "accuracy that\n  looks reasonable can still sit at chance on the "
             "call that decides most loops.")

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
    for r in ("final", "bestfit"):
        _spread([m["mean_realized"] for m in pooled_by_rule[r]],
                f"  same, {r}-selected", "")
    fold_means = [st.mean([m["mean_realized"] for m in per_fold_scores[s]])
                  for s in range(args.seeds)]
    _spread(fold_means, "fold means, per init seed", "")
    for s in range(args.seeds):
        vals = [m["mean_realized"] for m in per_fold_scores[s]]
        _spread(vals, f"across folds (seed {args.base_seed + s})",
                "data heterogeneity")

    if args.csv_out:
        write_csv(args.csv_out, fold_rows)
        log.info("\n  per-fold: %s", args.csv_out)
    if args.curve_out and curve_rows:
        write_csv(args.curve_out, curve_rows)
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

def write_csv(path, rows: list) -> None:
    """
    Write rows whose schemas may differ, using the UNION of keys.

    csv.DictWriter takes its fieldnames from whatever you hand it, so deriving
    them from rows[0] means one row with an extra key raises at write time —
    after the whole run has completed. Union + restval="" degrades to a blank
    cell instead of throwing away the work.
    """
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)


def warn_ignored_flags(args) -> None:
    """
    Flags the chosen agent silently ignores. Emitted ONCE at startup: these are
    properties of the invocation, and printing them per fold buries the results
    table. A silently-ignored flag costs a whole run to discover.
    """
    if args.agent.startswith("category") and args.bandit_warm_epochs > 0:
        log.warning("NOTE --bandit-warm-epochs %d is IGNORED for %s: "
                    "build_warm_start_entries encodes\n     action1 as an "
                    "unmerge BIT, which a 3-way head reads as a category index. "
                    "This agent\n     never warm-starts, so setting it to 0 "
                    "changes nothing.", args.bandit_warm_epochs, args.agent)
    if args.supcon_coef > 0 and args.supcon_warmup >= args.epochs:
        log.warning("NOTE --supcon-warmup %d >= --epochs %d: the contrastive "
                    "term can never\n     activate.", args.supcon_warmup,
                    args.epochs)
    if args.train_floor_penalty is not None:
        log.warning("NOTE --train-floor-penalty %.3f remaps the -1.0 clip floor in "
                    "the ROLLOUT reward\n     only. Every REPORTED figure — capture, "
                    "mean_realized, the oracle — still uses the\n     true stored "
                    "-1.0, so this cannot manufacture performance. The 744 cells at\n     the floor are timeouts AND >=100%% slowdowns mixed; the cache cannot\n     separate them, so ALL are remapped.", args.train_floor_penalty)
    if args.factor_head == "attn":
        # Capacity is NOT held constant here: removing Linear(128,64) takes out
        # 8,256 parameters and an attention block adds ~1,100. Printed once so
        # the change is a stated caveat rather than a hidden confound.
        from factor_attn import param_delta
        base, att, pct = param_delta(args.attn_tokens, args.attn_heads,
                                     args.attn_blocks, args.logit_cap)
        log.info("NOTE factor head is ATTENTION: %d params vs the dense head's "
                 "%d (%+.1f%%).\n     The dense hidden layer is REMOVED, so a "
                 "negative result is confounded with the\n     capacity drop; "
                 "only a positive result is clean.", att, base, pct)
    if args.factor_head != "attn" and (args.attn_tokens != 8
                                       or args.attn_heads != 4
                                       or args.attn_blocks != 1):
        log.warning("NOTE --attn-* are IGNORED with --factor-head %s.",
                    args.factor_head)
    if args.factor_head == "mlp" and args.factor_feats != "basic":
        log.warning("NOTE --factor-feats %s is IGNORED with --factor-head mlp: "
                    "the pipeline's\n     FactorActor takes no factor features "
                    "at all — the factor is an output index.\n     Pass "
                    "--factor-head scorer to make it mean anything.",
                    args.factor_feats)


def build_parser() -> argparse.ArgumentParser:
    """
    Separate from main() so the tests can construct args from the REAL defaults
    instead of a hand-written Namespace that drifts the moment a flag changes.
    """
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--mode", choices=["cv", "score-ckpt"], default="cv")
    p.add_argument("--agent",
                   choices=["ppo", "bandit", "category", "category-bandit"],
                   default="bandit",
                   help="'category*' use the 3-way head in "
                        "analysis/category_agent.py — an isolated test of the "
                        "2-head no-op defect. Nothing in the pipeline changes.")
    p.add_argument("--supcon-coef", type=float, default=0.0,
                   help="Weight on the supervised-contrastive term applied to "
                        "the CATEGORY head's embedding. 0 disables it. Labels "
                        "come from cells the agent has OBSERVED, never from the "
                        "reward table, so the term stays online-legitimate.")
    p.add_argument("--supcon-warmup", type=int, default=15,
                   help="Epoch floor before the term activates. Early on almost "
                        "every loop is provisionally no-op (the no-op is free), "
                        "so applying it immediately would pull everything into "
                        "one cluster and teach the head to always decline.")
    p.add_argument("--supcon-min-cells", type=int, default=4,
                   help="Observed cells a loop needs before it can enter the "
                        "contrastive batch. The REAL gate: without it, unvisited "
                        "loops all report 'no-op' by default and flood that "
                        "cluster with loops carrying no evidence.")
    p.add_argument("--supcon-batch", type=int, default=128,
                   help="States per contrastive step. Deliberately much larger "
                        "than --batch-size: 3 classes at batch 8 gives only "
                        "~3 same-label pairs.")
    p.add_argument("--supcon-steps", type=int, default=8,
                   help="Contrastive steps per epoch. The binding knob: the "
                        "rollout cadence runs ~76 Q updates per epoch, so 1 "
                        "step is ~1%% of the optimizer traffic. 8 puts the term "
                        "at ~10%%, which is a starting point, not a tuned value.")
    p.add_argument("--supcon-temp", type=float, default=0.1)
    p.add_argument("--rank-coef", type=float, default=0.0,
                   help="Weight on the per-loop soft-target RANKING loss over "
                        "the factor head. 0 disables. Targets are "
                        "p_f ~ exp(r_f/temp) over the factors OBSERVED for that "
                        "branch — soft, because the median arm has 2-3 factors "
                        "within the deadzone of the best and hard argmax would "
                        "train against an arbitrary tie-break.")
    p.add_argument("--rank-temp", type=float, default=0.1,
                   help="Target temperature. At 0.1 a 0.005 reward gap stays a "
                        "near-tie while -1.0 vs +0.2 separates by e^12.")
    p.add_argument("--rank-warmup", type=int, default=15)
    p.add_argument("--rank-min-cells", type=int, default=3,
                   help="Observed factors a (loop, branch) needs before it can "
                        "be ranked. Below ~2 there is no preference to express.")
    p.add_argument("--rank-steps", type=int, default=8)
    p.add_argument("--rank-batch", type=int, default=128)
    p.add_argument("--train-floor-penalty", type=float, default=None,
                   help="Remap the -1.0 clip floor to this value in the "
                        "ROLLOUT reward only. 744 of 8,352 cells sit there "
                        "and are a mix of timeouts and >=100%% slowdowns that "
                        "the cache cannot separate. The floor is ~10x the "
                        "median win (+0.19), so it may suppress firing beyond "
                        "the real risk. REPORTING still uses the true stored "
                        "value, so this cannot manufacture performance. Off by "
                        "default.")
    p.add_argument("--attn-tokens", type=int, default=8,
                   help="--factor-head attn only. The 128-d hidden vector is "
                        "reshaped into this many tokens before self-attention; "
                        "must divide 128. NOTE the token boundaries are "
                        "arbitrary slices of hidden units, not features.")
    p.add_argument("--attn-heads", type=int, default=4,
                   help="--factor-head attn only. Must divide 128/--attn-tokens.")
    p.add_argument("--attn-blocks", type=int, default=1, choices=(1, 2),
                   help="--factor-head attn only. Self-attention blocks back to "
                        "back at constant width. No FFN inside them — an FFN is "
                        "two dense layers, which is what this head removes.")
    p.add_argument("--factor-head", choices=("mlp", "scorer", "attn"),
                   default="mlp",
                   help="mlp: the pipeline's FactorActor, where the factor is an "
                        "OUTPUT INDEX and is never an input. scorer: one shared "
                        "network scoring (loop, factor) pairs (DISCARDED "
                        "2026-08-07, kept for reproduction). attn: mlp with the "
                        "hidden dense layer net[3] replaced by a self-attention "
                        "block over the 128-d hidden vector — the factor is "
                        "STILL an output index; only how the 93 loop features "
                        "mix is changed. Parameter delta is logged at startup.")
    p.add_argument("--factor-feats", choices=("basic", "interact"),
                   default="basic",
                   help="scorer only. basic: 5 intrinsic factor features "
                        "(f/10, log f, 1/f, is_pow2, f==1). interact: basic plus "
                        "f x {loopSize, numMemoryInsts, numComputeInsts} — the "
                        "body-growth interaction. No trip-count features: it is "
                        "unknown for most loops.")
    p.add_argument("--q-pessimism", type=float, default=0.0,
                   help="category-bandit only: subtract this from the "
                        "BOOTSTRAPPED transform Q-targets. The no-op target is "
                        "exact (0.0) while the transform arms bootstrap through "
                        "a max, so >0 offsets the max-of-noise advantage the "
                        "wider unmerge branch gets.")
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
    if args.mode == "score-ckpt" and args.agent.startswith("category"):
        # adapt_eval.fresh_agent only knows the 2-head agents; it would build a
        # 2-way head and fail on a size mismatch inside load(). Refuse clearly
        # instead. The existing checkpoints are all 2-head anyway.
        p.error("--mode score-ckpt does not support the 3-way head: "
                "fresh_agent builds a 2-head agent and the checkpoint's "
                "category layer is 3-wide. Use --mode cv.")
    torch.set_num_threads(max(1, args.threads))
    if args.supcon_coef > 0 and not args.agent.startswith("category"):
        # The term clusters THREE categories on the first head's embedding. The
        # 2-head agents' first head is a binary unmerge bit — the labels would
        # not correspond to anything it predicts.
        p.error("--supcon-coef needs a 3-way head: use --agent category or "
                "category-bandit")

    warn_ignored_flags(args)
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
