"""
Train the unroller in ISOLATION — no category head, no coupling.

WHY
---
The category is comparatively easy to learn and the factor is not; that produced
the study's recurring signature, high category accuracy with near-zero or
negative realised performance while real headroom existed. Every attempt to fix
the factor so far was made INSIDE the coupled agent, where the factor head only
trains on the (loop, branch) pairs the category head chose to fire on. PPO's
deployed model fires on ~19% of unmerge loops, so "the factor head is bad" has
always carried that confound.

THE SETUP
---------
444 loops x 2 states = 888 samples. The pre-unmerge state and the post-unmerge
state are simply two states to the unroller, each with its own measured reward
per factor. Sample a factor, look up the reward, regress. That is all.

HOW TO READ IT
--------------
~21k parameters against 888 states x 10 targets: this head can very nearly
memorise the table. A high FIT number is a sanity check that training works, not
a finding. The informational content is the HELD-OUT number against its constant
bar.

  held-out clears its bar  -> the factor is learnable and transfers.
  held-out below, fit high -> learnable in-sample, does not transfer. The
                              strongest form of the difficulty claim: no
                              category head, no coupling, many epochs, near
                              complete coverage, and it still does not carry.
  fit below its bar too    -> the features do not determine the factor at all.

A SEPARATE SCRIPT, DELIBERATELY
-------------------------------
Not an --agent in offline_train. That runner assumes a two-stage decision in
`act`, `rollout_one`, `RolloutEntry` and `ppo_update`; threading a head with no
category head through it would mean guarding six code paths that currently
produce every validated result in the study. Scoring is reused from
offline_data, so the numbers stay comparable.
"""

import argparse
import logging
import random
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import torch                                                     # noqa: E402
import torch.nn.functional as F                                  # noqa: E402

from adapt_eval import make_state                                # noqa: E402
from agent import FACTOR_VALUES, N_FACTORS, FactorActor          # noqa: E402
from factor_rank import rank_loss                                # noqa: E402
from category_agent import (UNMERGE_UNROLL, UNROLL_ONLY,         # noqa: E402
                            category_factor_mask)
from offline_data import (IDX_TRIP_COUNT, IDX_TRIP_KNOWN, NOOP,  # noqa: E402
                          best_constant_factor,
                          labelled_loops, load_run, loops_for,
                          oracle_of_gated, score_decisions)
from offline_train import write_csv                              # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
log = logging.getLogger("factor_only")

DEVICE = torch.device("cpu")

# The two state populations, and the CATEGORY whose mask each one uses. u=0 is
# scored under UNROLL_ONLY, which excludes factor 1 — (0,1) IS the no-op, not a
# factor choice. u=1 keeps factor 1: unmerging alone is a real transform.
# MIRROR: category_agent.category_factor_mask:74-88, imported not re-derived.
BRANCHES = ((0, UNROLL_ONLY, "pre-unmerge "), (1, UNMERGE_UNROLL, "post-unmerge"))


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class FactorOnly:
    """`factor_actor` and its optimizer. Nothing else exists."""

    def __init__(self, lr: float, weight_decay: float, max_grad_norm: float,
                 epsilon: float, batch_size: int, K: int,
                 loss_mode: str = "row", rank_temp: float = 0.1,
                 row_min_cells: int = 2, head: str = "mlp",
                 feats: str = "basic") -> None:
        if head == "scorer":
            # Shares weights across the ten factors instead of learning ten
            # independent output rows. Drop-in: it subclasses FactorActor and
            # overrides only forward, so select/update/probe_topk are unchanged.
            from factor_scorer import FactorScorer
            self.factor_actor = FactorScorer(feats=feats, logit_cap=0.0).to(DEVICE)
        else:
            self.factor_actor = FactorActor(logit_cap=0.0).to(DEVICE)
        self.head, self.feats = head, feats
        self.loss_mode = loss_mode
        if loss_mode == "row" and rank_temp <= 0:
            # r/0 is +-inf, and a softmax over a vector holding both +inf and
            # -inf is NaN, which would poison every weight in one step and show
            # up only as a silent flatline in the loss column.
            raise ValueError(f"--rank-temp must be > 0, got {rank_temp}")
        self.rank_temp = rank_temp
        self.row_min_cells = row_min_cells
        self.epsilon = epsilon
        self.max_grad_norm = max_grad_norm
        self.batch_size, self.K = batch_size, K
        # Set by update() so the epoch line can show whether the row loss is
        # actually being fed. Early on most rows have too few observed cells to
        # rank and the loss sees nothing; if that never changes the run is
        # meaningless and the curve is the only place it shows.
        self.last_rankable = float("nan")
        self.last_cells = float("nan")
        decay, no_decay = [], []
        for p in self.factor_actor.parameters():
            (decay if p.ndim >= 2 else no_decay).append(p)
        self._all_params = decay + no_decay
        self.optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": weight_decay},
             {"params": no_decay, "weight_decay": 0.0}], lr=lr)

    def select(self, state: torch.Tensor, mask: torch.Tensor,
               greedy: bool = False) -> int:
        """Index into FACTOR_VALUES. Epsilon-greedy over the LEGAL factors."""
        legal = [i for i, v in enumerate(mask) if v]
        if not legal:
            raise AssertionError("no legal factor for this state")
        if not greedy and random.random() < self.epsilon:
            return random.choice(legal)
        with torch.no_grad():
            q = self.factor_actor.forward(state.unsqueeze(0))[0]
            q = q.masked_fill(~mask, float("-inf"))
        return int(q.argmax())

    def _step(self, loss: torch.Tensor) -> float:
        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self._all_params, self.max_grad_norm)
        self.optimizer.step()
        return float(loss)

    def _row_batch(self, batch: list, observed: dict) -> tuple:
        """
        (states, targets, masks) for the rows in this minibatch.

        One row per buffer entry = one (loop, branch). The target is
        softmax(r / temp) over the cells MEASURED SO FAR for that row, with the
        unmeasured positions at -inf so they take exactly zero target mass.

        Two properties of this target worth being explicit about, because both
        are load-bearing:

          * softmax is shift-invariant, so adding a constant to the whole row
            changes nothing. The loop's overall level -- 74.7% of the variance
            in this cache, and invisible to argmax -- drops out for free. No
            separate centring step is needed.
          * a cell at -1.0 gets essentially zero target mass at temp=0.1, so it
            stops dominating the gradient. That is the whole reason this is not
            MSE: the large-magnitude cells here are the very negative ones, and
            no decision ever reads whether -1.0 was predicted as -0.9.

        Rows with fewer than `row_min_cells` measurements are dropped: with one
        observed cell the target is one-hot over a single unmasked position, the
        log-softmax is 0, and the row contributes nothing but bookkeeping.
        """
        S, T, M = [], [], []
        for s, _idx, _r, key in batch:
            row = observed.get(key, {})
            if len(row) < self.row_min_cells:
                continue
            rv = torch.full((N_FACTORS,), float("-inf"))
            for f, rew in row.items():
                rv[FACTOR_VALUES.index(f)] = rew
            S.append(s)
            M.append(torch.isfinite(rv))
            T.append(torch.softmax(rv / self.rank_temp, dim=0))
        return S, T, M

    def update(self, buf: list, observed: dict) -> float:
        """
        One pass of K epochs over the buffer. Buffer entries are
        (state, factor_idx, reward, row_key).

        loss_mode='cell'  MSE(Q[chosen], reward) — the original. Each sample
                          supervises ONE of the ten outputs and is then dropped,
                          so each output row is fitted from roughly a tenth of
                          the data and nothing ever compares the ten outputs of
                          the same loop to each other.

        loss_mode='row'   Soft cross-entropy over every cell measured so far for
                          that loop-branch. Same measurements, same budget: the
                          agent still only ever sees cells it paid for. What
                          changes is that a measurement keeps contributing after
                          the step in which it was taken, so every output is
                          supervised on every loop the agent has data for, and
                          the ten outputs of one loop finally appear in the same
                          loss term.
        """
        tot, n = 0.0, 0
        rows, cells, seen_rows = 0, 0, 0
        for _ in range(self.K):
            order = list(range(len(buf)))
            random.shuffle(order)
            for i in range(0, len(order), self.batch_size):
                batch = [buf[j] for j in order[i:i + self.batch_size]]
                if not batch:
                    continue
                seen_rows += len(batch)
                if self.loss_mode == "cell":
                    s = torch.stack([b[0] for b in batch])
                    a = torch.tensor([b[1] for b in batch], dtype=torch.long)
                    r = torch.tensor([b[2] for b in batch], dtype=torch.float32)
                    q = self.factor_actor.forward(s).gather(
                        1, a.unsqueeze(1)).squeeze(1)
                    tot += self._step(F.mse_loss(q, r))
                    n += 1
                    rows += len(batch)
                    cells += len(batch)          # one supervised cell per sample
                    continue

                S, T, M = self._row_batch(batch, observed)
                if not S:
                    continue                     # no rankable row in this batch
                logits = self.factor_actor.forward(torch.stack(S))
                tot += self._step(rank_loss(logits, torch.stack(T),
                                            torch.stack(M)))
                n += 1
                rows += len(S)
                cells += int(sum(int(m.sum()) for m in M))
        # Fraction of buffer entries the loss could actually use. Under 'cell'
        # this is 1.0 always. Under 'row' it starts near 0 — most rows have a
        # single measured cell early on — and climbs as the table fills. It is
        # the direct check that the change took effect.
        self.last_rankable = rows / seen_rows if seen_rows else float("nan")
        self.last_cells = cells / rows if rows else float("nan")
        # NaN, not 0.0, when no batch was usable. A zero loss reads as a perfect
        # fit and is the exact opposite of what an empty epoch means — and epoch
        # 1 of a row run is often empty.
        return tot / n if n else float("nan")


# ---------------------------------------------------------------------------
# States and rewards
# ---------------------------------------------------------------------------

def random_split(benchmarks: list, test_frac: float, seed: int) -> tuple:
    """
    (train_benchmarks, test_benchmarks), grouped by BENCHMARK.

    Replaces grouped_kfold. K-fold costs folds x seeds runs to answer a question
    that a handful of independent random splits answers just as well: nothing
    here needs every benchmark to appear in the held-out set exactly once, since
    the result is read as a distribution over splits rather than as one pooled
    prediction per loop.

    Grouping by benchmark is NOT optional. Loops inside a benchmark share source,
    kernels and often the reward itself, so splitting one across the boundary
    leaks the answer.

    sorted() before shuffling so the draw is a function of the seed alone and not
    of dict iteration order.
    """
    bs = sorted(benchmarks)
    random.Random(seed).shuffle(bs)
    n = max(1, min(len(bs) - 1, round(test_frac * len(bs))))
    return bs[n:], bs[:n]


def states_for(loop: dict, data: dict) -> list:
    """
    [(u, state, mask)] for both branches of one loop.

    NOTE the two states are identical when a loop has no post-unmerge vector —
    make_state falls back to s1 (adapt_eval.py:211-215). One loop of 446 is in
    that position, so it contributes contradictory training data: same input,
    two different reward rows. Recorded, not corrected.
    """
    raw = loop["pre_features_raw"]
    known, trip = raw[IDX_TRIP_KNOWN] > 0.5, int(raw[IDX_TRIP_COUNT])
    out = []
    for u, cat, _ in BRANCHES:
        _, s2 = make_state(loop, data["normalizer"], data["postf"], u)
        out.append((u, s2, category_factor_mask(cat, known, trip)))
    return out


def reward_for(table: dict, u: int, f: int, missing: float,
               floor_penalty=None) -> tuple:
    """(reward, measured). An absent cell is charged `missing`: after exhaustive
    collection it means the cell FAILED, so charging it is the honest reading."""
    if (u, f) in table:
        from offline_train import reshape_floor
        return reshape_floor(table[(u, f)], floor_penalty), True
    return missing, False


def branch_oracle(table: dict, u: int, deadzone: float) -> float:
    """
    Best gated reward available WITHIN one branch, 0.0 if nothing clears the
    deadzone.

    Not `oracle_of_gated`, which maximises over the whole table: scoring the
    pre-unmerge state against a target only reachable by unmerging would make
    that population look worse than it is. The deadzone gate and the 0.0 floor
    match label_loops' rule — a transform must beat declining, and declining is
    free.
    """
    best = 0.0
    for (uu, f), r in table.items():
        if uu != u or (uu, f) == NOOP:
            continue
        if r > deadzone and r > best:
            best = r
    return best


def branch_capture(picks: list, tables: dict, u: int, deadzone: float) -> tuple:
    """(capture, n_loops_with_headroom) for one state population."""
    num = den = 0.0
    n = 0
    for key, r in picks:
        orc = branch_oracle(tables[key], u, deadzone)
        if orc > deadzone:
            n += 1
            num += r
            den += orc
    return (num / den if den > 0 else float("nan")), n


def branch_bar(loops: list, tables: dict, u: int, cat: int,
               deadzone: float, missing: float) -> tuple:
    """(factor, capture) for the best single fixed factor on one population."""
    cands = []
    for i, f in enumerate(FACTOR_VALUES):
        picks = []
        for l in loops:
            raw = l["pre_features_raw"]
            mask = category_factor_mask(cat, raw[IDX_TRIP_KNOWN] > 0.5,
                                        int(raw[IDX_TRIP_COUNT]))
            if not bool(mask[i]):
                continue
            key = (l["benchmark_name"], l["loop_idx"])
            picks.append((key, reward_for(tables[key], u, f, missing)[0]))
        if not picks:
            continue
        c, _ = branch_capture(picks, tables, u, deadzone)
        if c == c:                       # skip NaN
            cands.append((c, f))
    if not cands:
        return 0, float("nan")
    c, f = max(cands, key=lambda t: (t[0], -t[1]))
    return f, c


def probe_picks(agent: FactorOnly, loops: list, data: dict) -> list:
    """
    (bench, loop_idx, action) on each loop's TRUE branch — the same measurement
    `factor_probe_picks` makes for the coupled agents, so the number is directly
    comparable. Loops labelled no-op are excluded: they have no headroom, so
    there is no factor question to ask.
    """
    out = []
    for l in loops:
        key = (l["benchmark_name"], l["loop_idx"])
        truth = data["labels"].get(key)
        if truth is None or truth == "noop":
            continue
        u = 1 if truth == "unmerge_unroll" else 0
        raw = l["pre_features_raw"]
        cat = UNMERGE_UNROLL if u else UNROLL_ONLY
        _, s2 = make_state(l, data["normalizer"], data["postf"], u)
        mask = category_factor_mask(cat, raw[IDX_TRIP_KNOWN] > 0.5,
                                    int(raw[IDX_TRIP_COUNT]))
        out.append((key[0], key[1],
                    (u, FACTOR_VALUES[agent.select(s2, mask, greedy=True)])))
    return out


def probe_arms(loops: list, data: dict, deadzone: float) -> list:
    """
    [(state, u, table, oracle, mask)] for the loops the probe scores.

    Built once so the model's top-k and the constant-k bar below read the same
    population, the same true branch and the same denominator. Only loops with
    headroom on the gated oracle are kept — a loop with nothing to win has no
    factor question and would sit in the denominator contributing zero.
    """
    out = []
    for l in loops:
        key = (l["benchmark_name"], l["loop_idx"])
        truth = data["labels"].get(key)
        if truth is None or truth == "noop":
            continue
        table = data["tables"][key]
        _, orc = oracle_of_gated(table, deadzone)
        if orc <= deadzone:
            continue
        u = 1 if truth == "unmerge_unroll" else 0
        cat = UNMERGE_UNROLL if u else UNROLL_ONLY
        raw = l["pre_features_raw"]
        _, s2 = make_state(l, data["normalizer"], data["postf"], u)
        mask = category_factor_mask(cat, raw[IDX_TRIP_KNOWN] > 0.5,
                                    int(raw[IDX_TRIP_COUNT]))
        if not bool(mask.any()):
            # A known trip count of 1 leaves unroll_only with no legal factor.
            # torch.topk would raise on an empty selection, and the arm has no
            # factor question to answer anyway.
            continue
        out.append((s2, u, table, orc, mask))
    return out


def _tried(table: dict, u: int, factors) -> float:
    """What trying `factors` on one arm earns. Declining is free, so anything
    worse than the no-op costs 0.0 — and a cell absent from the cache never
    built, which also leaves the program unchanged at 0.0."""
    got = [table[(u, f)] for f in factors if (u, f) in table]
    return max(0.0, max(got)) if got else 0.0


def probe_topk(agent: FactorOnly, arms: list, k: int) -> float:
    """
    Capture when the head nominates its k best LEGAL factors and the best of
    them is kept.

    k=1 is NOT the probe row and will normally sit above it. The probe commits
    blind to one factor and is charged whatever that factor does, so a genuine
    slowdown costs its real value (score_decisions applies r_applied, which only
    zeroes cells that never BUILT). Top-k measures its candidates before
    choosing, so it can decline — `_tried` clamps at 0.0. Two different
    deployment stories: "predict and ship" versus "shortlist, measure, keep the
    best". Compare each against its own bar, never against the other.
    """
    num = den = 0.0
    for s2, u, table, orc, mask in arms:
        with torch.no_grad():
            q = agent.factor_actor.forward(s2.unsqueeze(0))[0]
            q = q.masked_fill(~mask.to(q.device), float("-inf"))
        idx = torch.topk(q, min(k, int(mask.sum()))).indices.tolist()
        num += _tried(table, u, [FACTOR_VALUES[i] for i in idx])
        den += orc
    return num / den if den > 0 else float("nan")


def set_capture(arms: list, S) -> float:
    """Capture of one FIXED factor set on a population. Factors the arm's mask
    forbids are simply not tried there."""
    num = den = 0.0
    for _s2, u, table, orc, mask in arms:
        num += _tried(table, u,
                      [f for f in S if mask[FACTOR_VALUES.index(f)]])
        den += orc
    return num / den if den > 0 else float("nan")


def constant_topk_bar(arms: list, k: int) -> tuple:
    """
    (capture, set) for the best FIXED k factors — "always try {2,4,8}".

    Without it a top-k number cannot be read: trying more candidates raises
    capture whatever the candidates are, so the question is never "is top-3
    better than top-1" but "is the head's top-3 better than ANY three".
    Exhaustive over C(10,k) <= 252 sets, so no search heuristic to distrust.
    """
    best = (float("-inf"), None)
    for S in combinations(FACTOR_VALUES, k):
        c = set_capture(arms, S)
        if c == c and c > best[0]:
            best = (c, S)
    return best


def branch_picks(agent: FactorOnly, loops: list, data: dict,
                 u: int, cat: int, missing: float) -> list:
    """[(key, realized_reward)] for one state population, greedy."""
    out = []
    for l in loops:
        raw = l["pre_features_raw"]
        _, s2 = make_state(l, data["normalizer"], data["postf"], u)
        mask = category_factor_mask(cat, raw[IDX_TRIP_KNOWN] > 0.5,
                                    int(raw[IDX_TRIP_COUNT]))
        f = FACTOR_VALUES[agent.select(s2, mask, greedy=True)]
        key = (l["benchmark_name"], l["loop_idx"])
        out.append((key, reward_for(data["tables"][key], u, f, missing)[0]))
    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(loops: list, data: dict, args, seed: int) -> tuple:
    """(agent, coverage_fraction). Coverage is over (loop, branch, factor)."""
    torch.manual_seed(seed)
    random.seed(seed)
    # getattr, not attribute access: factor_adapt.py builds its OWN argparse and
    # calls this function, so a flag added here would otherwise crash that
    # script on its first split. The defaults match this module's parser.
    agent = FactorOnly(args.lr, args.weight_decay, args.max_grad_norm,
                       args.epsilon, args.batch_size, args.K,
                       loss_mode=getattr(args, "loss", "row"),
                       rank_temp=getattr(args, "rank_temp", 0.1),
                       row_min_cells=getattr(args, "row_min_cells", 2),
                       head=getattr(args, "factor_head", "mlp"),
                       feats=getattr(args, "factor_feats", "basic"))

    cells = [(l, u, s, m) for l in loops for u, s, m in states_for(l, data)]
    # Denominator for coverage: every LEGAL (loop, branch, factor) cell. Legal,
    # not all 10 — factor 1 is illegal on the pre-unmerge branch, so counting 10
    # there would cap coverage below 100% and make a full sweep look partial.
    n_legal = sum(int(m.sum()) for _, _, _, m in cells)
    seen = set()
    # The measured table, built as the agent goes. Keyed per (loop, branch), so
    # one entry is one row of the factor response. Grows monotonically and is
    # never read for a cell the agent has not paid for — that is what keeps the
    # row loss inside the see-as-you-go rule rather than peeking at
    # data["tables"].
    observed: dict = {}

    buf = []
    history = []
    for epoch in range(1, args.epochs + 1):
        frac = (epoch - 1) / (args.epochs - 1) if args.epochs > 1 else 0.0
        agent.epsilon = args.epsilon + frac * (args.epsilon_final - args.epsilon)
        order = list(range(len(cells)))
        random.shuffle(order)
        # Q-loss is the only view into whether the head is fitting at all. A flat
        # curve with high coverage means it cannot represent the table; a falling
        # one that still fails on held-out loops means it can, and the failure is
        # transfer. Those are different results and the final table cannot tell
        # them apart.
        loss_sum, n_upd = 0.0, 0
        rank, cpr = [], []
        for i in order:
            l, u, s, m = cells[i]
            idx = agent.select(s, m)
            f = FACTOR_VALUES[idx]
            key = (l["benchmark_name"], l["loop_idx"])
            # ROLLOUT only. branch_bar and branch_picks (the scoring paths)
            # deliberately do NOT pass it — they must read the true value.
            r, _ = reward_for(data["tables"][key], u, f, args.missing,
                              args.train_floor_penalty)
            seen.add((key, u, f))
            observed.setdefault((key[0], key[1], u), {})[f] = r
            # The row KEY, not a snapshot of the row: rows grow as the agent
            # measures, and a snapshot taken at append time would train against
            # a stale, smaller row than the one now available.
            buf.append((s, idx, r, (key[0], key[1], u)))
            if len(buf) >= args.buffer_size:
                _l = agent.update(buf, observed)
                if _l == _l:                      # skip empty updates in the mean
                    loss_sum += _l; n_upd += 1
                rank.append(agent.last_rankable)
                cpr.append(agent.last_cells)
                buf = []
        if buf:
            _l = agent.update(buf, observed)
            if _l == _l:
                loss_sum += _l; n_upd += 1
            rank.append(agent.last_rankable)
            cpr.append(agent.last_cells)
            buf = []
        cov = len(seen) / n_legal if n_legal else float("nan")
        ok = [c for c in cpr if c == c]
        rk = [c for c in rank if c == c]
        history.append({"epoch": epoch,
                        "loss": loss_sum / n_upd if n_upd else float("nan"),
                        "epsilon": agent.epsilon, "coverage": cov,
                        "updates": n_upd,
                        # Rows the loss actually consumed this epoch, and how
                        # many measured cells each carried. Under --loss cell
                        # cells_per_row is 1 by construction; under --loss row
                        # it is the supervision multiplier, and if it does not
                        # climb the row loss is starved and the run says nothing.
                        "rankable": (sum(rk) / len(rk)) if rk else float("nan"),
                        "cells_per_row": (sum(ok) / len(ok)) if ok else float("nan")})
        if args.log_every and (epoch % args.log_every == 0
                               or epoch in (1, args.epochs)):
            log.info("      epoch %4d | loss %.5f | eps %.3f | coverage %5.1f%% "
                     "| rankable %5.1f%% | cells/row %.2f",
                     epoch, history[-1]["loss"], agent.epsilon, 100 * cov,
                     100 * history[-1]["rankable"], history[-1]["cells_per_row"])
    return agent, (len(seen) / n_legal if n_legal else float("nan")), history


def evaluate(agent: FactorOnly, loops: list, data: dict, args,
             xfer_sets: "dict | None" = None) -> dict:
    """Probe on the true branch, plus each state population separately."""
    m = score_decisions(probe_picks(agent, loops, data), data["tables"],
                        data["labels"], args.deadzone, args.missing,
                        data.get("failures"))
    _, bar, _ = best_constant_factor(loops, data["tables"], data["labels"],
                                     args.deadzone, args.missing)
    # TOP-k. The head nominates its k best legal factors instead of collapsing
    # to argmax; the best of them is kept. Deployable as a pruned autotuning
    # budget -- k compiles instead of 19 -- and it asks a coarser question than
    # top-1, which the head may be able to answer even if it cannot rank.
    #
    # Each k carries the best FIXED set of k factors beside it, because trying
    # more candidates raises capture whatever they are. top1 must reproduce
    # `probe` above; if it does not, the two are measuring different things.
    arms = probe_arms(loops, data, args.deadzone)
    out = {}
    for k in (1, 2, 3):
        out[f"top{k}"] = probe_topk(agent, arms, k)
        c, S = constant_topk_bar(arms, k)
        out[f"top{k}_bar"] = c
        out[f"top{k}_set"] = "/".join(str(f) for f in S) if S else ""
        # `bar` above is chosen ON this population, so it knows the answer and
        # is an upper bound on any fixed strategy rather than a baseline anyone
        # could ship. `xfer` is the set chosen on the FIT split scored here,
        # which is what a constant policy would actually deliver. The learned
        # row has to beat THAT to be worth anything.
        if xfer_sets and xfer_sets.get(k):
            out[f"top{k}_xfer"] = set_capture(
                arms, tuple(int(x) for x in xfer_sets[k].split("/")))
        else:
            out[f"top{k}_xfer"] = float("nan")
    out["topk_n"] = len(arms)

    out.update({"probe": m["capture"], "probe_n": m["loops_with_headroom"],
           "probe_bar": bar,
           # Sums, so callers can POOL rather than average per-benchmark
           # ratios — with one or two evaluation loops apiece, a mean of
           # capture ratios is decided by whichever denominators are smallest.
           # `probe_realized` uses the APPLIED semantics: a pick that failed
           # to build contributes 0.0, because the original program stands.
           "probe_realized": m["realized_sum"],
           "probe_oracle": m["oracle_sum"],
           # `probe_slower` counts only picks that BUILT AND RAN — a compile
           # failure carries the same reward as a slowdown, and 31.4% of cells
           # in this cache are failures, so the raw n_regress counts programs
           # that never ran. `probe_failed` is that population, reported apart.
           "probe_mean": m["mean_realized"],
           "probe_mean_applied": m["mean_realized_applied"],
           "probe_slower": m["n_regress"], "probe_failed": m["n_failed"],
           "probe_clipped": m["n_clipped"]})
    for u, cat, label in BRANCHES:
        # ONLY loops whose true category IS this branch. Scoring an
        # unroll-only loop on the unmerge branch measures a decision the policy
        # never makes, and its unmerge oracle is often barely above the deadzone
        # — a single bad pick against a 0.01 denominator reads as -3000% and
        # swamps the column.
        want = "unmerge_unroll" if u else "unroll_only"
        sub = [l for l in loops
               if data["labels"].get((l["benchmark_name"], l["loop_idx"])) == want]
        picks = branch_picks(agent, sub, data, u, cat, args.missing)
        c, n = branch_capture(picks, data["tables"], u, args.deadzone)
        bf, bc = branch_bar(sub, data["tables"], u, cat, args.deadzone,
                            args.missing)
        out[f"u{u}"] = c
        out[f"u{u}_n"] = n
        out[f"u{u}_bar"] = bc
        out[f"u{u}_barf"] = bf
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--deadzone", type=float, default=0.005)
    p.add_argument("--labels", type=Path, default=None)
    p.add_argument("--splits", type=int, default=5,
                   help="Independent random train/test splits over BENCHMARKS. "
                        "Replaces k-fold: 5 splits is 5 runs, where 5 folds x 3 "
                        "init seeds was 15, and the result is read as a "
                        "distribution either way.")
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--base-seed", type=int, default=100)
    p.add_argument("--seeds", type=int, default=1,
                   help="Init seeds PER split. 1 by default — which loops land "
                        "in the split moves the result far more than the network "
                        "init does, so extra splits beat extra inits.")
    p.add_argument("--epochs", type=int, default=300,
                   help="Many epochs is the point: there is no policy to "
                        "collapse and no category head to over-decline.")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--buffer-size", type=int, default=128)
    p.add_argument("--K", type=int, default=2)
    p.add_argument("--epsilon", type=float, default=0.3)
    p.add_argument("--epsilon-final", type=float, default=0.05,
                   help="Set equal to --epsilon for flat exploration, or both "
                        "to 1.0 for uniform sampling — that arm separates "
                        "'cannot learn' from 'did not revisit'.")
    p.add_argument("--loss", choices=("row", "cell"), default="row",
                   help="'row' (default): soft cross-entropy over every cell "
                        "measured so far for that loop-branch. 'cell': the "
                        "original MSE on the single chosen cell — keep it to "
                        "reproduce earlier runs.")
    p.add_argument("--rank-temp", type=float, default=0.1,
                   help="Temperature of the softmax target over rewards. Small "
                        "makes near-ties stay ties and pushes very negative "
                        "cells to zero target mass. (--loss row only)")
    p.add_argument("--row-min-cells", type=int, default=2,
                   help="A row needs this many measured cells to be rankable. "
                        "Below 2 the target is one-hot over one position and "
                        "the loss is identically zero. (--loss row only)")
    p.add_argument("--factor-head", choices=("mlp", "scorer"), default="mlp",
                   help="'scorer' shares weights across the ten factors and "
                        "takes the factor as an INPUT, so every sample teaches "
                        "something about every factor.")
    p.add_argument("--factor-feats", choices=("basic", "interact"),
                   default="basic", help="Factor features for --factor-head "
                                         "scorer.")
    p.add_argument("--missing", type=float, default=-0.161,
                   help="Reward for a cell absent from the cache. After "
                        "exhaustive collection an absent row means it FAILED.")
    p.add_argument("--train-floor-penalty", type=float, default=None,
                   help="Remap the -1.0 clip floor in the ROLLOUT reward "
                        "only; scoring keeps the true value. Off by default.")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--log-every", type=int, default=25,
                   help="Emit the Q-loss every N epochs. 0 silences it. Epochs "
                        "1 and --epochs are always emitted.")
    p.add_argument("--csv-out", type=str, default=None,
                   help="Per-split rows, fit_* and test_* prefixed.")
    p.add_argument("--curve-out", type=str, default=None,
                   help="Per-epoch Q-loss, epsilon and coverage.")
    args = p.parse_args()

    torch.set_num_threads(max(1, args.threads))
    data = load_run(args.run_dir, args.deadzone, args.labels)
    loops = labelled_loops(data)
    log.info("Loaded %d loops / %d benchmarks | %d labelled | %d states",
             len(data["loops"]), len(data["benchmarks"]),
             data["n_labelled_loops"], 2 * len(loops))
    # Printed before anything runs so a log file says what produced it.
    if args.loss == "row":
        log.info("LOSS  row  — soft cross-entropy over every cell measured so "
                 "far for a\n            (loop, branch), target softmax(r/%.2f). "
                 "Rows need >=%d cells.\n            Level-invariant, so no "
                 "separate centring; very negative cells get\n            "
                 "near-zero target mass instead of dominating the gradient.",
                 args.rank_temp, args.row_min_cells)
    else:
        log.info("LOSS  cell — MSE on the single chosen cell (the original). "
                 "One output\n            supervised per sample.")
    if args.factor_head == "scorer":
        log.info("HEAD  scorer (%s) — factor is an INPUT, weights shared across "
                 "the ten.", args.factor_feats)
    else:
        log.info("HEAD  mlp — 93 -> 128 -> 64 -> 10, factor is an output index.")
    log.info("      epochs %d | lr %g | eps %.2f->%.2f | buffer %d | batch %d | "
             "K %d\n", args.epochs, args.lr, args.epsilon, args.epsilon_final,
             args.buffer_size, args.batch_size, args.K)

    rows, curve = [], []
    for sp in range(args.splits):
        sseed = args.split_seed + sp
        tr_b, te_b = random_split(data["benchmarks"], args.test_frac, sseed)
        fit_l, test_l = loops_for(loops, tr_b), loops_for(loops, te_b)
        if not fit_l or not test_l:
            log.warning("  split %d produced an empty side (%d train / %d test "
                        "loops) — skipped", sp + 1, len(fit_l), len(test_l))
            continue
        for si in range(args.seeds):
            seed = args.base_seed + si
            agent, cov, hist = train(fit_l, data, args, seed)
            fit = evaluate(agent, fit_l, data, args)
            # Test is scored a second way as well: under the constant set the
            # FIT split chose. That is the only constant baseline that could
            # actually be shipped, and it is the one the learned row must beat.
            test = evaluate(agent, test_l, data, args,
                            xfer_sets={k: fit[f"top{k}_set"] for k in (1, 2, 3)})
            rows.append({"split": sp + 1, "split_seed": sseed, "seed": seed,
                         "n_fit_loops": len(fit_l), "n_test_loops": len(test_l),
                         "n_test_benchmarks": len(te_b), "coverage": cov,
                         "loss_first": hist[0]["loss"],
                         "loss_final": hist[-1]["loss"],
                         # Final-epoch supervision width. Under --loss row this
                         # is the whole point of the change, so it belongs in
                         # the summary and not only in the curve file.
                         "cells_per_row": hist[-1]["cells_per_row"],
                         "rankable": hist[-1]["rankable"],
                         "fit": fit, "test": test})
            curve.extend(dict(split=sp + 1, seed=seed, **h) for h in hist)
            log.info("  split %d/%d seed %d | %3d/%3d loops | coverage %5.1f%% | "
                     "probe fit %5.1f%% (bar %5.1f%%)  test %5.1f%% (bar %5.1f%%)",
                     sp + 1, args.splits, seed, len(fit_l), len(test_l),
                     100 * cov,
                     100 * fit["probe"], 100 * fit["probe_bar"],
                     100 * test["probe"], 100 * test["probe_bar"])

    report(rows, args)
    if args.csv_out and rows:
        write_csv(Path(args.csv_out), [flatten(r) for r in rows])
        log.info("\n  per-split rows: %s", args.csv_out)
    if args.curve_out and curve:
        write_csv(Path(args.curve_out), curve)
        log.info("  q-loss curve  : %s", args.curve_out)


def flatten(row: dict) -> dict:
    """One flat CSV record. `fit` and `test` are nested dicts of the same keys,
    so they are prefixed rather than merged — otherwise the second silently
    overwrites the first and every reported number is the test one."""
    out = {k: v for k, v in row.items() if k not in ("fit", "test")}
    for side in ("fit", "test"):
        for k, v in row[side].items():
            out[f"{side}_{k}"] = round(v, 6) if isinstance(v, float) else v
    return out


def _mean(rows: list, split: str, key: str) -> float:
    vals = [r[split][key] for r in rows if r[split][key] == r[split][key]]
    return sum(vals) / len(vals) if vals else float("nan")


def _mode(rows: list, split: str, key: str) -> int:
    """Most common value across folds. sorted(), not set(): set iteration order
    varies with hash randomisation, so a tie would print differently run to run
    — the bug already fixed twice elsewhere in this codebase."""
    vals = [r[split][key] for r in rows if r[split][key]]
    if not vals:
        return 0
    return max(sorted(set(vals)), key=vals.count)


def _plain(rows: list, key: str) -> float:
    """Mean over a TOP-LEVEL row field (coverage, cells_per_row, ...). `_mean`
    reaches into rows[split][key] and cannot see these."""
    v = [r[key] for r in rows if key in r and r[key] == r[key]]
    return sum(v) / len(v) if v else float("nan")


def report(rows: list, args=None) -> None:
    if not rows:
        log.error("no folds produced a result")
        return
    cov = sum(r["coverage"] for r in rows) / len(rows)
    log.info("\n" + "=" * 78)
    log.info("  UNROLLER IN ISOLATION — no category head")
    log.info("=" * 78)
    if args is not None:
        log.info("  loss=%s  head=%s%s", args.loss, args.factor_head,
                 f" ({args.factor_feats})" if args.factor_head == "scorer" else "")
    log.info("  mean cell coverage %.1f%% of legal (loop, branch, factor) cells."
             "\n  High coverage with a poor result is a LEARNING failure; low "
             "coverage is an\n  exploration failure. That is what this column "
             "is for.", 100 * cov)
    cpr = _plain(rows, "cells_per_row")
    log.info("  supervision: %.2f measured cells per row at the last epoch, "
             "%.0f%% of rows rankable.\n  Under loss=cell this is 1.00 by "
             "construction. Under loss=row it is the\n  multiplier on how much "
             "of each measurement the loss actually uses — if it\n  is near 1 "
             "the rows never filled and the change did not take effect.\n",
             cpr, 100 * _plain(rows, "rankable"))
    log.info("                              FIT                    HELD-OUT")
    log.info("                     learned     bar    n      learned"
             "     bar    n")
    log.info("  " + "-" * 68)
    for label, k in (("probe (true branch)", "probe"),
                     (BRANCHES[0][2] + " state", "u0"),
                     (BRANCHES[1][2] + " state", "u1")):
        log.info("  %-19s %6.1f%%  %6.1f%%  %4.0f     %6.1f%%  %6.1f%%  %4.0f",
                 label,
                 100 * _mean(rows, "fit", k),
                 100 * _mean(rows, "fit", k + "_bar"),
                 _mean(rows, "fit", k + "_n"),
                 100 * _mean(rows, "test", k),
                 100 * _mean(rows, "test", k + "_bar"),
                 _mean(rows, "test", k + "_n"))
    log.info("  " + "-" * 68)

    # TOP-k: the head nominates k factors, the best is kept. Read each row
    # against its OWN bar, not against top-1 — capture rises with k whatever is
    # nominated, so "top-3 beats top-1" is not a result.
    log.info("\n  TOP-k — head nominates k factors, best of them is kept")
    log.info("                 HELD-OUT")
    log.info("       k    learned   shippable    oracle-bar   fit's set")
    for k in (1, 2, 3):
        log.info("       %d   %7.1f%%   %8.1f%%   %10.1f%%   %s", k,
                 100 * _mean(rows, "test", f"top{k}"),
                 100 * _mean(rows, "test", f"top{k}_xfer"),
                 100 * _mean(rows, "test", f"top{k}_bar"),
                 _mode(rows, "fit", f"top{k}_set"))
    log.info("\n    shippable  = the constant set chosen on FIT, scored on "
             "held-out. This is\n                 the only column the learned "
             "row has to beat.")
    log.info("    oracle-bar = the best constant set chosen ON the held-out "
             "loops. It knows\n                 the answer, so it is a ceiling "
             "for any fixed policy, not a rival.")
    log.info("    Costs k compiles per loop and assumes you MEASURE the "
             "shortlist and keep\n    the best, so unlike the probe row above "
             "it may decline afterwards and will\n    normally read higher. "
             "Compare it to these bars, never to the probe.")

    # Which constant wins on each population. If the two differ, a head that
    # cannot tell the states apart cannot beat both bars at once — that is the
    # concrete form of "does it need to differentiate the branches".
    log.info("\n  best constant factor, fit split:  %s f=%d  |  %s f=%d",
             BRANCHES[0][2].strip(), _mode(rows, "fit", "u0_barf"),
             BRANCHES[1][2].strip(), _mode(rows, "fit", "u1_barf"))
    log.info(
        "\n  The bar is the best single FIXED factor on the same loops, so it "
        "moves\n  with the population and not with the model. Fit is a sanity "
        "check — this\n  head has ~21k parameters against ~888 states and can "
        "nearly memorise the\n  table. The HELD-OUT column against its bar is "
        "the result.")


if __name__ == "__main__":
    main()
