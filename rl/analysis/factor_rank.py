"""
Per-loop soft-target ranking on the FACTOR head.

WHY
---
The factor is the measured bottleneck. Every intervention so far has raised
CATEGORY accuracy and left reward where it was — +7.6pp of category accuracy
bought +0.0005 of reward on the training set itself, with no transfer argument
involved. Meanwhile the median unmerge arm has a best-minus-worst spread of
+1.06, so the factor is where a correct decision is actually worth something.

What the factor head is trained on today is not a ranking signal:
    bandit  MSE(Q2, r) — fits MAGNITUDES, and the magnitudes are pathological
            (744 cells at exactly -1.0, 31% failure penalties). The gradient is
            dominated by separating -1.0 from -0.161, not +0.19 from 0.0.
    ppo     advantage-weighted policy gradient — ~10 samples per arm to
            establish a preference over 10 arms.
Either way the ordering, which is the only thing the decision consumes, is a
by-product.

THE TARGET
----------
Per loop, per BRANCH, over the factors OBSERVED for that branch:

    p_f  ∝  exp(r_f / temp)
    loss  =  -sum_f p_f * log softmax(head_out)_f      (masked to observed+valid)

Soft rather than argmax because the data says ties are the common case: the
median arm has 2-3 of 10 factors within the deadzone of the best, and only
30-45% have a unique winner. Hard argmax picks one of several equals and trains
against that arbitrary choice most of the time. At temp=0.1 a 0.005 gap stays a
near-tie (exp(0.05) ~= 1.05) while -1.0 against +0.2 separates by e^12.

Targets come from OBSERVED cells, never the reward table — same discipline as
supcon.py, so the term stays online-legitimate.

THE TENSION, WHICH SPLITS BY AGENT
----------------------------------
ppo     The factor head is a softmax policy. This is behaviour cloning with soft
        labels. No conflict.
bandit  Q1's target is max_f Q2, so Q2 must stay calibrated to reward MAGNITUDE.
        A ranking loss makes its scale arbitrary, which inflates the max backup
        and worsens the maximisation bias that backup already carries. Here the
        term is ADDITIVE to the MSE rather than a replacement, and
        factor_calibration() reports mean |Q2 - r| so the drift is visible
        instead of silent. Read that number before trusting a bandit result.

WHAT WOULD COUNT AS SUCCESS (fixed in advance)
----------------------------------------------
capture_fit, currently 50-77%. If it moves toward 90% the factor IS learnable
in-distribution and the earlier failures were the objective. If it does not
move, the factor is not learnable even with full information on seen loops, and
the cliff is a property of the problem rather than of the optimiser. Test
performance is NOT the criterion — transfer is already known not to happen.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import torch                                                     # noqa: E402
import torch.nn.functional as F                                  # noqa: E402

from agent import FACTOR_VALUES, N_FACTORS                       # noqa: E402
from category_agent import (UNMERGE_UNROLL, UNROLL_ONLY,         # noqa: E402
                            category_factor_mask)
from offline_data import IDX_TRIP_COUNT, IDX_TRIP_KNOWN          # noqa: E402

# (branch unmerge bit, the category that owns it). u=0 is scored against the
# unroll_only mask, which excludes factor 1 — (0,1) is the no-op, not a factor
# choice, so ranking it against the unroll factors would compare a decline to a
# transform.
_BRANCHES = ((0, UNROLL_ONLY), (1, UNMERGE_UNROLL))


def branch_rows(loops: list, observed: dict, data: dict, temp: float,
                min_cells: int) -> tuple:
    """
    (states, masks, targets, rewards) — one row per (loop, branch) with enough
    observed factors to rank. `rewards` carries the measured value at every
    observed cell (0 elsewhere) so the calibration check needs no second pass.

    A branch needs at least `min_cells` observed factors: with one you cannot
    express a preference, and the softmax target is degenerate.
    """
    from adapt_eval import make_state

    states, masks, targets, rewards = [], [], [], []
    for l in loops:
        key = (l["benchmark_name"], l["loop_idx"])
        seen = observed.get(key)
        if not seen:
            continue
        raw = l["pre_features_raw"]
        known, trip = raw[IDX_TRIP_KNOWN] > 0.5, int(raw[IDX_TRIP_COUNT])
        for u, cat in _BRANCHES:
            legal = category_factor_mask(cat, known, trip)
            cells = {f: r for (uu, f), r in seen.items()
                     if uu == u and f in FACTOR_VALUES
                     and bool(legal[FACTOR_VALUES.index(f)])}
            if len(cells) < min_cells:
                continue
            _, s2 = make_state(l, data["normalizer"], data["postf"], u)
            m = torch.zeros(N_FACTORS, dtype=torch.bool)
            r = torch.full((N_FACTORS,), float("-inf"))
            raw_r = torch.zeros(N_FACTORS)
            for f, rew in cells.items():
                i = FACTOR_VALUES.index(f)
                m[i], r[i], raw_r[i] = True, rew, rew
            # softmax over the OBSERVED subset only; -inf elsewhere makes the
            # unobserved factors carry exactly zero target mass.
            states.append(s2)
            masks.append(m)
            targets.append(torch.softmax(r / temp, dim=0))
            rewards.append(raw_r)
    return states, masks, targets, rewards


def rank_loss(logits: torch.Tensor, target: torch.Tensor,
              mask: torch.Tensor) -> torch.Tensor:
    """
    Soft cross-entropy between the head's masked distribution and the
    reward-derived target.

    The masked entries are zeroed AFTER log_softmax rather than left at -inf:
    target is 0 there, and 0 * -inf is NaN, not 0. That is the same trap the
    hand-rolled entropy hit — worth writing explicitly rather than relying on
    the target happening to be zero.
    """
    logp = F.log_softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)
    logp = torch.where(mask, logp, torch.zeros_like(logp))
    return -(target * logp).sum(dim=-1).mean()


def factor_calibration(agent, states: torch.Tensor, masks: torch.Tensor,
                       rewards: torch.Tensor) -> float:
    """
    Mean |Q2 - r| over observed cells — meaningful for the BANDIT only.

    Q1's target is max_f Q2, so if the ranking term pulls Q2's scale away from
    reward magnitude the backup silently inflates. This is the number that
    catches it; for PPO the head emits logits and calibration is not a property
    it is supposed to have.
    """
    with torch.no_grad():
        q = agent.factor_actor.forward(states)
        err = (q - rewards).abs()[masks]
    return float(err.mean()) if err.numel() else float("nan")


def is_active(epoch: int, n_rows: int, args) -> bool:
    """Gate: the coefficient is on, the warmup has passed, and there is at least
    one rankable (loop, branch) to learn from."""
    return args.rank_coef > 0 and epoch >= args.rank_warmup and n_rows > 0
