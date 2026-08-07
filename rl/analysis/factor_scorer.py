"""
Factor head that SEES the factor.

WHY
---
`FactorActor` is `_MLP(93 -> 128 -> 64 -> 10)`. The chosen factor is an OUTPUT
INDEX: factor 3 is neuron 2. The factor is never an input, so the head cannot
compute any function of (loop, factor). It learns ten independent output rows,
each from roughly a tenth of the data, with no notion that 3 and 4 are adjacent
while 3 and 10 are not.

This is not the capacity question that was already settled. That one concerned
the CATEGORY head, which does fit its training split (79% against
benchmark-dominant's 81.5%). The factor head does not: capture on the fit split
sits near 70%. So for the factor, representation is still open.

THE HEAD
--------
One scorer, applied to all ten factors:

    score(loop_state, factor_features) -> scalar        x10

Weights are shared across factors, so every sample teaches something about every
factor, and ordinality becomes expressible instead of having to be rediscovered
per output neuron.

NO TRIP COUNT
-------------
`tripCount` is unknown for most loops (`tripCountKnown == 0`, `tripCount == 0`),
so `tc mod f` and `tc / f` are undefined almost everywhere and would inject a
constant dressed up as a feature.

That absence points at the mechanism rather than merely removing an option. The
oracle prefers low factors MONOTONICALLY (f=1 wins 21.5%, decaying to f=10 at
3.6%, with the cost of being wrong rising +0.146 -> +0.418). Divisibility would
produce a BUMPY preference — for a trip count of 20, factors 2/4/5/10 would beat
3/6/7. Monotone decay with rising cost is what BODY GROWTH predicts: unrolling by
f multiplies body size, register pressure and I-cache footprint by f. So the
interaction worth encoding is factor x body-size, and every input it needs is
always present.

WHY THE PRODUCTS USE Z-SCORED VALUES DIRECTLY
---------------------------------------------
The first draft of this used `log1p(f * raw)` with raw recovered from the
normalizer. Two things were wrong with it, and both are load-bearing:

  1. `log1p(f*raw) ~= log(f) + log(raw)` for anything but small values — a SUM of
     a factor-only term and a loop-only term. The trunk can already form that
     from `log f` (feature 1) plus a linear readout of the state, so the feature
     would have added almost nothing while looking like it added an interaction.
  2. It needed the normalizer inverted, and post-unmerge features were normalised
     by a DIFFERENT normalizer than the one at hand — plus `log1p` of anything
     below -1 is NaN, one clamp away from poisoning a run.

The plain product `f * z` is a genuine interaction (no linear layer over [f, z]
can produce it), needs no inversion, and cannot go NaN. If z is an affine image
of the true raw value, `f*(a*true + b) = a*(f*true) + b*f` — the true interaction
plus a term linear in f, and f/10 is already feature 0. Nothing is lost.

COMPATIBILITY
-------------
Subclasses `FactorActor` so `log_prob` and `sample` are INHERITED, not copied —
both call `self.forward`, which is the only thing overridden. `__init__` calls
`_MLP.__init__` directly because `FactorActor.__init__` hard-codes the output
width at N_FACTORS and this head emits 1.

`.net` stays the same 7-element Sequential, which matters: the few-shot
adaptation path freezes layers by index (`offline_adapt._freeze_for_adaptation`
-> `net[0,1,3,4,6]`) and few-shot is the study's positive result.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import torch                                                     # noqa: E402

from agent import (FACTOR_VALUES, N_FACTORS, N_FEATURES,         # noqa: E402
                   FactorActor, _MLP)
from hecbench import FEATURE_COLUMNS                             # noqa: E402

# Body-size columns the interaction products multiply the factor against. Named
# and asserted rather than hard-coded blind: if FEATURE_COLUMNS is ever
# reordered, this must fail at import, not silently multiply the factor by
# `containsBarrier`.
_PRODUCT_COLUMNS = ("loopSize", "numMemoryInsts", "numComputeInsts")
_PRODUCT_IDX = [FEATURE_COLUMNS.index(c) for c in _PRODUCT_COLUMNS]
assert _PRODUCT_IDX == [2, 13, 14], (
    f"FEATURE_COLUMNS moved: {_PRODUCT_COLUMNS} are at {_PRODUCT_IDX}, "
    f"expected [2, 13, 14]")

FEATURE_SETS = ("basic", "interact")

# Widths, stated here so callers can size things without constructing a module.
N_BASIC = 5
N_INTERACT = N_BASIC + len(_PRODUCT_IDX)          # 8


def basic_table() -> torch.Tensor:
    """
    (N_FACTORS, N_BASIC) constant factor features, row i describing
    FACTOR_VALUES[i].

      0  f / 10           linear ordinality
      1  log(f) / log(10) the response is compressive, not linear
      2  1 / f            iteration-count shrink shape, no trip count needed
      3  is_pow2(f)       1,2,4,8 — alignment and vectorisation boundaries
      4  f == 1           the degenerate "no unroll" choice, categorically apart

    All five are bounded in [0, 1] so they sit on the same scale as the z-scored
    state rather than dominating the first layer.
    """
    rows = []
    for f in FACTOR_VALUES:
        rows.append([
            f / 10.0,
            math.log10(f),          # exact at the ends: f=1 -> 0.0, f=10 -> 1.0
            1.0 / f,
            1.0 if (f & (f - 1)) == 0 else 0.0,
            1.0 if f == 1 else 0.0,
        ])
    return torch.tensor(rows, dtype=torch.float32)


class FactorScorer(FactorActor):
    """Shared scorer over the ten factors. Drop-in for `FactorActor`."""

    def __init__(self, feats: str = "basic", logit_cap: float = 0.0) -> None:
        if feats not in FEATURE_SETS:
            raise ValueError(f"feats must be one of {FEATURE_SETS}, got {feats!r}")
        k = N_BASIC if feats == "basic" else N_INTERACT
        # Deliberately NOT FactorActor.__init__: that one fixes out_dim at
        # N_FACTORS. Everything else about FactorActor — log_prob, sample — is
        # inherited and works unchanged because both go through self.forward.
        _MLP.__init__(self, N_FEATURES + k, 1, logit_cap=logit_cap)
        self.feats = feats
        self.k = k
        # BUFFER, not Parameter. As a Parameter it would enter _all_params, pick
        # up weight decay, and drift — silently converting fixed features into
        # learned embeddings and destroying the ablation this head exists for.
        self.register_buffer("factor_feats", basic_table())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        (B, N_FEATURES) -> (B, N_FACTORS).

            x                (B, 93)
            expand           (B, 10, 93)
            factor feats     (B, 10, k)
            cat              (B, 10, 93+k)
            net              (B*10, 1)
            view             (B, 10)
        """
        b = x.size(0)
        xs = x.unsqueeze(1).expand(b, N_FACTORS, N_FEATURES)
        ff = self.factor_feats.unsqueeze(0).expand(b, N_FACTORS, N_BASIC)

        if self.feats == "interact":
            # f/10 (column 0 of the table) times the z-scored body-size columns.
            # (1,10,1) * (B,1,3) -> (B,10,3): factor varies down, loop across.
            f_lin = self.factor_feats[:, 0].view(1, N_FACTORS, 1)
            sel = x[:, _PRODUCT_IDX].unsqueeze(1)
            ff = torch.cat([ff, f_lin * sel], dim=-1)

        # .reshape, not .view: `expand` returns a non-contiguous stride-0 view
        # and .view would raise on it.
        out = self.net(torch.cat([xs, ff], dim=-1).reshape(b * N_FACTORS, -1))
        out = out.view(b, N_FACTORS)
        # _MLP.forward applies this after self.net; overriding forward means it
        # has to be reapplied here or the actor heads silently lose their bound.
        if self.logit_cap > 0:
            out = self.logit_cap * torch.tanh(out / self.logit_cap)
        return out


def swap_factor_head(agent, factory):
    """
    Replace `agent.factor_actor` with `factory(logit_cap)`, IN PLACE, rebuilding
    the optimizer. Architecture-agnostic on purpose: one rebuild, so a second
    head type cannot arrive with a second copy of this logic that drifts.

    The rebuild is the whole point. Every agent builds `_all_params` and the
    AdamW groups from the three modules' parameters inside __init__
    (`category_agent.py:186-193`). Assigning a new `factor_actor` afterwards and
    stopping there leaves the optimizer holding the DISCARDED head's tensors:
    training runs, losses go down, every reported number is real — and the new
    head never receives a single update. Nothing downstream would notice.

    Two values are read back off the agent rather than from `args`:

      * `logit_cap`, from the head being replaced, so the new head is capped
        exactly as the old one was (the bandit and PPO differ here).
      * `lr` / `weight_decay`, from the live optimizer's param_groups, so this
        cannot drift if the pipeline ever changes how it builds them.

    The decay/no-decay split (`ndim >= 2` -> decay) is duplicated from the agent
    because there is no accessor for it. `test_factor_scorer_shapes_and_wiring`
    and `test_factor_attn_head` assert that the new head's parameters are in the
    optimizer, the old head's are not, and `_all_params` agrees with it.

    CONSEQUENCE worth knowing for the attention head: its positional embedding is
    2-D, so this rule puts it in the WEIGHT-DECAY group. That is consistent with
    how the agent treats every other 2-D tensor, and it is not special-cased
    here — but it does mean the position signal is pulled toward zero over
    training, which is not what a transformer implementation would normally do.
    """
    old = agent.factor_actor
    lr = agent.optimizer.param_groups[0]["lr"]
    wd = agent.optimizer.param_groups[0]["weight_decay"]

    agent.factor_actor = factory(getattr(old, "logit_cap", 0.0)).to(agent.device)

    decay, no_decay = [], []
    for m in (agent.unmerge_actor, agent.factor_actor, agent.critic):
        for p in m.parameters():
            (decay if p.ndim >= 2 else no_decay).append(p)
    agent._all_params = decay + no_decay
    agent.optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": wd},
         {"params": no_decay, "weight_decay": 0.0}], lr=lr)
    return agent
