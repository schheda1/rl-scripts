"""
Supervised-contrastive term for the 3-way category head.

WHY
---
The measured failure is not over-firing. On held-out applications the pooled
confusion matrix has a near-correct MARGINAL (predicted 37/21/42 against a true
40/14/46) but assigns categories to the wrong loops: no-op vs unmerge+unroll,
which is 86% of the population, comes out at 143 correct / 170 confused — a coin
flip on the decision that matters.

A regression loss on Q has no term that says "these two loops want the same
action and that one wants a different one". This adds one, on the category
head's penultimate embedding.

THE LABEL IS EARNED, NOT LOOKED UP
----------------------------------
Labels come from the cells the agent has actually OBSERVED by iteration t, not
from the reward table:

    provisional(i, t) = oracle_of_gated( observed_cells_i(t) + free no-op, dz )

The no-op is known at 0.0 without sampling anything, so early on nearly every
loop is provisionally no-op and the label carries no information. It sharpens as
coverage grows. That is exactly why the term is gated: applied from epoch 1 it
would pull every loop into one cluster and teach the head to always decline.

Two gates, and the second is the real one:
  * an epoch floor  (--supcon-warmup), a crude proxy
  * a per-loop coverage threshold (--supcon-min-cells): a loop enters the
    contrastive batch only once enough of its cells have been paid for. This is
    the actual precondition; the epoch floor just avoids doing the bookkeeping
    early.

THE MECHANISM WORTH THE EFFORT
------------------------------
Better clusters raise fit, which is the stated target. But the interesting
consequence is different: epsilon-greedy explores uniformly and has no notion
that loop i resembles loops where unmerging paid. A clustered embedding gives
exactly that — if loop i lands among unmerge-labelled neighbours, the head
predicts unmerge for it even though loop i's OWN samples were all bad. That is
neighbourhood-transported action priors, i.e. directed exploration, which
nothing else in the agent provides.

Nothing in the pipeline is modified. The embedding is net[:6] of the existing
category head, obtained by Sequential slicing — no restructuring, no change to
any state_dict key.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import torch                                                     # noqa: E402
import torch.nn.functional as F                                  # noqa: E402

from offline_data import category_of, oracle_of_gated            # noqa: E402

_CAT_INDEX = {"noop": 0, "unroll_only": 1, "unmerge_unroll": 2}


def provisional_labels(loops: list, observed: dict, deadzone: float,
                       min_cells: int) -> dict:
    """
    {(bench, loop_idx): category_index} from OBSERVED cells only.

    A loop is included only once it has at least `min_cells` observed transform
    cells. Without that threshold every unvisited loop reports "no-op" — true by
    default, since the no-op is free — and the no-op cluster fills with loops
    that carry no evidence at all, teaching the head to decline everywhere.

    The free no-op at 0.0 is injected exactly as build_tables does, so a loop
    whose observed transforms are all harmful is correctly labelled no-op rather
    than "least bad transform".
    """
    out = {}
    for l in loops:
        key = (l["benchmark_name"], l["loop_idx"])
        seen = observed.get(key, {})
        if len(seen) < min_cells:
            continue
        cells = dict(seen)
        cells[(0, 1)] = 0.0
        action, _ = oracle_of_gated(cells, deadzone)
        out[key] = _CAT_INDEX[category_of(action)]
    return out


def embed(agent, states: torch.Tensor) -> torch.Tensor:
    """
    Penultimate activations of the category head.

    net[:6] is everything before the output projection — Sequential slicing, so
    no restructuring and no state_dict change. Agnostic to whether net[6] emits
    logits (PPO) or Q-values (bandit): the contrastive term shapes what feeds
    the decision, not the decision itself.
    """
    return agent.unmerge_actor.net[:6](states)


def supcon_loss(emb: torch.Tensor, labels: torch.Tensor,
                temperature: float = 0.1) -> torch.Tensor:
    """
    Supervised contrastive loss (Khosla et al.), multi-positive form.

    Anchors with no same-label partner in the batch contribute nothing — with
    three classes and a small batch that is common, and averaging over anchors
    that HAVE positives keeps the scale stable instead of letting it swing with
    how many happened to be present.

    Returns a 0-d zero (not None) when no anchor has a positive, so the caller
    can add it unconditionally.
    """
    n = emb.size(0)
    if n < 2:
        return emb.new_zeros(())
    z = F.normalize(emb, dim=1)
    sim = (z @ z.T) / temperature
    eye = torch.eye(n, dtype=torch.bool, device=emb.device)
    # Exclude self-similarity from the denominator, not just the numerator:
    # exp(1/tau) would otherwise dominate every row.
    sim = sim.masked_fill(eye, float("-inf"))
    logprob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~eye
    n_pos = pos.sum(1)
    keep = n_pos > 0
    if not bool(keep.any()):
        return emb.new_zeros(())
    per_anchor = -(logprob * pos).sum(1)[keep] / n_pos[keep]
    return per_anchor.mean()


def is_active(epoch: int, labels: dict, args) -> bool:
    """
    Gate. Needs the epoch floor AND at least two categories actually present —
    a single-category batch has no negatives, so the loss is identically zero
    and stepping on it only wastes an optimizer update.
    """
    if args.supcon_coef <= 0 or epoch < args.supcon_warmup:
        return False
    return len(set(labels.values())) >= 2


def label_summary(labels: dict) -> str:
    counts = [sum(1 for v in labels.values() if v == c) for c in (0, 1, 2)]
    return (f"{len(labels)} loops labelled "
            f"(no-op {counts[0]}, unroll {counts[1]}, unmerge {counts[2]})")
