"""
Three-way category head — an isolated test of the 2-head factorization defect.

Changes NOTHING in the pipeline. Imports the trunk, the masks and the rollout
containers from agent.py and replaces only the decision structure, so a negative
result costs nothing and a positive one is a targeted patch to agent.py rather
than a rewrite.

THE DEFECT
----------
The pipeline factorizes an action as (unmerge, factor), so the no-op is the
CONJUNCTION unmerge==0 AND factor==1. At initialisation that is
P(u=0)*P(f=1) ~= 0.5 * 0.1 = 5% prior mass for an action that is correct on
40.1% of loops. It also encodes the no-op twice: (0,1) is reachable as "decline"
and as "unroll by 1", which are the same program, so the factor head is trained
to separate identical actions.

THE CHANGE
----------
    CategoryActor : s1 -> P(noop | unroll_only | unmerge_unroll)   _MLP(93, 3)
    FactorActor   : s2 -> P(factor)                                unchanged

    noop            -> no factor decision (factor_active=False)
    unroll_only     -> factor in {2..10}   (factor 1 masked out — it IS the no-op)
    unmerge_unroll  -> factor in {1..10}

No-op prior mass becomes 1/3, the duplicate encoding disappears, and the head
emits the same categories the evaluation scores.

THE Q-TARGET, AND A CAVEAT THAT CUTS THE OTHER WAY
--------------------------------------------------
Q1(s1, noop) regresses to EXACTLY 0.0 — the no-op's known value — instead of
bootstrapping through max_f Q2. That arm stops being estimated from noise.

But the two transform arms still bootstrap through a max, which is inflated by
max-of-noise, and inflated MORE on the unmerge branch: the measured factor
curves put its median best-worst spread at +1.06 against +0.25 for unroll-only,
and a max over a ~4x wider distribution is larger even at equal mean. An exact
no-op target beside optimistic transform targets can therefore WORSEN the
over-firing bias it is meant to fix. --q-pessimism subtracts a constant from the
transform targets so that can be measured rather than argued about.

WHAT WOULD COUNT AS SUCCESS (fixed in advance)
----------------------------------------------
  * no-op recall on test rises toward the 40% base rate
  * the correlation between a fold's no-op rate and policy accuracy (-0.64
    measured, the over-firing signature) weakens toward 0
  * fit accuracy clears the ~64-69% plateau toward benchmark-dominant's ~79%
If fit stays pinned in the 60s, the parameterization is not the constraint and
the remaining candidate is the features themselves.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch                                                     # noqa: E402
import torch.nn as nn                                            # noqa: E402
from dataclasses import dataclass                                # noqa: E402
import torch.nn.functional as F                                  # noqa: E402

from agent import (FACTOR_VALUES, N_FACTORS, N_FEATURES, Critic,  # noqa: E402
                   FactorActor, RolloutBuffer, RolloutEntry, _MLP,
                   _clipped_pg_loss, _policy_entropy, build_factor_mask)

# Index order matches label_loops.CATEGORIES so a head index IS a category.
CATEGORIES = ("noop", "unroll_only", "unmerge_unroll")
NOOP, UNROLL_ONLY, UNMERGE_UNROLL = 0, 1, 2
N_CATEGORIES = 3
_F1 = FACTOR_VALUES.index(1)


def category_factor_mask(category: int, trip_known: bool,
                         trip_count: int) -> torch.Tensor:
    """
    Valid factors for a category, on top of the trip-count mask.

    Factor 1 is removed from unroll_only because (unmerge=0, factor=1) IS the
    no-op — leaving it in restores the duplicate encoding this class exists to
    remove. unmerge_unroll keeps factor 1: unmerging by itself is a real
    transform that pays for path duplication.
    """
    mask = build_factor_mask(trip_known, trip_count)
    if category == UNROLL_ONLY:
        mask = mask.clone()
        mask[_F1] = False
    return mask


def category_mask(trip_known: bool, trip_count: int) -> torch.Tensor:
    """
    Which CATEGORIES are selectable for this loop.

    A loop with a known trip count of 1 has no valid unroll factor above 1, so
    unroll_only has an empty action set. Selecting it would softmax over all
    -inf and produce NaN. no-op is always available; unmerge_unroll always has
    factor 1.
    """
    f_mask = build_factor_mask(trip_known, trip_count)
    can_unroll = bool(f_mask.clone().index_fill_(
        0, torch.tensor([_F1]), False).any())
    return torch.tensor([True, can_unroll, True], dtype=torch.bool)


def to_pipeline_action(category: int, factor: int) -> tuple:
    """(category, factor) -> the pipeline's (unmerge, factor), for table lookup."""
    if category == NOOP:
        return 0, 1
    if category == UNROLL_ONLY:
        return 0, factor
    return 1, factor


@dataclass
class CategoryRolloutEntry(RolloutEntry):
    """
    RolloutEntry plus the collection-time CATEGORY mask.

    The 2-head agents never need this: unmerge is always {0,1}, both legal. The
    3-way head can have unroll_only masked out (a known trip count of 1 leaves
    it no legal factor), and PPO's ratio is only valid if the update reapplies
    the SAME mask the action was sampled under.
    """
    mask1: "torch.Tensor | None" = None


class CategoryActor(_MLP):
    """s1 -> P(category). Same trunk and cap as UnmergeActor, 3 outputs not 2."""

    def __init__(self, logit_cap: float = 0.0) -> None:
        super().__init__(N_FEATURES, N_CATEGORIES, logit_cap=logit_cap)

    def log_prob(self, features: torch.Tensor, actions: torch.Tensor,
                 mask: "torch.Tensor | None" = None) -> torch.Tensor:
        logits = self.forward(features)
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        return F.log_softmax(logits, dim=-1).gather(
            1, actions.unsqueeze(1)).squeeze(1)

    def sample(self, features: torch.Tensor, mask: torch.Tensor,
               greedy: bool = False) -> tuple:
        with torch.no_grad():
            logits = self.forward(features.unsqueeze(0)).masked_fill(
                ~mask.unsqueeze(0), float("-inf"))
            if greedy:
                action = logits.argmax(dim=-1)
            else:
                action = torch.distributions.Categorical(logits=logits).sample()
        log_p = F.log_softmax(logits, dim=-1)[0, action.item()]
        return int(action.item()), log_p.detach()


class CategoryAgent:
    """
    PPO over a 3-way category head plus the existing factor head.

    Interface-compatible with agent.Agent where offline_train needs it:
    ppo_update(buffer), save/load, and the same three modules so _snapshot and
    _restore work unchanged.
    """

    def __init__(self, *, clip_eps: float = 0.2, K: int = 2,
                 batch_size: int = 8, lr: float = 3e-4,
                 value_loss_coef: float = 0.5, entropy_coef: float = 0.01,
                 entropy_coef_category: "float | None" = None,
                 logit_cap: float = 0.0, weight_decay: float = 0.01,
                 max_grad_norm: float = 0.5, q_pessimism: float = 0.0,
                 device=None) -> None:
        self.clip_eps, self.K, self.batch_size = clip_eps, K, batch_size
        self.lr, self.weight_decay = lr, weight_decay
        self.value_loss_coef, self.max_grad_norm = value_loss_coef, max_grad_norm
        self.entropy_coef = entropy_coef
        # MIRROR agent.Agent: the low-arity head gets its own coefficient so a
        # 3-way head is not swamped by the 10-way factor head under one weight.
        self.entropy_coef_category = (entropy_coef if entropy_coef_category
                                      is None else entropy_coef_category)
        self.logit_cap, self.q_pessimism = logit_cap, q_pessimism
        self.device = device or torch.device("cpu")

        self.unmerge_actor = CategoryActor(logit_cap=logit_cap).to(self.device)
        self.factor_actor = FactorActor(logit_cap=logit_cap).to(self.device)
        self.critic = Critic().to(self.device)

        decay, no_decay = [], []
        for m in (self.unmerge_actor, self.factor_actor, self.critic):
            for p in m.parameters():
                (decay if p.ndim >= 2 else no_decay).append(p)
        self._all_params = decay + no_decay
        self.optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": weight_decay},
             {"params": no_decay, "weight_decay": 0.0}], lr=lr)

    # -- selection ---------------------------------------------------------

    def select_category(self, features, trip_known=False, trip_count=0,
                        greedy=False) -> tuple:
        mask = category_mask(trip_known, trip_count)
        cat, log_p = self.unmerge_actor.sample(
            features.to(self.device), mask=mask.to(self.device), greedy=greedy)
        return cat, log_p, mask

    def select_factor(self, features, category: int, trip_known=False,
                      trip_count=0, greedy=False) -> tuple:
        mask = category_factor_mask(category, trip_known, trip_count)
        idx, log_p = self.factor_actor.sample(
            features.to(self.device), mask=mask.to(self.device), greedy=greedy)
        return idx, log_p, mask

    def predict_value(self, features) -> float:
        with torch.no_grad():
            return self.critic.value(features.unsqueeze(0).to(self.device)).item()

    # -- update ------------------------------------------------------------

    def ppo_update(self, buffer: RolloutBuffer) -> dict:
        """Clipped PPO over both heads. Same name/signature as agent.Agent."""
        tot_a = tot_v = tot_e = tot_ec = tot_ef = 0.0
        n = 0
        for _ in range(self.K):
            with torch.no_grad():
                all_s1 = torch.stack([e.state1 for e in buffer._entries])
                all_r = torch.tensor([e.reward for e in buffer._entries],
                                     dtype=torch.float32)
                adv_raw = all_r - self.critic.value(all_s1)
                adv_mu, adv_sd = adv_raw.mean(), adv_raw.std(correction=0) + 1e-8
            for batch in buffer.sample_batches(self.batch_size):
                if not batch:
                    continue
                s1 = torch.stack([e.state1 for e in batch])
                s2 = torch.stack([e.state2 for e in batch])
                a1 = torch.tensor([e.action1 for e in batch], dtype=torch.long)
                a2 = torch.tensor([e.action2 for e in batch], dtype=torch.long)
                olp1 = torch.stack([e.log_prob1 for e in batch])
                olp2 = torch.stack([e.log_prob2 for e in batch])
                r = torch.tensor([e.reward for e in batch], dtype=torch.float32)
                m2 = torch.stack([
                    e.mask2 if e.mask2 is not None
                    else torch.ones(N_FACTORS, dtype=torch.bool) for e in batch])
                fa = torch.tensor([e.factor_active for e in batch],
                                  dtype=torch.bool)
                values = self.critic.value(s1)
                adv = ((r - values.detach()) - adv_mu) / adv_sd

                m1 = torch.stack([
                    getattr(e, "mask1", None)
                    if getattr(e, "mask1", None) is not None
                    else torch.ones(N_CATEGORIES, dtype=torch.bool)
                    for e in batch])
                nlp1 = self.unmerge_actor.log_prob(s1, a1, mask=m1)
                loss_a = _clipped_pg_loss(nlp1, olp1, adv, self.clip_eps)
                ent_c = _policy_entropy(
                    self.unmerge_actor.forward(s1).masked_fill(
                        ~m1, float("-inf")))
                ent_f = torch.zeros((), device=self.device)
                if fa.any():
                    nlp2 = self.factor_actor.log_prob(s2[fa], a2[fa], mask=m2[fa])
                    loss_a = loss_a + _clipped_pg_loss(nlp2, olp2[fa], adv[fa],
                                                       self.clip_eps)
                    ent_f = _policy_entropy(
                        self.factor_actor.forward(s2[fa]).masked_fill(
                            ~m2[fa], float("-inf")))
                # Computed ONCE. agent.Agent does the same; recomputing it after
                # backward() just to log it builds a second graph node for a
                # number already in hand.
                value_loss = F.mse_loss(values, r)
                loss = (loss_a
                        + self.value_loss_coef * value_loss
                        - self.entropy_coef_category * ent_c
                        - self.entropy_coef * ent_f)
                self.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self._all_params, self.max_grad_norm)
                self.optimizer.step()
                tot_a += float(loss_a)
                tot_v += float(value_loss)
                # MIRROR agent.Agent: "entropy" is the SUM over both heads, so
                # the column means the same thing whichever agent produced it.
                # Per-head values go in their own keys, as there.
                tot_e += float(ent_c) + float(ent_f)
                tot_ec += float(ent_c); tot_ef += float(ent_f)
                n += 1
        d = max(n, 1)
        return {"actor_loss": tot_a / d, "value_loss": tot_v / d,
                "entropy": tot_e / d, "entropy_unmerge": tot_ec / d,
                "entropy_factor": tot_ef / d}

    def save(self, path: str) -> None:
        torch.save({"unmerge_actor": self.unmerge_actor.state_dict(),
                    "factor_actor": self.factor_actor.state_dict(),
                    "critic": self.critic.state_dict()}, path)

    def load(self, path: str) -> None:
        d = torch.load(path, map_location=self.device)
        self.unmerge_actor.load_state_dict(d["unmerge_actor"])
        self.factor_actor.load_state_dict(d["factor_actor"])
        self.critic.load_state_dict(d["critic"])


class CategoryBanditAgent(CategoryAgent):
    """
    Value-based variant: epsilon-greedy selection, Q-regression update.

    MIRROR agent.BanditAgent, with the one structural change this file exists
    for — Q1's target for the no-op is the constant 0.0, not a bootstrapped
    max over a head that has never seen a no-op factor.
    """

    def __init__(self, *, epsilon: float = 0.1, **kw) -> None:
        super().__init__(**kw)
        self.epsilon = epsilon

    def select_category(self, features, trip_known=False, trip_count=0,
                        greedy=False) -> tuple:
        import random
        mask = category_mask(trip_known, trip_count)
        valid = [i for i, v in enumerate(mask) if v]
        if greedy or random.random() >= self.epsilon:
            with torch.no_grad():
                q = self.unmerge_actor.forward(
                    features.unsqueeze(0).to(self.device)).masked_fill(
                        ~mask.unsqueeze(0).to(self.device), float("-inf"))
            cat = int(q.argmax(dim=-1).item())
        else:
            cat = random.choice(valid)
        return cat, torch.zeros(()), mask

    def select_factor(self, features, category: int, trip_known=False,
                      trip_count=0, greedy=False) -> tuple:
        import random
        mask = category_factor_mask(category, trip_known, trip_count)
        valid = [i for i, v in enumerate(mask) if v]
        if not valid:
            # Unreachable: category_mask() removes unroll_only exactly when it
            # has no legal factor. Assert rather than fall back — returning
            # factor 1 here would emit (0,1), i.e. a NO-OP, while the entry is
            # recorded as unroll_only, and the head would be trained on an
            # action it did not take.
            raise AssertionError(
                f"category {category} has no valid factor (trip_known="
                f"{trip_known}, trip_count={trip_count}) — category_mask "
                f"should have excluded it")
        if greedy or random.random() >= self.epsilon:
            with torch.no_grad():
                q = self.factor_actor.forward(
                    features.unsqueeze(0).to(self.device)).masked_fill(
                        ~mask.unsqueeze(0).to(self.device), float("-inf"))
            idx = int(q.argmax(dim=-1).item())
        else:
            idx = random.choice(valid)
        return idx, torch.zeros(()), mask

    def ppo_update(self, buffer: RolloutBuffer) -> dict:
        tot_q = tot_v = 0.0
        n = 0
        for _ in range(self.K):
            for batch in buffer.sample_batches(self.batch_size):
                if not batch:
                    continue
                s1 = torch.stack([e.state1 for e in batch])
                s2 = torch.stack([e.state2 for e in batch])
                a1 = torch.tensor([e.action1 for e in batch], dtype=torch.long)
                a2 = torch.tensor([e.action2 for e in batch], dtype=torch.long)
                r = torch.tensor([e.reward for e in batch], dtype=torch.float32)
                m2 = torch.stack([
                    e.mask2 if e.mask2 is not None
                    else torch.ones(N_FACTORS, dtype=torch.bool) for e in batch])
                fa = torch.tensor([e.factor_active for e in batch],
                                  dtype=torch.bool)

                q2_all = self.factor_actor.forward(s2)
                q1 = self.unmerge_actor.forward(s1).gather(
                    1, a1.unsqueeze(1)).squeeze(1)
                values = self.critic.value(s1)

                with torch.no_grad():
                    boot = q2_all.detach().masked_fill(
                        ~m2, float("-inf")).max(dim=1).values
                    # Optional pessimism on the BOOTSTRAPPED arms only. The
                    # no-op target is exact, so penalising it would be pure bias.
                    boot = boot - self.q_pessimism
                    # no-op's value is known exactly; never bootstrap it.
                    q1_target = torch.where(a1 == NOOP,
                                            torch.zeros_like(boot), boot)

                # Q2 trains only on entries that actually chose a factor. Under
                # the 3-way head the no-op has none, so including it would
                # regress the factor head toward 0 on an action it never took.
                q_loss = F.mse_loss(q1, q1_target)
                if fa.any():
                    q2 = q2_all[fa].gather(1, a2[fa].unsqueeze(1)).squeeze(1)
                    q_loss = q_loss + F.mse_loss(q2, r[fa])
                # Once, as agent.BanditAgent does — the logging line below
                # reuses it rather than rebuilding the same number.
                value_loss = F.mse_loss(values, r)
                loss = q_loss + self.value_loss_coef * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self._all_params, self.max_grad_norm)
                self.optimizer.step()
                tot_q += float(q_loss)
                tot_v += float(value_loss)
                n += 1
        d = max(n, 1)
        return {"actor_loss": tot_q / d, "value_loss": tot_v / d,
                "entropy": self.epsilon}
