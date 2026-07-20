"""
Clipped Policy Gradient agent for the per-loop UU optimization pipeline.

Two actor networks (UnmergeActor, FactorActor) and one critic network (Critic)
share one Adam optimizer.  The critic provides a feature-conditioned baseline
V(state1) for variance reduction — replacing the previous global EMA scalar.
No-op actions (unmerge=0, factor=1) are handled by the environment returning
reward=0 directly; the critic learns to predict 0 for those loop feature regions.

Hyperparameters (all tunable via constructor):
  clip_eps         = 0.2   PPO clip range
  K                = 4     update epochs per rollout
  batch_size       = 8
  lr               = 3e-4
  value_loss_coef  = 0.5   weight on critic MSE loss
  entropy_coef     = 0.01  weight on entropy bonus (encourages exploration)

Entropy regularization:
  Both actors have an entropy bonus subtracted from the total loss, weighted by
  entropy_coef.  This prevents premature policy collapse to a deterministic
  choice (e.g. always unmerge=0, or always factor=2) before the agent has seen
  enough loops to know when each action is profitable.  The unmerge actor (binary)
  is especially prone to early collapse given the small action space.
  Entropy is computed from the CURRENT policy logits each batch, not the old
  log-probs used in the clipped ratio — this is the standard PPO formulation.
  Start with entropy_coef=0.01; increase if the no-op rate stays >80% early in
  training, decrease if rewards plateau without improvement.
"""

import logging
import random
from dataclasses import dataclass, field
from typing import Optional

import torch

_agent_log = logging.getLogger("agent")
import torch.nn as nn
import torch.nn.functional as F

# Action space for the factor decision.
# Factors 16/32 removed: measured 57-59% compile-failure rate and most compile
# timeouts in the first full run, for negligible policy value.  Reinstating
# them changes the FactorActor output dim — old checkpoints become incompatible.
FACTOR_VALUES: list[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
N_FEATURES: int = 18   # must match environment.py FEATURE_COLUMNS length
N_FACTORS: int = len(FACTOR_VALUES)

# Indices of tripCountKnown and tripCount within the RAW feature vector.
# Must stay in sync with FEATURE_COLUMNS in hecbench.py.
# NOTE: these indices are only meaningful on UN-normalised tensors.  The
# tensors fed to the networks are z-scored by FeatureNormalizer, so the mask
# must never be derived from them — build_factor_mask takes raw scalars instead.
_IDX_TRIP_COUNT_KNOWN: int = 10
_IDX_TRIP_COUNT: int = 11


def build_factor_mask(trip_known: bool, trip_count: int) -> torch.Tensor:
    """
    Return a bool mask of shape (N_FACTORS,) over FACTOR_VALUES.

    mask[i] = True  if FACTOR_VALUES[i] is a valid choice for this loop.
    mask[i] = False if FACTOR_VALUES[i] exceeds the loop's known trip count.

    Takes RAW (un-normalised) trip-count scalars.  Deriving these from the
    z-scored feature tensor is incorrect: a normalised tripCount of e.g. -0.27
    truncates to 0 and the mask silently never applies, while LLVM still caps
    the factor — mislabelling the recorded action against the observed reward.

    Trip count is invariant under unmerge (path specialisation does not change
    the iteration count), so the pre-unmerge raw values are valid for the
    factor decision on both the unmerge=0 and unmerge=1 branches.

    When the trip count is unknown all factors are valid.
    factor=1 (no-op unroll) is always valid regardless of trip count.
    """
    if trip_known and trip_count > 0:
        return torch.tensor(
            [f == 1 or f <= trip_count for f in FACTOR_VALUES], dtype=torch.bool
        )
    return torch.ones(N_FACTORS, dtype=torch.bool)


# ---------------------------------------------------------------------------
# Network definitions
# ---------------------------------------------------------------------------

class _MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UnmergeActor(_MLP):
    """Policy 1: pre-unmerge features → P(unmerge ∈ {0, 1})."""
    def __init__(self) -> None:
        super().__init__(N_FEATURES, 2)

    def log_prob(self, features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        logits = self.forward(features)
        return F.log_softmax(logits, dim=-1).gather(1, actions.unsqueeze(1)).squeeze(1)

    def sample(
        self, features: torch.Tensor, greedy: bool = False
    ) -> tuple[int, torch.Tensor]:
        """greedy=True takes the argmax action (deployment / evaluation mode)."""
        with torch.no_grad():
            logits = self.forward(features.unsqueeze(0))
            if greedy:
                action = logits.argmax(dim=-1)
            else:
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
        log_p = F.log_softmax(logits, dim=-1)[0, action.item()]
        return int(action.item()), log_p.detach()


class FactorActor(_MLP):
    """Policy 2: features → P(factor_idx ∈ 0..11).

    Supports action masking: pass mask=tensor([True, True, False, ...]) to
    zero out invalid factor choices (e.g. factors exceeding a known trip count)
    before sampling.  Invalid actions receive -inf logit and zero probability.
    """
    def __init__(self) -> None:
        super().__init__(N_FEATURES, N_FACTORS)

    def log_prob(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Log prob of *actions* under the current policy, optionally under the
        same action mask used at collection time.  The mask MUST match the one
        applied when the action was sampled — otherwise the PPO ratio compares
        a masked old-policy log-prob against an unmasked new-policy log-prob,
        biasing the update for every loop with a known trip count.
        """
        logits = self.forward(features)
        if mask is not None:
            logits = logits.masked_fill(~mask.to(logits.device), float("-inf"))
        return F.log_softmax(logits, dim=-1).gather(1, actions.unsqueeze(1)).squeeze(1)

    def sample(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        greedy: bool = False,
    ) -> tuple[int, torch.Tensor]:
        """greedy=True takes the argmax of the (masked) logits."""
        with torch.no_grad():
            logits = self.forward(features.unsqueeze(0))[0]
            if mask is not None:
                logits = logits.masked_fill(~mask.to(logits.device), float("-inf"))
            if greedy:
                action = logits.argmax(dim=-1)
            else:
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
        log_p = F.log_softmax(logits, dim=-1)[action.item()]
        return int(action.item()), log_p.detach()


class Critic(_MLP):
    """
    Value network: pre-unmerge features → scalar expected reward V(state1).

    Conditions on the loop's intrinsic structural properties before any
    transformation decision, learning a separate expected return for each
    region of feature space.  Replaces the previous global EMA baseline.
    """
    def __init__(self) -> None:
        super().__init__(N_FEATURES, 1)

    def value(self, features: torch.Tensor) -> torch.Tensor:
        """Return scalar value estimates, shape (batch,)."""
        return self.forward(features).squeeze(-1)


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

@dataclass
class RolloutEntry:
    state1: torch.Tensor      # pre-unmerge features  → input to UnmergeActor + Critic
    state2: torch.Tensor      # post-unmerge (or original) features → input to FactorActor
    action1: int              # unmerge decision (0 or 1)
    action2: int              # factor_idx (0–11)
    log_prob1: torch.Tensor   # log prob under policy at collection time
    log_prob2: torch.Tensor
    reward: float
    mask2: Optional[torch.Tensor] = None  # factor mask at collection time
                                          # (bool, (N_FACTORS,)); None = all valid


class RolloutBuffer:
    def __init__(self, capacity: int = 128) -> None:
        self.capacity = capacity
        self._entries: list[RolloutEntry] = []

    def append(self, entry: RolloutEntry) -> None:
        self._entries.append(entry)

    def full(self) -> bool:
        return len(self._entries) >= self.capacity

    def clear(self) -> None:
        self._entries = []

    def __len__(self) -> int:
        return len(self._entries)

    def sample_batches(self, batch_size: int) -> list[list[RolloutEntry]]:
        entries = list(self._entries)
        random.shuffle(entries)
        return [entries[i:i + batch_size] for i in range(0, len(entries), batch_size)]


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    def __init__(
        self,
        *,
        clip_eps: float = 0.2,
        K: int = 2,
        batch_size: int = 8,
        lr: float = 3e-4,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        device: Optional[torch.device] = None,
    ) -> None:
        self.clip_eps = clip_eps
        self.K = K
        self.batch_size = batch_size
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.device = device or torch.device("cpu")

        self.unmerge_actor = UnmergeActor().to(self.device)
        self.factor_actor = FactorActor().to(self.device)
        self.critic = Critic().to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.unmerge_actor.parameters())
            + list(self.factor_actor.parameters())
            + list(self.critic.parameters()),
            lr=lr,
        )

    def predict_value(self, features: torch.Tensor) -> float:
        """Return the critic's expected reward for a single loop feature vector."""
        with torch.no_grad():
            return self.critic.value(features.unsqueeze(0).to(self.device)).item()

    def select_unmerge(
        self, features: torch.Tensor, greedy: bool = False
    ) -> tuple[int, torch.Tensor]:
        """Sample unmerge action (argmax if greedy). Returns (action, log_prob)."""
        return self.unmerge_actor.sample(features.to(self.device), greedy=greedy)

    def select_factor(
        self,
        features: torch.Tensor,
        trip_known: bool = False,
        trip_count: int = 0,
        loop_idx: Optional[int] = None,
        greedy: bool = False,
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        """
        Sample factor action with trip-count masking.
        Returns (factor_idx, log_prob, mask).

        *trip_known* / *trip_count* must be the RAW (un-normalised) values —
        *features* is the z-scored tensor fed to the network and cannot be
        used to recover the trip count.  The returned mask must be stored in
        the RolloutEntry so ppo_update reapplies it when recomputing log-probs.
        """
        mask = build_factor_mask(trip_known, trip_count)
        if not mask.all():
            masked_out = [FACTOR_VALUES[i] for i, v in enumerate(mask) if not v]
            _agent_log.info(
                "  trip-count mask applied | loop_idx=%s tripCount=%d "
                "masked_factors=%s",
                loop_idx if loop_idx is not None else "?",
                trip_count,
                masked_out,
            )
        factor_idx, log_p = self.factor_actor.sample(
            features.to(self.device), mask=mask.to(self.device), greedy=greedy
        )
        return factor_idx, log_p, mask

    def ppo_update(self, buffer: RolloutBuffer) -> dict[str, float]:
        """Run K epochs of clipped mini-batch updates over the rollout buffer."""
        total_actor_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        # Pre-compute global advantage statistics for normalisation.
        # Critic values are computed fresh each K epoch so normalisation
        # uses the current value estimates rather than stale ones.
        for k in range(self.K):
            # Recompute values for the full buffer at the start of each epoch
            # so normalisation reflects the critic's current predictions.
            with torch.no_grad():
                all_s1 = torch.stack([e.state1 for e in buffer._entries]).to(self.device)
                all_values = self.critic.value(all_s1)
                all_rewards = torch.tensor(
                    [e.reward for e in buffer._entries],
                    dtype=torch.float32, device=self.device
                )
                advantages_raw = all_rewards - all_values
                adv_mean = advantages_raw.mean()
                adv_std = advantages_raw.std(correction=0) + 1e-8

            for batch in buffer.sample_batches(self.batch_size):
                if not batch:
                    continue

                s1 = torch.stack([e.state1 for e in batch]).to(self.device)
                s2 = torch.stack([e.state2 for e in batch]).to(self.device)
                a1 = torch.tensor([e.action1 for e in batch], dtype=torch.long, device=self.device)
                a2 = torch.tensor([e.action2 for e in batch], dtype=torch.long, device=self.device)
                old_lp1 = torch.stack([e.log_prob1 for e in batch]).to(self.device)
                old_lp2 = torch.stack([e.log_prob2 for e in batch]).to(self.device)
                r = torch.tensor([e.reward for e in batch], dtype=torch.float32, device=self.device)
                # Collection-time factor masks (all-valid for entries without one)
                m2 = torch.stack([
                    e.mask2 if e.mask2 is not None
                    else torch.ones(N_FACTORS, dtype=torch.bool)
                    for e in batch
                ]).to(self.device)

                # Critic forward (not detached — value loss trains the critic)
                values = self.critic.value(s1)

                # Advantage: use critic baseline, normalised by buffer-level stats
                adv = ((r - values.detach()) - adv_mean) / adv_std

                # Clipped actor losses.  The factor actor's new log-probs are
                # computed under the SAME mask used at collection time so the
                # PPO ratio compares like-for-like distributions.
                new_lp1 = self.unmerge_actor.log_prob(s1, a1)
                new_lp2 = self.factor_actor.log_prob(s2, a2, mask=m2)
                actor_loss = _clipped_pg_loss(new_lp1, old_lp1, adv, self.clip_eps)
                actor_loss = actor_loss + _clipped_pg_loss(new_lp2, old_lp2, adv, self.clip_eps)

                # Entropy bonus — computed from CURRENT policy logits (not old
                # log-probs) so it reflects the policy being updated this step.
                # Subtracted from total loss to maximise entropy (encourage
                # exploration).  Tracked separately so training logs can show
                # whether the policy is collapsing prematurely.
                # Factor entropy respects the collection-time mask: invalid
                # factors carry zero probability, so pushing entropy toward
                # them would be pushing probability mass onto unreachable actions.
                factor_logits = self.factor_actor.forward(s2).masked_fill(
                    ~m2, float("-inf")
                )
                entropy1 = _policy_entropy(self.unmerge_actor.forward(s1))
                entropy2 = _policy_entropy(factor_logits)
                entropy = entropy1 + entropy2

                # Critic (value) loss: MSE against actual rewards
                value_loss = F.mse_loss(values, r)

                loss = (
                    actor_loss
                    + self.value_loss_coef * value_loss
                    - self.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_actor_loss += actor_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                num_updates += 1

        n = max(num_updates, 1)
        return {
            "actor_loss": total_actor_loss / n,
            "value_loss": total_value_loss / n,
            "entropy": total_entropy / n,
        }

    def save(self, path: str) -> None:
        torch.save({
            "unmerge_actor": self.unmerge_actor.state_dict(),
            "factor_actor": self.factor_actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.unmerge_actor.load_state_dict(ckpt["unmerge_actor"])
        self.factor_actor.load_state_dict(ckpt["factor_actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.optimizer.load_state_dict(ckpt["optimizer"])


def _policy_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Mean entropy of a categorical distribution over a batch of logits.

    H = -sum(p * log p), averaged over the batch.  Used as the entropy bonus
    term in the PPO loss.  Computed from the CURRENT policy logits so it
    reflects the distribution being trained, not the behaviour policy at
    collection time.
    """
    return torch.distributions.Categorical(logits=logits).entropy().mean()


def _clipped_pg_loss(
    log_prob_new: torch.Tensor,
    log_prob_old: torch.Tensor,
    advantage: torch.Tensor,
    clip_eps: float,
) -> torch.Tensor:
    ratio = (log_prob_new - log_prob_old).exp()
    clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps)
    return -torch.min(ratio * advantage, clipped * advantage).mean()
