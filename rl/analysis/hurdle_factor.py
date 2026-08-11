"""
Hurdle + censored unroller — a drop-in replacement for factor_only.FactorOnly.

THE DEFECT IT ADDRESSES
-----------------------
`FactorOnly.update` is `F.mse_loss(q, r)` on the raw stored value
(factor_only.py:127). That value is not one quantity. Of 8,352 cells in this
cache, ~2,621 are compile failures carrying --compile-failure-penalty and 744
sit at the -1.0 clip. So more than a third of the regression targets are not
performance measurements, and the gradient goes into separating -1.0 from -0.161
rather than +0.19 from 0.0 — which is what factor_rank.py's docstring already
argued, before factor_only was built without it.

Worse, the failure penalty is a number someone chose. What a failed pick
actually realises is 0.0: the transform did not build, the compiler keeps the
original, the program is unchanged. score_decisions already scores it that way
(`r_applied = 0.0 if failed else r`). So the quantity the scorer measures is

    E[realised]  =  P(ok | loop, f) * E[r | loop, f, ok]

and nothing in the current model predicts either factor of it.

THE MODEL
---------
Two heads over the same input, no shared trunk — two FactorActors, so the
shapes, the init and the optimizer grouping are the ones already validated
elsewhere in this study:

    feas : s -> 10 logits    P(the cell builds and runs)
    mag  : s -> 10 means     E[r | it ran],  with a learned scalar sigma

and a three-part likelihood keyed on the OBSERVED regime:

    ok        Gaussian NLL against the measured value.
    censored  -log Phi((c - mu)/sigma). The -1.0 clip means "at least this bad";
              fitting it as a point target pulls mu toward a value that was
              never measured. 744 cells do that today.
    failed    excluded from `mag` entirely. It contributes only its 0 label to
              `feas`. No gradient reaches the magnitude head.

`--floor-as failed` moves the censored cells to the feasibility-negative class
instead, for the reading where the -1.0 population is dominated by compile
timeouts (nothing shipped, so the realised effect is 0.0 as well). The two
readings are not separable from the cache — `is_timeout` never reaches it — so
both are reachable by a flag rather than one being asserted.

WHY IT IS A DROP-IN
-------------------
`select(state, mask, greedy)` and `update(buf)` match FactorOnly's signatures,
so factor_only.probe_picks and factor_only.branch_picks — which only ever call
`agent.select(s, mask, greedy=True)` — score this head unmodified, through the
same score_decisions path. The numbers are directly comparable to the runs
already on disk. Only the buffer payload widens, from (state, idx, reward) to
(state, idx, status, value).

`select` deliberately does NOT decline. The scored populations in factor_only
force the branch, so a head that could answer "no factor" would be measuring a
different thing. The decline signal is available through `predict` and the
runner reports how often the best score would have fallen below the deadzone.
"""

import copy
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import random                                                    # noqa: E402

import torch                                                     # noqa: E402
import torch.nn.functional as F                                  # noqa: E402

from agent import FactorActor                                    # noqa: E402
from cell_status import CENSORED, CLIP_FLOOR, OK                 # noqa: E402

DEVICE = torch.device("cpu")

FLOOR_AS = ("censored", "failed")

# sigma is learned in log space and clamped. Unclamped, the Gaussian term is
# minimised by driving sigma to 0 on any subset the head fits exactly, which
# sends the NLL to -inf and the gradients with it. The upper bound stops the
# opposite degeneracy, where a large sigma makes every residual cheap and mu
# stops being pulled anywhere.
_LOG_SIGMA_MIN = math.log(1e-2)
_LOG_SIGMA_MAX = math.log(10.0)
_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)


class HurdleFactor:
    """Feasibility head + censored magnitude head. Interface-compatible with
    factor_only.FactorOnly."""

    def __init__(self, lr: float, weight_decay: float, max_grad_norm: float,
                 epsilon: float, batch_size: int, K: int,
                 floor_as: str = "censored", mag_coef: float = 1.0,
                 head_factory=None) -> None:
        if floor_as not in FLOOR_AS:
            raise ValueError(f"floor_as must be one of {FLOOR_AS}, got {floor_as!r}")
        self.floor_as = floor_as
        self.mag_coef = mag_coef
        self.epsilon = epsilon
        self.max_grad_norm = max_grad_norm
        self.batch_size, self.K = batch_size, K

        # head_factory lets a caller swap in FactorScorer without this module
        # importing it — the ordinality question is orthogonal to the hurdle
        # question and mixing them would make a result unattributable.
        make = head_factory or (lambda: FactorActor(logit_cap=0.0))
        self.feas = make().to(DEVICE)
        self.mag = make().to(DEVICE)
        self.log_sigma = torch.nn.Parameter(torch.zeros((), device=DEVICE))

        # MIRROR: FactorOnly.__init__ — 2-D tensors decay, everything else does
        # not. log_sigma is 0-dim, so it lands in no_decay, which is what it
        # wants: weight decay on a variance parameter shrinks it toward 1.0 for
        # no reason.
        decay, no_decay = [], []
        for m in (self.feas, self.mag):
            for p in m.parameters():
                (decay if p.ndim >= 2 else no_decay).append(p)
        no_decay.append(self.log_sigma)
        self._all_params = decay + no_decay
        self.optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": weight_decay},
             {"params": no_decay, "weight_decay": 0.0}], lr=lr)

        self.last_feas_loss = float("nan")
        self.last_mag_loss = float("nan")

        # Per-target output offsets, used ONLY by few-shot adaptation. Kept as
        # plain floats rather than Parameters so they are invisible to the
        # optimizer during ordinary training and cannot drift: a base model must
        # score identically whether or not adaptation was ever run. One scalar
        # per head is the lowest-variance thing k measurements can fit, which is
        # the point — factor_adapt's 650-parameter readout was being estimated
        # from ~38 noisy cells.
        self.feas_bias = 0.0
        self.mag_bias = 0.0

    # -- regime bookkeeping -------------------------------------------------

    def ran(self, status: str) -> bool:
        """Did this cell produce a running program? Under floor_as='failed' a
        clip-floor cell is read as a timeout, which shipped nothing."""
        if status == OK:
            return True
        if status == CENSORED:
            return self.floor_as == "censored"
        return False

    # -- inference ----------------------------------------------------------

    def predict(self, state: torch.Tensor, mask: torch.Tensor) -> tuple:
        """(p_ok, mu, score) each (N_FACTORS,), score -inf on illegal factors.

        score = p_ok * mu is the expected realised delta: a cell that does not
        build leaves the program unchanged at exactly 0.0, so the failure branch
        contributes nothing rather than contributing a penalty.
        """
        with torch.no_grad():
            p_ok, mu = self.predict_batch(state.unsqueeze(0))
            p_ok, mu = p_ok[0], mu[0]
            score = (p_ok * mu).masked_fill(~mask.to(p_ok.device), float("-inf"))
        return p_ok, mu, score

    def predict_batch(self, states: torch.Tensor) -> tuple:
        """(p_ok, mu), each (B, N_FACTORS). No masking — callers that need it
        hold the per-loop masks. Adaptation offsets are applied here, so every
        inference path picks them up without knowing they exist."""
        with torch.no_grad():
            p_ok = torch.sigmoid(
                self.feas.forward(states.to(DEVICE)) + self.feas_bias)
            mu = self.mag.forward(states.to(DEVICE)) + self.mag_bias
        return p_ok, mu

    def logits_and_mu(self, states: torch.Tensor) -> tuple:
        """Graph-carrying (feas_logits, mu) for the adaptation fit. Same offsets
        as predict_batch, so what adaptation optimises is what inference reads."""
        return (self.feas.forward(states.to(DEVICE)) + self.feas_bias,
                self.mag.forward(states.to(DEVICE)) + self.mag_bias)

    def select(self, state: torch.Tensor, mask: torch.Tensor,
               greedy: bool = False) -> int:
        """Index into FACTOR_VALUES. Epsilon-greedy over the LEGAL factors.

        MIRROR: FactorOnly.select — same signature, same exploration rule, so
        the only thing that differs between the two agents is what the score
        means.

        WORTH KNOWING, because it looks like a bug and is not: when every legal
        factor has mu < 0, `p_ok * mu` is maximised by the SMALLEST p_ok, so the
        argmax prefers the cell most likely to fail. That is correct under this
        model — a failure realises exactly 0.0, which beats a slowdown — and it
        is the same rule score_decisions applies with `r_applied`. In deployment
        the right answer there is to decline, which the runner reads off
        `predict`; this method never declines because the scored populations
        force the branch.

        The consequence for TRAINING is real and is why the runner logs a
        ran-rate: once the model believes a branch is hopeless, greedy picks
        drift toward cells it expects to fail, which teach the feasibility head
        and give the magnitude head nothing. Epsilon and the per-epoch sweep over
        every cell are what keep that from closing off.
        """
        legal = [i for i, v in enumerate(mask) if v]
        if not legal:
            raise AssertionError("no legal factor for this state")
        if not greedy and random.random() < self.epsilon:
            return random.choice(legal)
        _, _, score = self.predict(state, mask)
        return int(score.argmax())

    # -- training -----------------------------------------------------------

    def _sigma(self) -> torch.Tensor:
        return self.log_sigma.clamp(_LOG_SIGMA_MIN, _LOG_SIGMA_MAX).exp()

    def _batch_loss(self, batch: list) -> tuple:
        """(total, feas_loss, mag_loss) for one minibatch of
        (state, factor_idx, status, value)."""
        s = torch.stack([b[0] for b in batch]).to(DEVICE)
        a = torch.tensor([b[1] for b in batch], dtype=torch.long, device=DEVICE)
        ran = torch.tensor([self.ran(b[2]) for b in batch],
                           dtype=torch.bool, device=DEVICE)

        all_logits, all_mu = self.logits_and_mu(s)

        # Feasibility over EVERY sample, including the failures. This is the
        # half that gets a clean 0/1 label out of every measurement, and the
        # half the current scalar model spends its capacity regressing onto a
        # chosen penalty instead.
        logit = all_logits.gather(1, a.unsqueeze(1)).squeeze(1)
        feas_loss = F.binary_cross_entropy_with_logits(logit, ran.float())

        # Magnitude over the samples that RAN only. `failed` never reaches here,
        # so no gradient carries a fabricated -0.161 into mu.
        mag_loss = torch.zeros((), device=DEVICE)
        if bool(ran.any()):
            mu = all_mu.gather(1, a.unsqueeze(1)).squeeze(1)[ran]
            r = torch.tensor([b[3] for b in batch], dtype=torch.float32,
                             device=DEVICE)[ran]
            cens = torch.tensor(
                [b[2] == CENSORED for b in batch],
                dtype=torch.bool, device=DEVICE)[ran]
            sigma = self._sigma()

            # Both terms are evaluated on every element and selected with
            # `where`, rather than written into a buffer by boolean index. Both
            # are finite everywhere, so the unused half costs arithmetic and
            # nothing else — and this keeps the magnitude loss a pure function
            # with no in-place write into the autograd graph.
            #
            # Full Gaussian NLL, constant included, so the value logged is a
            # real log-likelihood and comparable across configurations.
            z = (r - mu) / sigma
            gauss = _HALF_LOG_2PI + torch.log(sigma) + 0.5 * z * z
            # Left-censored at the clip: all that is known is r <= c, so the
            # contribution is -log Phi((c - mu)/sigma). log_ndtr, not
            # log(cdf(...)) — the naive form underflows to -inf once mu sits a
            # few sigma above the floor, which is the common case here.
            censored_nll = -torch.special.log_ndtr((CLIP_FLOOR - mu) / sigma)
            mag_loss = torch.where(cens, censored_nll, gauss).mean()

        return feas_loss + self.mag_coef * mag_loss, feas_loss, mag_loss

    def update(self, buf: list) -> float:
        """K passes of shuffled minibatches. Returns the mean TOTAL loss, so the
        caller's `loss_sum += agent.update(buf)` works unchanged; the per-term
        breakdown is left on self.last_*_loss for the runner to log."""
        tot = f_tot = m_tot = 0.0
        n = 0
        for _ in range(self.K):
            order = list(range(len(buf)))
            random.shuffle(order)
            for i in range(0, len(order), self.batch_size):
                batch = [buf[j] for j in order[i:i + self.batch_size]]
                if not batch:
                    continue
                loss, f_loss, m_loss = self._batch_loss(batch)
                self.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self._all_params,
                                                   self.max_grad_norm)
                self.optimizer.step()
                tot += float(loss)
                f_tot += float(f_loss)
                m_tot += float(m_loss)
                n += 1
        n = max(n, 1)
        self.last_feas_loss = f_tot / n
        self.last_mag_loss = m_tot / n
        return tot / n

    # -- checkpointing ------------------------------------------------------

    def snapshot(self) -> dict:
        return {"feas": copy.deepcopy(self.feas.state_dict()),
                "mag": copy.deepcopy(self.mag.state_dict()),
                "log_sigma": self.log_sigma.detach().clone(),
                # Carried so restoring a base snapshot also clears whatever a
                # previous target's adaptation left behind. Omitting them would
                # leak one benchmark's offset into the next, which is the exact
                # failure offline_adapt's leak check exists to catch.
                "feas_bias": self.feas_bias, "mag_bias": self.mag_bias}

    def restore(self, snap: dict) -> None:
        """In-place, so self.optimizer keeps pointing at the same tensors —
        rebinding self.log_sigma instead would silently leave the optimizer
        updating a detached parameter.

        It does NOT reset the optimizer's Adam moments. Safe for every current
        caller, which restores only to EVALUATE (hurdle_run scores each rule,
        then moves on) or before an adaptation that builds its own optimizer.
        Resuming training from a restored snapshot would need that reset.
        """
        self.feas.load_state_dict(snap["feas"])
        self.mag.load_state_dict(snap["mag"])
        with torch.no_grad():
            self.log_sigma.copy_(snap["log_sigma"])
        self.feas_bias = snap.get("feas_bias", 0.0)
        self.mag_bias = snap.get("mag_bias", 0.0)

    @property
    def sigma(self) -> float:
        return float(self._sigma())
