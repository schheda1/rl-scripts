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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import torch                                                     # noqa: E402
import torch.nn.functional as F                                  # noqa: E402

from adapt_eval import make_state                                # noqa: E402
from agent import FACTOR_VALUES, N_FACTORS, FactorActor          # noqa: E402
from category_agent import (UNMERGE_UNROLL, UNROLL_ONLY,         # noqa: E402
                            category_factor_mask)
from offline_data import (IDX_TRIP_COUNT, IDX_TRIP_KNOWN, NOOP,  # noqa: E402
                          best_constant_factor,
                          labelled_loops, load_run, loops_for,
                          score_decisions)
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
                 epsilon: float, batch_size: int, K: int) -> None:
        self.factor_actor = FactorActor(logit_cap=0.0).to(DEVICE)
        self.epsilon = epsilon
        self.max_grad_norm = max_grad_norm
        self.batch_size, self.K = batch_size, K
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

    def update(self, buf: list) -> float:
        """MSE(Q[chosen], reward) over the buffer. Mirrors the bandit's Q2 term
        (category_agent.py:392-394) with Q1, the critic and the bootstrap gone —
        there is no first stage to back a value up from."""
        tot, n = 0.0, 0
        for _ in range(self.K):
            order = list(range(len(buf)))
            random.shuffle(order)
            for i in range(0, len(order), self.batch_size):
                batch = [buf[j] for j in order[i:i + self.batch_size]]
                if not batch:
                    continue
                s = torch.stack([b[0] for b in batch])
                a = torch.tensor([b[1] for b in batch], dtype=torch.long)
                r = torch.tensor([b[2] for b in batch], dtype=torch.float32)
                q = self.factor_actor.forward(s).gather(
                    1, a.unsqueeze(1)).squeeze(1)
                loss = F.mse_loss(q, r)
                self.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self._all_params,
                                                   self.max_grad_norm)
                self.optimizer.step()
                tot += float(loss)
                n += 1
        return tot / max(n, 1)


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
    agent = FactorOnly(args.lr, args.weight_decay, args.max_grad_norm,
                       args.epsilon, args.batch_size, args.K)

    cells = [(l, u, s, m) for l in loops for u, s, m in states_for(l, data)]
    # Denominator for coverage: every LEGAL (loop, branch, factor) cell. Legal,
    # not all 10 — factor 1 is illegal on the pre-unmerge branch, so counting 10
    # there would cap coverage below 100% and make a full sweep look partial.
    n_legal = sum(int(m.sum()) for _, _, _, m in cells)
    seen = set()

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
            buf.append((s, idx, r))
            if len(buf) >= args.buffer_size:
                loss_sum += agent.update(buf); n_upd += 1
                buf = []
        if buf:
            loss_sum += agent.update(buf); n_upd += 1
            buf = []
        cov = len(seen) / n_legal if n_legal else float("nan")
        history.append({"epoch": epoch, "q_loss": loss_sum / max(n_upd, 1),
                        "epsilon": agent.epsilon, "coverage": cov,
                        "updates": n_upd})
        if args.log_every and (epoch % args.log_every == 0
                               or epoch in (1, args.epochs)):
            log.info("      epoch %4d | q_loss %.5f | eps %.3f | coverage %5.1f%%",
                     epoch, history[-1]["q_loss"], agent.epsilon, 100 * cov)
    return agent, (len(seen) / n_legal if n_legal else float("nan")), history


def evaluate(agent: FactorOnly, loops: list, data: dict, args) -> dict:
    """Probe on the true branch, plus each state population separately."""
    m = score_decisions(probe_picks(agent, loops, data), data["tables"],
                        data["labels"], args.deadzone, args.missing,
                        data.get("failures"))
    _, bar, _ = best_constant_factor(loops, data["tables"], data["labels"],
                                     args.deadzone, args.missing)
    out = {"probe": m["capture"], "probe_n": m["loops_with_headroom"],
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
           "probe_clipped": m["n_clipped"]}
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
    log.info("Loaded %d loops / %d benchmarks | %d labelled | %d states\n",
             len(data["loops"]), len(data["benchmarks"]),
             data["n_labelled_loops"], 2 * len(loops))

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
            fit, test = evaluate(agent, fit_l, data, args), \
                evaluate(agent, test_l, data, args)
            rows.append({"split": sp + 1, "split_seed": sseed, "seed": seed,
                         "n_fit_loops": len(fit_l), "n_test_loops": len(test_l),
                         "n_test_benchmarks": len(te_b), "coverage": cov,
                         "q_loss_first": hist[0]["q_loss"],
                         "q_loss_final": hist[-1]["q_loss"],
                         "fit": fit, "test": test})
            curve.extend(dict(split=sp + 1, seed=seed, **h) for h in hist)
            log.info("  split %d/%d seed %d | %3d/%3d loops | coverage %5.1f%% | "
                     "probe fit %5.1f%% (bar %5.1f%%)  test %5.1f%% (bar %5.1f%%)",
                     sp + 1, args.splits, seed, len(fit_l), len(test_l),
                     100 * cov,
                     100 * fit["probe"], 100 * fit["probe_bar"],
                     100 * test["probe"], 100 * test["probe_bar"])

    report(rows)
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


def report(rows: list) -> None:
    if not rows:
        log.error("no folds produced a result")
        return
    cov = sum(r["coverage"] for r in rows) / len(rows)
    log.info("\n" + "=" * 78)
    log.info("  UNROLLER IN ISOLATION — no category head")
    log.info("=" * 78)
    log.info("  mean cell coverage %.1f%% of legal (loop, branch, factor) cells."
             "\n  High coverage with a poor result is a LEARNING failure; low "
             "coverage is an\n  exploration failure. That is what this column "
             "is for.\n", 100 * cov)
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
