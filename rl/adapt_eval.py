"""
Few-shot domain-adaptation study: does a policy that fits the training
distribution transfer to unseen benchmarks, and can a handful of measured
loops from a target benchmark recover the gap?

Runs ENTIRELY OFFLINE against the reward cache — no compiles, no GPU.  Every
evaluation is a table lookup, so the whole study reruns in seconds and is
exactly reproducible.  That requires the cells to be there: run
collect_cells.py on the splits first, or numbers will be computed over a
partial table (reported, never silently ignored).

THREE PARTS
-----------
  1. FIT          greedy policy over the full TRAIN split.  Not held out on
                  purpose — it establishes that the model can represent the
                  mapping at all, separating "cannot fit" from "cannot
                  transfer".
  2. ZERO-SHOT    the same checkpoint on TEST.  The generalization gap.
  3. ADAPTATION   per test benchmark, fine-tune on a few of ITS loops and
                  evaluate on the REST of that benchmark's loops.

The adaptation/evaluation split is at LOOP granularity within each benchmark.
Adapting on some actions of a loop and testing on other actions of the same
loop would measure action-space interpolation, which is a far weaker claim.

Benchmarks are assigned by loop count:
    >= 3 loops -> 2 adaptation loops
       2 loops -> 1 adaptation loop
       1 loop  -> 0; the loop is evaluated zero-shot only.
Those single-loop benchmarks are not waste: they are the control group,
measured in the same experiment under the same checkpoint, showing what
happens with no target adaptation at all.

METRIC
------
capture ratio = sum(realized reward) / sum(oracle reward), over loops where
the oracle has something to capture.  A ratio of sums, not a mean of per-loop
ratios, which explodes when a loop's oracle is near zero.  Reported three ways
because an average hides the thing that matters:
    per-benchmark mean   (primary — every application counts once)
    pooled over loops    (deployment-weighted; loop counts are heavily skewed,
                          one benchmark held 23 of 74 test loops)
    improved/flat/regressed benchmark COUNTS

Usage:
  python3 adapt_eval.py CKPT.pt [CKPT2.pt ...] --run-dir RUN --agent ppo \\
      --compile-failure-penalty -0.16 --reward-deadzone 0.005 --out study.csv
"""

import argparse
import csv
import json
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from hecbench import HECBENCH_SRC, discover_benchmarks
from train import build_loop_assignments, precheck_benchmarks, split_benchmarks

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("adapt")

NOOP = (0, 1)


# ---------------------------------------------------------------------------
# Ground-truth table (pure, torch-free — unit-testable without a GPU box)
# ---------------------------------------------------------------------------

def build_tables(assignments: list[dict], rewards: dict) -> dict:
    """
    {(bench, loop_idx): {(unmerge, factor): reward}} from the cache.

    The no-op is injected as an exact 0.0: it is free, always available, and
    never stored (train.py only caches cells it had to compile).  Omitting it
    would make the oracle claim a transform is required where declining is
    already optimal.
    """
    tables: dict = {}
    for a in assignments:
        key = (a["benchmark_name"], a["loop_idx"])
        tables[key] = {NOOP: 0.0}
    for k, v in rewards.items():
        parts = k.split("|")
        if len(parts) != 4:
            continue
        bench, li, u, f = parts
        try:
            cell_key = (bench, int(li))
            cell = (int(u), int(f))
        except ValueError:
            continue
        if cell_key in tables:
            tables[cell_key][cell] = float(v)
    return tables


def oracle_of(table: dict) -> tuple:
    """(best_action, best_reward) including the free no-op at 0.0."""
    best_a, best_r = NOOP, 0.0
    for a, r in table.items():
        if r > best_r:
            best_a, best_r = a, r
    return best_a, best_r


def score(picks: list, tables: dict, deadzone: float) -> dict:
    """
    Aggregate a list of (bench, loop_idx, chosen_action) into the metric set.

    `chosen_action` None means the policy picked a cell that has never been
    measured — with a complete table this cannot happen; with a partial one it
    is counted and excluded rather than silently scored as zero.
    """
    realized_sum = oracle_sum = 0.0
    n = n_scored = n_unmeasured = n_regress = n_headroom = 0
    per_bench: dict = {}
    for bench, li, action in picks:
        n += 1
        table = tables[(bench, li)]
        _, orc = oracle_of(table)
        if action is None or action not in table:
            n_unmeasured += 1
            continue
        r = table[action]
        n_scored += 1
        if r < -deadzone:
            n_regress += 1
        b = per_bench.setdefault(bench, {"realized": 0.0, "oracle": 0.0,
                                         "loops": 0, "regress": 0})
        b["loops"] += 1
        b["regress"] += int(r < -deadzone)
        # Only loops with headroom enter the capture ratio: a loop whose oracle
        # is 0 has nothing to capture, and including it makes the denominator
        # reward-free while the numerator can still go negative.
        if orc > deadzone:
            n_headroom += 1
            realized_sum += r
            oracle_sum += orc
            b["realized"] += r
            b["oracle"] += orc
    bench_caps = [v["realized"] / v["oracle"]
                  for v in per_bench.values() if v["oracle"] > deadzone]
    return {
        "loops": n,
        "loops_scored": n_scored,
        "loops_unmeasured": n_unmeasured,
        "loops_with_headroom": n_headroom,
        "capture_pooled": realized_sum / oracle_sum if oracle_sum > 0 else float("nan"),
        "capture_per_benchmark": (sum(bench_caps) / len(bench_caps)
                                  if bench_caps else float("nan")),
        "mean_realized": (realized_sum / n_headroom) if n_headroom else 0.0,
        "regression_rate": n_regress / n_scored if n_scored else 0.0,
        "n_benchmarks": len(per_bench),
        "per_bench": per_bench,
    }


def split_adaptation_loops(loops: list, rng: random.Random) -> tuple:
    """
    (adaptation_loops, evaluation_loops) under the agreed rule:
      >=3 loops -> 2 adapt;  2 loops -> 1 adapt;  1 loop -> 0 adapt (control).
    """
    if len(loops) >= 3:
        k = 2
    elif len(loops) == 2:
        k = 1
    else:
        k = 0
    shuffled = list(loops)
    rng.shuffle(shuffled)
    return shuffled[:k], shuffled[k:]


def marginal_policy(tables: dict, train_keys: list) -> list:
    """
    Actions ranked by mean reward over the TRAIN split, ignoring features.

    The baseline any feature-conditioned policy must beat: if the learned model
    only matches this, it has learned a global prior about which factor is
    usually good and nothing about the individual loop.
    """
    tot: dict = {}
    cnt: dict = {}
    for k in train_keys:
        for a, r in tables[k].items():
            tot[a] = tot.get(a, 0.0) + r
            cnt[a] = cnt.get(a, 0) + 1
    return [a for a, _ in sorted(((a, tot[a] / cnt[a]) for a in tot),
                                 key=lambda t: -t[1])]


# ---------------------------------------------------------------------------
# Policy evaluation (needs torch)
# ---------------------------------------------------------------------------

def make_state(loop: dict, normalizer, postf: dict, unmerge: int):
    """
    (s1, s2) for a loop.  MIRROR: train._worker_fn — s2 is the post-unmerge
    vector only on the unmerge branch, and falls back to s1 when no vector was
    extracted, exactly as the live pipeline does.
    """
    import torch
    s1 = normalizer.normalize(
        torch.tensor(loop["pre_features_raw"], dtype=torch.float32))
    if unmerge == 1:
        pf = postf.get(f"{loop['benchmark_name']}|{loop['loop_idx']}")
        if pf is not None:
            return s1, torch.tensor(pf, dtype=torch.float32)
    return s1, s1


def greedy_pick(agent, loop: dict, normalizer, postf: dict):
    """
    The policy's deployment action for one loop.

    MIRROR: train._worker_fn — unmerge decided on s1, factor decided on the
    state matching that decision, trip-count mask built from RAW features.
    Identical for Agent and BanditAgent: both expose greedy argmax selection.
    """
    from agent import FACTOR_VALUES, _IDX_TRIP_COUNT_KNOWN, _IDX_TRIP_COUNT
    raw = loop["pre_features_raw"]
    s1, _ = make_state(loop, normalizer, postf, 0)
    unmerge, _ = agent.select_unmerge(s1, greedy=True)
    _, s2 = make_state(loop, normalizer, postf, unmerge)
    factor_idx, _, _ = agent.select_factor(
        s2,
        trip_known=raw[_IDX_TRIP_COUNT_KNOWN] > 0.5,
        trip_count=int(raw[_IDX_TRIP_COUNT]),
        loop_idx=loop["loop_idx"], greedy=True,
    )
    return (int(unmerge), FACTOR_VALUES[factor_idx])


def evaluate(agent, loops: list, normalizer, postf: dict, tables: dict,
             deadzone: float) -> dict:
    picks = [(l["benchmark_name"], l["loop_idx"],
              greedy_pick(agent, l, normalizer, postf)) for l in loops]
    return score(picks, tables, deadzone)


# ---------------------------------------------------------------------------
# Adaptation
# ---------------------------------------------------------------------------

def freeze_trunk(agent) -> list:
    """
    Freeze everything but each actor head's final projection.

    Two adaptation loops give ~20-40 supervised examples against ~60k
    parameters; full fine-tuning memorizes them instantly.  net[6] is the
    output Linear (0 Linear, 1 LayerNorm, 2 ReLU, 3 Linear, 4 LayerNorm,
    5 ReLU, 6 Linear).  The critic is left frozen entirely — it plays no part
    in greedy action selection.
    """
    trainable = []
    for module in (agent.unmerge_actor, agent.factor_actor):
        for p in module.parameters():
            p.requires_grad = False
        for p in module.net[6].parameters():
            p.requires_grad = True
            trainable.append(p)
    return trainable


def adapt(agent, loops: list, normalizer, postf: dict, tables: dict,
          kind: str, lr: float, steps: int):
    """
    Fine-tune on the adaptation loops' measured cells.

    The signal matches each agent's native one, so both consume the SAME data:
      ppo    — cross-entropy toward the oracle-best action (behaviour cloning;
               a policy-gradient step is impossible here, the cells carry no
               collection-time log-probs).
      bandit — MSE of Q(s,a) against the measured reward over every cell.
    """
    import torch
    import torch.nn.functional as F
    from agent import FACTOR_VALUES, _IDX_TRIP_COUNT_KNOWN, _IDX_TRIP_COUNT

    trainable = freeze_trunk(agent)
    if not loops or not trainable:
        return agent
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)

    samples = []
    for l in loops:
        table = tables[(l["benchmark_name"], l["loop_idx"])]
        raw = l["pre_features_raw"]
        mask_known = raw[_IDX_TRIP_COUNT_KNOWN] > 0.5
        trip = int(raw[_IDX_TRIP_COUNT])
        if kind == "ppo":
            (u, f), _ = oracle_of(table)
            s1, s2 = make_state(l, normalizer, postf, u)
            samples.append((s1, s2, u, FACTOR_VALUES.index(f), 0.0,
                            mask_known, trip))
        else:
            for (u, f), r in table.items():
                if f not in FACTOR_VALUES:
                    continue
                s1, s2 = make_state(l, normalizer, postf, u)
                samples.append((s1, s2, u, FACTOR_VALUES.index(f), r,
                                mask_known, trip))
    if not samples:
        return agent

    from agent import build_factor_mask
    s1b = torch.stack([s[0] for s in samples])
    s2b = torch.stack([s[1] for s in samples])
    a1b = torch.tensor([s[2] for s in samples], dtype=torch.long)
    a2b = torch.tensor([s[3] for s in samples], dtype=torch.long)
    rb = torch.tensor([s[4] for s in samples], dtype=torch.float32)
    # Same trip-count mask the policy uses at selection time.  Without it,
    # behaviour cloning spends probability mass on factors that are masked out
    # at evaluation — training masks before log_softmax, so this keeps the
    # adaptation objective on the same distribution the policy is scored on.
    m2b = torch.stack([build_factor_mask(s[5], s[6]) for s in samples])

    for _ in range(steps):
        opt.zero_grad()
        if kind == "ppo":
            f_logits = agent.factor_actor.forward(s2b).masked_fill(
                ~m2b, float("-inf"))
            loss = (F.cross_entropy(agent.unmerge_actor.forward(s1b), a1b)
                    + F.cross_entropy(f_logits, a2b))
        else:
            q1 = agent.unmerge_actor.forward(s1b).gather(1, a1b.unsqueeze(1)).squeeze(1)
            q2 = agent.factor_actor.forward(s2b).gather(1, a2b.unsqueeze(1)).squeeze(1)
            loss = F.mse_loss(q1, rb) + F.mse_loss(q2, rb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 0.5)
        opt.step()
    return agent


def fresh_agent(kind: str, ckpt: "Path | None", logit_cap: float = 0.0):
    """
    A new agent, optionally loaded from a checkpoint (CPU, eval-only).

    logit_cap must match the training run for PPO checkpoints.  It adds no
    parameters, so the state_dict loads either way and greedy argmax is
    identical (tanh is monotone) — but it bounds what the ADAPTATION step can
    do to the logits, so leaving it off would let fine-tuning push the policy
    into a saturation regime training never allowed.  Bandit heads are
    Q-values, not logits, and are never capped.
    """
    import torch
    from agent import Agent, BanditAgent
    if kind == "bandit":
        a = BanditAgent(device=torch.device("cpu"))
    else:
        a = Agent(device=torch.device("cpu"), logit_cap=logit_cap)
    if ckpt is not None:
        a.load(str(ckpt))
    for m in (a.unmerge_actor, a.factor_actor, a.critic):
        m.eval()
    return a


# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoints", nargs="+", type=Path)
    p.add_argument("--run-dir", type=Path, required=True,
                   help="Dir with eligible_benchmarks.json and reward_cache.json")
    p.add_argument("--agent", choices=("ppo", "bandit"), default="ppo",
                   help="Must match how the checkpoints were trained")
    p.add_argument("--reward-deadzone", type=float, required=True,
                   help="REQUIRED: must match the run that built the cache")
    p.add_argument("--seeds", type=int, default=5,
                   help="Random draws of the adaptation loops (default: 5)")
    p.add_argument("--adapt-lr", type=float, default=1e-3)
    p.add_argument("--adapt-steps", type=int, default=50)
    p.add_argument("--min-fill", type=float, default=0.0,
                   help="Restrict the FIT part to loops with at least this "
                        "fraction of their cells measured. Train coverage is "
                        "partial, and a partial oracle is a LOWER bound, which "
                        "flatters capture — set 0.9 for the fair comparison "
                        "against a fully-covered test split.")
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--logit-cap", type=float, default=4.0,
                   help="Must match the training run's --logit-cap "
                        "for PPO checkpoints (default: 4.0). Ignored "
                        "for --agent bandit.")
    p.add_argument("--hecbench-src", default=None)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    src = Path(args.hecbench_src) if args.hecbench_src else HECBENCH_SRC
    all_b, _, records, normalizer = precheck_benchmarks(
        discover_benchmarks(src), args.run_dir / "eligible_benchmarks.json",
        skip=True, strict=True)
    train_b, val_b, test_b = split_benchmarks(
        all_b, args.val_ratio, args.test_ratio, args.split_seed)

    cache = json.loads((args.run_dir / "reward_cache.json").read_text())
    rewards = cache.get("rewards", {})
    postf = cache.get("post_features", {})

    train_loops = build_loop_assignments(train_b, records)
    test_loops = build_loop_assignments(test_b, records)
    tables = build_tables(train_loops + test_loops, rewards)

    # Coverage: an incomplete table makes the oracle a lower bound and the
    # capture ratio an over-estimate.  Report it before any number is read.
    from agent import (FACTOR_VALUES, build_factor_mask,
                       _IDX_TRIP_COUNT_KNOWN, _IDX_TRIP_COUNT)

    def n_valid(loop) -> int:
        """
        Cells that CAN exist for this loop.  A flat 19 would understate fill on
        trip-count-masked loops — they can never reach 19 — and a fill number
        that cannot reach 100%% would trip the completeness warning forever.
        """
        raw = loop["pre_features_raw"]
        mask = build_factor_mask(raw[_IDX_TRIP_COUNT_KNOWN] > 0.5,
                                 int(raw[_IDX_TRIP_COUNT])).tolist()
        return 2 * sum(mask) - 1        # both unmerge branches, minus the no-op

    def fill(loops):
        if not loops:
            return 0.0
        # len(table) - 1 drops the injected free no-op, which is never measured.
        return sum((len(tables[(l["benchmark_name"], l["loop_idx"])]) - 1)
                   / max(n_valid(l), 1) for l in loops) / len(loops)
    log.info("Cell fill: train %.1f%%, test %.1f%%  (100%% = exact oracle)",
             100 * fill(train_loops), 100 * fill(test_loops))
    if fill(test_loops) < 0.95:
        log.warning("TEST fill is below 95%% — run collect_cells.py --split test "
                    "first, or the oracle is a lower bound and every capture "
                    "ratio below is optimistic.")

    fit_loops = train_loops
    if args.min_fill > 0:
        fit_loops = [l for l in train_loops
                     if (len(tables[(l["benchmark_name"], l["loop_idx"])]) - 1)
                     / max(n_valid(l), 1) >= args.min_fill]
        log.info("FIT restricted to %d/%d train loops with >=%.0f%% fill",
                 len(fit_loops), len(train_loops), 100 * args.min_fill)

    by_bench: dict = {}
    for l in test_loops:
        by_bench.setdefault(l["benchmark_name"], []).append(l)

    rows: list = []

    def emit(ckpt, method, split, k, seed, st):
        rows.append({
            "checkpoint": ckpt, "method": method, "split": split, "k": k,
            "seed": seed, "loops": st["loops"], "scored": st["loops_scored"],
            "unmeasured": st["loops_unmeasured"],
            "benchmarks": st["n_benchmarks"],
            "capture_per_benchmark": round(st["capture_per_benchmark"], 6),
            "capture_pooled": round(st["capture_pooled"], 6),
            "mean_realized": round(st["mean_realized"], 6),
            "regression_rate": round(st["regression_rate"], 6),
        })
        log.info("%-22s %-14s %-5s k=%s seed=%s | capture/bench=%+.3f "
                 "pooled=%+.3f mean_r=%+.4f regress=%.0f%% (%d loops)",
                 ckpt, method, split, k, seed,
                 st["capture_per_benchmark"], st["capture_pooled"],
                 st["mean_realized"], 100 * st["regression_rate"],
                 st["loops_scored"])

    # --- Reference policies (checkpoint-independent) ---
    ranked = marginal_policy(tables, [(l["benchmark_name"], l["loop_idx"])
                                      for l in train_loops])
    rng = random.Random(args.split_seed)
    for name, chooser in (
        ("oracle", lambda t: oracle_of(t)[0]),
        ("always_noop", lambda t: NOOP),
        ("marginal_best", lambda t: next((a for a in ranked if a in t), NOOP)),
        ("random_cell", lambda t: rng.choice(sorted(t.keys()))),
    ):
        picks = [(l["benchmark_name"], l["loop_idx"],
                  chooser(tables[(l["benchmark_name"], l["loop_idx"])]))
                 for l in test_loops]
        emit("-", name, "test", 0, "-", score(picks, tables, args.reward_deadzone))

    # --- Per checkpoint ---
    for ckpt in args.checkpoints:
        tag = ckpt.stem
        agent = fresh_agent(args.agent, ckpt, args.logit_cap)
        emit(tag, "policy", "train", 0, "-",
             evaluate(agent, fit_loops, normalizer, postf, tables,
                      args.reward_deadzone))
        emit(tag, "policy", "test", 0, "-",
             evaluate(agent, test_loops, normalizer, postf, tables,
                      args.reward_deadzone))

        for seed in range(args.seeds):
            srng = random.Random(1000 + seed)
            adapted, scratch, control = [], [], []
            for bench, loops in sorted(by_bench.items()):
                a_loops, e_loops = split_adaptation_loops(loops, srng)
                if not e_loops:
                    continue
                if not a_loops:
                    # Single-loop benchmark: no target data exists, so this is
                    # the zero-shot control group inside the same experiment.
                    control.extend(e_loops)
                    continue
                m = adapt(fresh_agent(args.agent, ckpt, args.logit_cap),
                          a_loops, normalizer,
                          postf, tables, args.agent, args.adapt_lr,
                          args.adapt_steps)
                adapted += [(l["benchmark_name"], l["loop_idx"],
                             greedy_pick(m, l, normalizer, postf))
                            for l in e_loops]
                s = adapt(fresh_agent(args.agent, None, args.logit_cap),
                          a_loops, normalizer,
                          postf, tables, args.agent, args.adapt_lr,
                          args.adapt_steps)
                scratch += [(l["benchmark_name"], l["loop_idx"],
                             greedy_pick(s, l, normalizer, postf))
                            for l in e_loops]
            if adapted:
                emit(tag, "adapted", "test", "1-2", seed,
                     score(adapted, tables, args.reward_deadzone))
                emit(tag, "from_scratch", "test", "1-2", seed,
                     score(scratch, tables, args.reward_deadzone))
                # Same evaluation loops, unadapted — the paired comparison.
                eval_only = [(b, li) for b, li, _ in adapted]
                lookup = {(l["benchmark_name"], l["loop_idx"]): l
                          for l in test_loops}
                emit(tag, "zeroshot_paired", "test", 0, seed,
                     score([(b, li, greedy_pick(agent, lookup[(b, li)],
                                                normalizer, postf))
                            for b, li in eval_only],
                           tables, args.reward_deadzone))
            if control:
                emit(tag, "control_1loop", "test", 0, seed,
                     score([(l["benchmark_name"], l["loop_idx"],
                             greedy_pick(agent, l, normalizer, postf))
                            for l in control],
                           tables, args.reward_deadzone))

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        log.info("Wrote %s (%d rows)", args.out, len(rows))


if __name__ == "__main__":
    main()
