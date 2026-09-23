"""Collision floor for ProGraML loop graphs, to compare against IR2Vec.

The bake-off's input-side question: do distinct loops get distinct representations?
We answer it the same way `analysis/dealias_decomposition.py` does for IR2Vec
vectors, but on the loop GRAPHS: assign each loop graph a canonical signature (a
Weisfeiler-Lehman hash over node labels + directed, flow-typed edges) and count

    #collisions = #loops - #distinct(signatures)

WL is an isomorphism APPROXIMATION: isomorphic graphs always hash equal, and
non-isomorphic graphs almost always differ. The rare failure is a false collision
(two different graphs hash equal), which only ever OVER-counts collisions — so the
reported ProGraML collision count is a conservative (upper-bound) estimate, i.e.
it never overstates ProGraML's advantage over IR2Vec.

The node label is pluggable. The default here is a conservative proxy
(kind|text|boundary); scheme C (vectorize.py) passes a callee-aware label so the
measured floor reflects the actual model features. The comparison to IR2Vec is
done by running this over the SAME eligible-loop set that dealias reports on.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Callable, Dict, List, Optional

from .loader import Node
from .slicer import Subgraph

# A label function maps (node, is_boundary) -> a string token.
LabelFn = Callable[[Node, bool], str]


def default_label(node: Node, is_boundary: bool) -> str:
    """Conservative proxy label: node kind, its text (opcode / type token /
    "var"/"val"), and whether it is a boundary leaf. Calls collapse to "call"
    here (callee identity is folded in by scheme C's label in vectorize.py)."""
    return "%s|%s|%d" % (node.type, node.text, 1 if is_boundary else 0)


def _sha(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def wl_hash(sg: Subgraph, iterations: Optional[int] = None,
            label_fn: Optional[LabelFn] = None) -> str:
    """Weisfeiler-Lehman hash of a subgraph. Directed and flow-aware: a node's
    refinement folds in its outgoing and incoming (flow, neighbor-label) multisets
    separately, so edge direction and flow type both matter. Deterministic
    (neighbor multisets are sorted; final labels sorted), so isomorphic graphs hash
    equal regardless of node numbering.

    Runs to CONVERGENCE by default (iterations=None): refine until the label
    partition stops growing — the finest 1-WL coloring — capped at n rounds (WL
    always stabilizes within n). A fixed round count risks two non-isomorphic loops
    sharing their k-hop label multiset and falsely colliding; convergence minimizes
    that. Pass an int to cap rounds explicitly."""
    label_fn = label_fn or default_label
    n = sg.num_nodes
    if n == 0:
        return _sha("<empty>")
    labels: List[str] = [label_fn(sg.nodes[k], sg.boundary[k]) for k in range(n)]

    out_adj: List[List[tuple]] = [[] for _ in range(n)]
    in_adj: List[List[tuple]] = [[] for _ in range(n)]
    for (s, t, flow, _pos) in sg.edges:
        out_adj[s].append((flow, t))
        in_adj[t].append((flow, s))

    cap = n if iterations is None else iterations
    prev_distinct = len(set(labels))
    for _ in range(cap):
        nxt: List[str] = []
        for k in range(n):
            outs = sorted("o:%s:%s" % (flow, labels[t]) for (flow, t) in out_adj[k])
            ins = sorted("i:%s:%s" % (flow, labels[s]) for (flow, s) in in_adj[k])
            nxt.append(_sha("%s||%s||%s" % (labels[k], ",".join(outs), ",".join(ins))))
        labels = nxt
        distinct = len(set(labels))
        if distinct == prev_distinct:   # partition stabilized (WL fixed point)
            break
        prev_distinct = distinct

    return _sha(",".join(sorted(labels)))


def collision_report(subgraphs: Dict[object, Subgraph],
                     iterations: Optional[int] = None,
                     label_fn: Optional[LabelFn] = None) -> dict:
    """Given {loop_key: loop_subgraph}, return the collision summary and the
    colliding groups (keys that share a signature)."""
    sigs: Dict[object, str] = {k: wl_hash(sg, iterations, label_fn)
                               for k, sg in subgraphs.items()}
    groups: Dict[str, List[object]] = defaultdict(list)
    for k, h in sigs.items():
        groups[h].append(k)
    colliding = {h: sorted(ks, key=str) for h, ks in groups.items() if len(ks) > 1}
    n = len(sigs)
    distinct = len(groups)
    return {
        "n": n,
        "distinct": distinct,
        "collisions": n - distinct,
        "collision_rate": (n - distinct) / n if n else 0.0,
        "colliding_groups": colliding,
    }


# ---------------------------------------------------------------------------
# Self-test. Run: python -m rl.pgraph.collide
# ---------------------------------------------------------------------------
def _selftest() -> None:
    from .loader import load_graph
    from .slicer import loop_slice

    # Two structurally IDENTICAL loops (A, B) + one DIFFERENT loop (C), so a
    # correct collision report finds A==B colliding and C distinct: n=3,
    # distinct=2, collisions=1.
    def two_instr_loop(base: int, loop_id: int, op1: str, op2: str) -> dict:
        # instr b (op1, in loop) -> var (b+2) -> instr b+1 (op2, in loop); type (b+3).
        return {
            "node": [
                {"text": op1, "function": 0, "features": {"feature": {
                    "loopcount_loops": {"int64_list": {"value": [str(loop_id)]}}}}},
                {"text": op2, "function": 0, "features": {"feature": {
                    "loopcount_loops": {"int64_list": {"value": [str(loop_id)]}}}}},
                {"type": "VARIABLE", "text": "var", "function": 0},
                {"type": "TYPE", "text": "i32"},
            ],
            "edge": [
                {"flow": "DATA", "source": base + 0, "target": base + 2},
                {"flow": "DATA", "source": base + 2, "target": base + 1},
                {"flow": "TYPE", "source": base + 3, "target": base + 2},
                {"source": base + 0, "target": base + 1},   # CONTROL
            ],
        }

    # Build one module with 3 loops (ids 0,1,2). Loops 0 and 1 identical (add/mul),
    # loop 2 different (sub/xor).
    nodes: List[dict] = []
    edges: List[dict] = []
    for gi, (lid, o1, o2) in enumerate([(0, "add", "mul"), (1, "add", "mul"),
                                        (2, "sub", "xor")]):
        base = gi * 4
        blk = two_instr_loop(base, lid, o1, o2)
        nodes.extend(blk["node"])
        edges.extend(blk["edge"])
    g = load_graph({"node": nodes, "edge": edges, "function": [{"name": "k"}]})

    subs = {lid: loop_slice(g, lid) for lid in (0, 1, 2)}
    # sanity: identical loops give equal hashes, different one differs.
    h = {lid: wl_hash(subs[lid]) for lid in (0, 1, 2)}
    assert h[0] == h[1], "identical loops must hash equal"
    assert h[0] != h[2], "different loop must hash differently"

    rep = collision_report(subs)
    assert rep["n"] == 3 and rep["distinct"] == 2 and rep["collisions"] == 1, rep
    assert list(rep["colliding_groups"].values())[0] == [0, 1], rep["colliding_groups"]
    print("pgraph.collide self-test: PASS  (%s)"
          % {k: rep[k] for k in ("n", "distinct", "collisions")})


if __name__ == "__main__":
    _selftest()
