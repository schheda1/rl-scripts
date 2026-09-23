"""Slice a module ProgramGraph into per-loop and per-kernel subgraphs.

Two granularities (mirroring IR2Vec emb / kemb):

  loop_slice(g, L)      - the loop's region: instructions with L in their
                          loopcount_loops set, their operand/result/type nodes,
                          and one-level boundary producers (loop-invariant
                          live-ins, kept as leaves). Distinct per loop.
  kernel_graph(g, fn)   - a whole function's subgraph (its instruction+variable
                          nodes + adjacent shared const/type nodes). Shared by
                          every loop in that kernel, like kemb.

loop -> kernel context: the loop's OWN function comes from the graph (any seed
node's `function`); whether that function is a kernel, and if not which kernels
call it, come from the LoopCount CSV (isKernelFunction / kernelParents), matched
to graph functions by mangled name (both are emitted from the same stamped IR).

Boundary policy: input side only. A variable consumed in the loop but produced
outside it keeps its producer as a one-level leaf (marked boundary); we do NOT
expand past it, and we do NOT pull external consumers of loop-produced values.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set

from .loader import (Graph, Node, CONSTANT, DATA, INSTRUCTION, TYPE,
                     TYPE_FLOW, VARIABLE)


@dataclass
class Subgraph:
    """A re-indexed induced subgraph. Local index i <-> original orig_index[i]."""
    orig_index: List[int]                       # local -> original node index
    nodes: List[Node]                           # Node objects in local order
    edges: List[tuple]                           # (src_local, tgt_local, flow, position)
    boundary: List[bool]                        # local -> external boundary node?
    target_loop: Optional[int] = None           # set for loop slices
    function: Optional[int] = None              # set for kernel graphs

    @property
    def num_nodes(self) -> int:
        return len(self.orig_index)

    @property
    def num_edges(self) -> int:
        return len(self.edges)


def _induce(g: Graph, included: Set[int], boundary: Set[int],
            target_loop: Optional[int] = None,
            function: Optional[int] = None) -> Subgraph:
    """Build the induced subgraph over `included`, re-indexed to 0..n-1.
    Keeps every edge whose BOTH endpoints are included (cross-boundary edges to
    excluded nodes are cut)."""
    orig = sorted(included)
    local = {o: k for k, o in enumerate(orig)}
    nodes = [g.nodes[o] for o in orig]
    is_boundary = [o in boundary for o in orig]
    edges = []
    for e in g.edges:
        if e.source in local and e.target in local:
            edges.append((local[e.source], local[e.target], e.flow, e.position))
    return Subgraph(orig_index=orig, nodes=nodes, edges=edges,
                    boundary=is_boundary, target_loop=target_loop, function=function)


def loop_seed(g: Graph, loop_id: int) -> Set[int]:
    """Instruction node indices whose loopcount_loops set contains loop_id.
    (For an outer loop this includes its nested subloops' instructions, since
    those carry the outer id too — matching emb/femb's L.blocks() scope.)"""
    return {n.index for n in g.nodes
            if n.type == INSTRUCTION and loop_id in n.loops}


def loop_function_index(g: Graph, loop_id: int) -> Optional[int]:
    """The function index the loop lives in (all its instructions share it)."""
    for n in g.nodes:
        if n.type == INSTRUCTION and loop_id in n.loops:
            return n.function
    return None


def loop_slice(g: Graph, loop_id: int) -> Subgraph:
    seed = loop_seed(g, loop_id)
    included: Set[int] = set(seed)

    # 1. operands (in-DATA: var/const -> instr) and results (out-DATA: instr -> var).
    data_nodes: Set[int] = set()
    for i in seed:
        for e in g.in_edges(i, DATA):
            data_nodes.add(e.source)
        for e in g.out_edges(i, DATA):
            data_nodes.add(e.target)
    included |= data_nodes

    # 2. the type node of each included var/const (in-TYPE: type -> node).
    for v in data_nodes:
        t = g.type_node_of(v)
        if t is not None:
            included.add(t)

    # 3. one-level boundary: producers of loop-consumed variables that live
    #    OUTSIDE the loop (loop-invariant live-ins). Kept as leaves.
    boundary: Set[int] = set()
    for v in data_nodes:
        if g.nodes[v].type != VARIABLE:
            continue
        for p in g.producers_of(v):
            if p not in seed:
                boundary.add(p)
    included |= boundary

    return _induce(g, included, boundary, target_loop=loop_id)


def kernel_graph(g: Graph, function_index: int) -> Subgraph:
    """The whole function's subgraph: its instruction+variable nodes (which carry
    the function index) plus the shared const/type nodes they touch (constants and
    types are module-shared and carry no function, so we pull them by adjacency)."""
    included: Set[int] = {n.index for n in g.nodes if n.function == function_index
                          and n.type in (INSTRUCTION, VARIABLE)}
    extra: Set[int] = set()
    for i in list(included):
        for nb in g.out_neighbors(i) + g.in_neighbors(i):
            if g.nodes[nb].type in (CONSTANT, TYPE):
                extra.add(nb)
    included |= extra
    return _induce(g, included, boundary=set(), function=function_index)


def function_indices_by_name(g: Graph, names: List[str]) -> List[int]:
    """Map mangled function names (e.g. from the CSV kernelParents) to graph
    function indices; silently drops names not present in the graph."""
    idx = {name: i for i, name in enumerate(g.functions)}
    return [idx[n] for n in names if n in idx]


def kernel_context_indices(g: Graph, loop_id: int, is_kernel: bool,
                           parent_names: List[str]) -> List[int]:
    """Function indices whose kernel_graph(s) form loop L's kernel context.
    is_kernel  -> the loop's own function (from the graph).
    otherwise  -> the calling kernels named in the CSV kernelParents.
    (Caller supplies is_kernel / parent_names from the CSV row for loop L.)"""
    if is_kernel:
        fi = loop_function_index(g, loop_id)
        return [fi] if fi is not None else []
    return function_indices_by_name(g, parent_names)


# ---------------------------------------------------------------------------
# Self-test. Run: python -m rl.pgraph.slicer
# ---------------------------------------------------------------------------
def _selftest() -> None:
    from .loader import load_graph
    # Function 0 = kernel "kern". Nodes:
    #   0 instr entry (no loop)          -- produces %inv (loop-invariant)
    #   1 instr in loop 0                -- consumes %inv and %x, produces %y
    #   2 instr in loop 0                -- consumes %y
    #   3 var  %inv  (produced by 0, consumed by 1)   [live-in]
    #   4 var  %x    (argument-like, consumed by 1)
    #   5 var  %y    (produced by 1, consumed by 2)
    #   6 type i32   (type of vars)
    #   7 instr OUTSIDE loop (fn 0) not connected to loop -> excluded
    g = load_graph({
        "node": [
            {"text": "load", "function": 0},                                   # 0
            {"text": "add", "function": 0, "features": {"feature": {
                "loopcount_loops": {"int64_list": {"value": ["0"]}}}}},          # 1
            {"text": "mul", "function": 0, "features": {"feature": {
                "loopcount_loops": {"int64_list": {"value": ["0"]}}}}},          # 2
            {"type": "VARIABLE", "text": "var", "function": 0},                 # 3 %inv
            {"type": "VARIABLE", "text": "var", "function": 0},                 # 4 %x
            {"type": "VARIABLE", "text": "var", "function": 0},                 # 5 %y
            {"type": "TYPE", "text": "i32"},                                    # 6
            {"text": "sub", "function": 0},                                    # 7 (outside)
        ],
        "edge": [
            {"flow": "DATA", "source": 0, "target": 3},   # 0 produces %inv
            {"flow": "DATA", "source": 3, "target": 1},   # %inv -> instr1 (live-in use)
            {"flow": "DATA", "source": 4, "target": 1},   # %x  -> instr1
            {"flow": "DATA", "source": 1, "target": 5},   # instr1 produces %y
            {"flow": "DATA", "source": 5, "target": 2},   # %y  -> instr2
            {"flow": "TYPE", "source": 6, "target": 3},   # i32 -> %inv
            {"flow": "TYPE", "source": 6, "target": 4},   # i32 -> %x
            {"flow": "TYPE", "source": 6, "target": 5},   # i32 -> %y
            {"source": 1, "target": 2},                   # CONTROL instr1 -> instr2
        ],
        "function": [{"name": "kern"}],
    })

    sl = loop_slice(g, 0)
    orig = set(sl.orig_index)
    # in-loop instrs 1,2; their data nodes 3(%inv),4(%x),5(%y); type 6; boundary producer 0.
    assert orig == {0, 1, 2, 3, 4, 5, 6}, "slice node set: %s" % sorted(orig)
    assert 7 not in orig, "unrelated instr excluded"
    # node 0 is the external producer of %inv -> boundary; the in-loop instrs are not.
    bmap = {sl.orig_index[k]: sl.boundary[k] for k in range(sl.num_nodes)}
    assert bmap[0] is True, "external producer marked boundary"
    assert bmap[1] is False and bmap[2] is False, "in-loop instrs not boundary"
    # the control edge 1->2 survives (both in-slice).
    assert any(f == "CONTROL" for (_, _, f, _) in sl.edges), "control edge kept"
    assert sl.target_loop == 0

    kg = kernel_graph(g, 0)
    korig = set(kg.orig_index)
    # all fn-0 instr+var nodes (0,1,2,3,4,5,7) + type node 6 by adjacency; type has no fn.
    assert korig == {0, 1, 2, 3, 4, 5, 6, 7}, "kernel node set: %s" % sorted(korig)

    assert kernel_context_indices(g, 0, is_kernel=True, parent_names=[]) == [0]
    assert kernel_context_indices(g, 0, is_kernel=False,
                                  parent_names=["kern", "absent"]) == [0]
    print("pgraph.slicer self-test: PASS")


if __name__ == "__main__":
    _selftest()
