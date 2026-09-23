"""Scheme A: pretrained inst2vec node featurization + OOV report (the baseline arm).

inst2vec's 2018 vocabulary (8568 statements) predates opaque pointers (`ptr`) and
modern NVPTX intrinsics, so LLVM-21 GPU IR has a high out-of-vocabulary rate. That
is the POINT of this arm: it quantifies why a static pretrained embedding is
crippled on our IR (the evidence motivating scheme C). Faithful to vanilla
inst2vec — statements are preprocessed and looked up as-is; misses fall to !UNK.

Per node the "feature" is an inst2vec embedding INDEX (into the augmented
embeddings table): an instruction's index is its preprocessed statement's dict
entry (or !UNK); variables use !IDENTIFIER; constants/types use !IMMEDIATE.

The dictionary maps preprocessed-statement -> index and includes the special keys
!UNK / !IDENTIFIER / !IMMEDIATE. Struct inlining (ProGraML's optional IR-based
step) is skipped here; it only reduces OOV slightly and needs the raw .ll.
"""
from __future__ import annotations

import base64
import pickle
from typing import Dict, List, Tuple

from .inst2vec import inst2vec_preprocess
from .loader import (CONSTANT, Graph, INSTRUCTION, TYPE, VARIABLE,
                     node_full_text)

# inst2vec treats the module root as having no statement; exclude the sentinel and
# any instruction whose full_text is empty (declaration placeholders) from the OOV
# denominator so the rate reflects REAL statements.
_EMPTY_STATEMENT_TEXTS = frozenset({"", "[external]", "; undefined function"})


class Inst2vecDict:
    """The inst2vec statement->index dictionary (+ special keys)."""

    def __init__(self, d: Dict[str, int]):
        self.d = d
        self.unk = d["!UNK"]
        self.identifier = d["!IDENTIFIER"]
        self.immediate = d["!IMMEDIATE"]

    @classmethod
    def from_pickle(cls, path: str) -> "Inst2vecDict":
        with open(path, "rb") as fh:
            return cls(pickle.load(fh))

    def lookup(self, statement: str) -> Tuple[int, bool]:
        """(embedding index, is_oov). OOV -> !UNK index."""
        hit = statement in self.d
        return (self.d[statement] if hit else self.unk), (not hit)


def _preprocess_statements(full_texts: List[str]) -> List[str]:
    """inst2vec batch preprocessing -> one normalized statement per input."""
    lines = [[t] for t in full_texts]
    pre, _ = inst2vec_preprocess.preprocess(lines)
    return [inst2vec_preprocess.PreprocessStatement(x[0] if x else "") for x in pre]


def encode_scheme_a(g: Graph, dic: Inst2vecDict) -> dict:
    """Assign every node an inst2vec embedding index and report the OOV rate over
    real instruction statements."""
    instr_nodes = [n for n in g.nodes if n.type == INSTRUCTION]
    statements = _preprocess_statements([node_full_text(n) for n in instr_nodes])

    emb_index: Dict[int, int] = {}
    n_real = 0
    oov = 0
    oov_samples: List[str] = []
    for node, stmt in zip(instr_nodes, statements):
        idx, is_oov = dic.lookup(stmt)
        emb_index[node.index] = idx
        # Only real statements count toward OOV (skip root / empty-text placeholders).
        if node.text not in _EMPTY_STATEMENT_TEXTS and node_full_text(node):
            n_real += 1
            if is_oov:
                oov += 1
                if len(oov_samples) < 12:
                    oov_samples.append(stmt)

    for node in g.nodes:
        if node.type == VARIABLE:
            emb_index[node.index] = dic.identifier
        elif node.type in (CONSTANT, TYPE):
            emb_index[node.index] = dic.immediate

    return {
        "embedding_index": emb_index,          # node.index -> inst2vec index
        "n_instructions": len(instr_nodes),
        "n_real_statements": n_real,
        "oov": oov,
        "oov_rate": (oov / n_real) if n_real else 0.0,
        "oov_samples": oov_samples,
    }


# ---------------------------------------------------------------------------
# Self-test (uses a FAKE dictionary + the real vendored preprocessing).
# Run: python -m rl.pgraph.scheme_a
# ---------------------------------------------------------------------------
def _selftest() -> None:
    from .loader import load_graph

    def ft(text: str) -> dict:  # a full_text feature, base64 as in proto3 JSON
        b64 = base64.b64encode(text.encode("utf-8")).decode("ascii")
        return {"feature": {"full_text": {"bytes_list": {"value": [b64]}}}}

    # Two instructions: an "add" that inst2vec knows, and an opaque-pointer "load"
    # that it does NOT (the opaque-pointer OOV case). Plus a var and a const.
    g = load_graph({
        "node": [
            {"text": "add", "function": 0, "features": ft("%3 = add nsw i32 %1, 1")},
            {"text": "load", "function": 0,
             "features": ft("%5 = load i32, ptr %4, align 4")},
            {"type": "VARIABLE", "text": "var", "function": 0},
            {"type": "CONSTANT", "text": "val", "function": 0},
        ],
        "edge": [],
        "function": [{"name": "k"}],
    })
    # Fake dict: knows the preprocessed "add" statement, not the "load".
    fake = Inst2vecDict({
        "!UNK": 0, "!IDENTIFIER": 1, "!IMMEDIATE": 2,
        "<%ID> = add nsw i32 <%ID>, <INT>": 3,
    })
    rep = encode_scheme_a(g, fake)
    assert rep["n_instructions"] == 2, rep
    assert rep["n_real_statements"] == 2, rep
    assert rep["oov"] == 1, rep                       # the opaque-pointer load
    assert abs(rep["oov_rate"] - 0.5) < 1e-9, rep
    ei = rep["embedding_index"]
    assert ei[0] == 3, ei                             # add -> known index
    assert ei[1] == fake.unk, ei                      # load -> !UNK (OOV)
    assert ei[2] == fake.identifier and ei[3] == fake.immediate, ei
    assert any("ptr" in s for s in rep["oov_samples"]), rep["oov_samples"]
    print("pgraph.scheme_a self-test: PASS  (oov_rate=%.2f, sample=%r)"
          % (rep["oov_rate"], rep["oov_samples"][0]))


if __name__ == "__main__":
    _selftest()
