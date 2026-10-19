"""
Self-contained checks for collect_graphs.py: the pure helpers (dump-name
parsing, device-module selection, gz round trip) and the pgraph package's own
self-tests (loader, slicer, vectorize, emit, collide) — no compiler, no
llvm2graph, no torch, no GPU.

  python3 rl/test/test_collect_graphs.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collect_graphs import (  # noqa: E402
    device_graphs_for, graph_file, is_device_triple, parse_dump_name,
    read_gz_text, write_gz_text,
)

fails = 0


def check(name: str, cond, detail: str = "") -> None:
    global fails
    cond = bool(cond)
    print(("PASS  " if cond else "FAIL  ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        fails += 1


# --- 1. the per-module dump name convention -----------------------------------------
check("dump name: basename with dots, nvptx triple",
      parse_dump_name("main.cu.nvptx64-nvidia-cuda.ll") == ("main.cu", "nvptx64-nvidia-cuda"))
check("dump name: amdgcn triple",
      parse_dump_name("compact.cu.amdgcn-amd-amdhsa.ll") == ("compact.cu", "amdgcn-amd-amdhsa"))
check("dump name: host triple keeps the full basename",
      parse_dump_name("main.cu.x86_64-unknown-linux-gnu.ll") == ("main.cu", "x86_64-unknown-linux-gnu"))
check("dump name: without .ll suffix (as used on .json.gz names)",
      parse_dump_name("main.cu.nvptx64-nvidia-cuda") == ("main.cu", "nvptx64-nvidia-cuda"))
try:
    parse_dump_name("nodots"); check("dump name: malformed rejected", False)
except ValueError:
    check("dump name: malformed rejected", True)
check("device triple: nvptx / amdgcn yes, x86 no",
      is_device_triple("nvptx64-nvidia-cuda") and is_device_triple("amdgcn-amd-amdhsa")
      and not is_device_triple("x86_64-unknown-linux-gnu"))

# --- 2. graph file selection + gz round trip --------------------------------------------
with tempfile.TemporaryDirectory() as d:
    out = Path(d)
    write_gz_text(graph_file(out, "b", "main.cu", "nvptx64-nvidia-cuda"), '{"node": []}')
    write_gz_text(graph_file(out, "b", "main.cu", "x86_64-unknown-linux-gnu"), '{"host": 1}')
    write_gz_text(graph_file(out, "b", "other.cu", "nvptx64-nvidia-cuda"), '{"node": []}')
    sel = device_graphs_for(out, "b", "src/main.cu")
    check("device graph for a file: picks the device module only, by basename",
          [p.name for p in sel] == ["main.cu.nvptx64-nvidia-cuda.json.gz"], str([p.name for p in sel]))
    check("device graph: unknown file -> none", device_graphs_for(out, "b", "nope.cu") == [])
    check("device graph: unknown benchmark -> none", device_graphs_for(out, "zzz", "main.cu") == [])
    check("gz round trip", read_gz_text(sel[0]) == '{"node": []}')
    check("gz write is atomic (no .tmp left)", not any(p.name.endswith(".tmp") for p in (out / "graphs" / "b").iterdir()))

# --- 3. phase 2 + phase 3 glue on a synthetic graph (no compiler, no llvm2graph) --------
import base64  # noqa: E402
import json  # noqa: E402

from collect_graphs import build_corpus_vocab, records_benchmark  # noqa: E402


def _ft(text: str) -> dict:
    b64 = base64.b64encode(text.encode("utf-8")).decode("ascii")
    return {"full_text": {"bytes_list": {"value": [b64]}}}


TINY = {   # the emit self-test's graph: kernel "k", loop 0 = {add, a variable, its i32 type}
    "node": [
        {"text": "[external]"},
        {"text": "add", "function": 0, "features": {"feature": dict(
            _ft("%2 = add nsw i32 %1, 1"),
            **{"loopcount_loops": {"int64_list": {"value": ["0"]}}})}},
        {"type": "VARIABLE", "text": "var", "function": 0},
        {"type": "TYPE", "text": "i32"},
    ],
    "edge": [
        {"flow": "DATA", "source": 2, "target": 1},
        {"flow": "TYPE", "source": 3, "target": 2},
        {"flow": "CALL", "source": 0, "target": 1},
    ],
    "function": [{"name": "k"}],
}
with tempfile.TemporaryDirectory() as d:
    out = Path(d)
    write_gz_text(graph_file(out, "b", "main.cu", "nvptx64-nvidia-cuda"), json.dumps(TINY))
    vsize = build_corpus_vocab(out, ["b"], out / "vocab_C.json")
    check("phase 2: vocab built and saved", vsize > 1 and (out / "vocab_C.json").is_file(), str(vsize))
    name, n, missing = records_benchmark({
        "name": "b", "out": str(out), "scheme": "C", "arch": "sm_80",
        "vocab": str(out / "vocab_C.json"), "inst2vec_dict": None,
        "records": [
            {"loop_idx": 0, "filename": "main.cu", "is_kernel_function": True, "kernel_parents": []},
            {"loop_idx": 7, "filename": "main.cu", "is_kernel_function": True, "kernel_parents": []},
            {"loop_idx": 1, "filename": "other.cu", "is_kernel_function": True, "kernel_parents": []},
        ]})
    check("phase 3: one record for the stamped loop", n == 1, str((n, missing)))
    check("phase 3: unstamped loop and unknown file reported as missing, not fatal",
          sorted(i for i, _ in missing) == [1, 7], str(missing))
    rec_file = out / "records" / "b.json.gz"
    check("phase 3: records file written", rec_file.is_file())
    recs = json.loads(read_gz_text(rec_file)) if rec_file.is_file() else {}
    r0 = recs.get("0", {})
    check("phase 3: record keyed (benchmark, arch, loop_idx) with a loop graph and one kernel graph",
          r0.get("key") == ["b", "sm_80", 0] and r0.get("loop", {}).get("num_nodes", 0) > 0
          and len(r0.get("kernels", [])) == 1, str(list(r0.keys())))

# --- 3b. the empty-dump diagnosis picks the line that explains it ------------------------
from collect_graphs import compile_diagnosis  # noqa: E402

check("diagnosis: LOOPCOUNT WARNING wins over later noise",
      compile_diagnosis("clang: note: x\nLOOPCOUNT WARNING: cannot open -loopcount-emit-ir file '/s/ir/'\nmake: *** [all] Error 1")
      .startswith("LOOPCOUNT WARNING: cannot open"))
check("diagnosis: first error line when there is no warning",
      compile_diagnosis("note: a\nclang: error: unknown argument '-mllvm -foo'\nmake: *** Error 1")
      == "clang: error: unknown argument '-mllvm -foo'")
check("diagnosis: last line as a fallback", compile_diagnosis("a\nb\n\n") == "b")
check("diagnosis: empty stderr", compile_diagnosis("") == "no compiler output")

# --- 4. the pgraph package self-tests (its own, stdlib-only) ----------------------------
import importlib  # noqa: E402

for mod in ("pgraph.loader", "pgraph.slicer", "pgraph.vectorize", "pgraph.emit", "pgraph.collide"):
    try:
        m = importlib.import_module(mod)
        m._selftest()
        check(f"{mod} self-test", True)
    except Exception as e:  # noqa: BLE001
        check(f"{mod} self-test", False, repr(e)[:200])

print(f"\n=== {fails} check(s) failed ===")
sys.exit(1 if fails else 0)
