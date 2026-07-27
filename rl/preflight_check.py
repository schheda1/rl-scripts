"""
Zero-dependency static checks for the RL pipeline.  Run before every launch:

    python3 rl/preflight_check.py rl/*.py

Exits non-zero if anything is found.  These are not style checks — each one
corresponds to a bug class that reached a GPU run and destroyed it:

  1. USE-BEFORE-ASSIGN.  A name read at the top level of a loop body but only
     assigned inside a conditional branch of that same body.  This is what
     `err_sig` was: bound only on the reward-cache MISS path, read
     unconditionally, so the first cache hit raised UnboundLocalError and
     killed the worker.  23 epochs collected ~0 samples and still "completed".

  2. QUEUE CONTRACT.  Keys read off a message dict (msg["k"]) that no
     result_q.put() in the file ever supplies.  This is what made per-benchmark
     measure_failed counts silently zero: `step_failed` messages carried no
     "benchmark" key, so the consumer's attribution had nothing to attribute.

  3. STATUS SYMMETRY.  Status strings that are READ (st.get("noop")) but never
     WRITTEN by any status= argument or assignment.  `loops_noop` could never be
     non-zero in the parallel path for exactly this reason — the no-op send
     never set status="noop", so every no-op landed in the "ok" bucket.

What this CANNOT check, and what still needs a human reading two files side by
side: whether a CSV carries the columns its consumer needs, whether a JSON
writer preserves keys its reader depends on, whether a reproduction actually
reproduces.  Those were four of the ten bugs.  Run this, then still read.
"""

import ast
import sys
from pathlib import Path


def _terminates(stmt) -> bool:
    """Does this statement transfer control away (so it cannot fall through)?"""
    return isinstance(stmt, (ast.Continue, ast.Break, ast.Return, ast.Raise))


def _stores(node) -> set:
    return {n.id for n in ast.walk(node)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}


def _comprehension_targets(node) -> set:
    """Names bound by comprehensions — scoped to the comprehension, not leaked."""
    out = set()
    for n in ast.walk(node):
        if isinstance(n, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            for gen in n.generators:
                out |= _stores(gen.target)
    return out


def _simple_reads(stmt) -> set:
    """Names LOADed by a statement, excluding the bodies of nested blocks."""
    skip = set()
    for sub in ast.iter_child_nodes(stmt):
        if isinstance(sub, ast.stmt) or isinstance(sub, ast.excepthandler):
            skip.add(id(sub))
    out = set()
    for field, value in ast.iter_fields(stmt):
        items = value if isinstance(value, list) else [value]
        for item in items:
            if not isinstance(item, ast.AST) or id(item) in skip:
                continue
            out |= {n.id for n in ast.walk(item)
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    return out


def _walk_block(stmts, uncond, cond, fn, path, findings):
    """
    Walk a statement list in order, tracking which names are definitely bound
    (`uncond`) vs. only bound on some path (`cond`).  Recurses into nested
    blocks so an assignment and a read inside the SAME `if` are not mistaken
    for a read-before-assign.  Returns (uncond, cond) after the block.
    """
    for stmt in stmts:
        for name in sorted((_simple_reads(stmt) & cond) - uncond):
            findings.append(
                f"{path}:{stmt.lineno}: {fn.name}(): '{name}' is read here but "
                f"only assigned inside a conditional branch above — unbound on "
                f"the path that skips it")

        if isinstance(stmt, ast.If):
            ua, ca = _walk_block(stmt.body, set(uncond), set(cond), fn, path, findings)
            ub, cb = (_walk_block(stmt.orelse, set(uncond), set(cond), fn, path, findings)
                      if stmt.orelse else (set(uncond), set(cond)))
            body_escapes = bool(stmt.body) and _terminates(stmt.body[-1])
            else_escapes = bool(stmt.orelse) and _terminates(stmt.orelse[-1])
            # A branch that transfers control away contributes nothing to the
            # fall-through path — the other branch's bindings are then certain.
            if body_escapes and not else_escapes:
                uncond, cond = ub, cb
            elif else_escapes and not body_escapes:
                uncond, cond = ua, ca
            else:
                uncond = ua & ub
                cond = (ca | cb | ua | ub) - uncond
        elif isinstance(stmt, ast.Try):
            ut, ct = _walk_block(stmt.body, set(uncond), set(cond), fn, path, findings)
            falls_through = []
            for h in stmt.handlers:
                uh, _ch = _walk_block(h.body, set(uncond), set(cond),
                                      fn, path, findings)
                if not (h.body and _terminates(h.body[-1])):
                    falls_through.append(uh)
            if not falls_through:
                # No handlers, or every handler transfers control away: reaching
                # the next statement means the try body ran to completion.
                uncond, cond = ut, ct
            else:
                # Bound only if the body binds it AND every handler that can
                # fall through binds it too — the common try/except idiom
                # `try: x = f() except: x = default` does bind x.
                common = set(ut)
                for uh in falls_through:
                    common &= uh
                cond |= (ut | ct) - common
                uncond = common
            for extra in (stmt.orelse, stmt.finalbody):
                if extra:
                    uncond, cond = _walk_block(extra, uncond, cond, fn, path, findings)
        elif isinstance(stmt, (ast.For, ast.While)):
            # Inside the body the loop target IS bound; after the loop it is
            # only conditionally bound, because the iterable may be empty.
            inner = set(uncond)
            if isinstance(stmt, ast.For):
                inner |= _stores(stmt.target)
            _walk_block(stmt.body, inner, set(cond), fn, path, findings)
            if stmt.orelse:
                _walk_block(stmt.orelse, set(uncond), set(cond), fn, path, findings)
            cond |= _stores(stmt) - uncond
        elif isinstance(stmt, ast.With):
            uncond |= _stores(stmt)
            uncond, cond = _walk_block(stmt.body, uncond, cond, fn, path, findings)
        else:
            uncond |= _stores(stmt)
        cond -= uncond
    return uncond, cond


def check_use_before_assign(tree, path, findings):
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        seed = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
        seed |= _comprehension_targets(fn)
        if fn.args.vararg:
            seed.add(fn.args.vararg.arg)
        if fn.args.kwarg:
            seed.add(fn.args.kwarg.arg)
        # Module-level names need no seeding: `cond` only ever grows from
        # assignments seen inside this function, so a global that is never
        # assigned here can never be flagged.
        _walk_block(fn.body, seed, set(), fn, path, findings)


def _msg_keys_read(node) -> dict:
    """msg["k"] / msg.get("k") reads under `node`, as {key: lineno}."""
    out = {}
    for n in ast.walk(node):
        if (isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
                and n.value.id in ("msg", "m") and isinstance(n.slice, ast.Constant)
                and isinstance(n.slice.value, str)):
            out.setdefault(n.slice.value, n.lineno)
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "get" and isinstance(n.func.value, ast.Name)
                and n.func.value.id in ("msg", "m") and n.args
                and isinstance(n.args[0], ast.Constant)
                and isinstance(n.args[0].value, str)):
            out.setdefault(n.args[0].value, n.lineno)
    return out


def check_queue_contract(tree, path, findings):
    """
    Per-MESSAGE-TYPE key contract.  A file-wide union is not enough: "benchmark"
    was supplied by `entry` and `eval_result` but not by `step_failed`, so the
    consumer's per-benchmark attribution silently counted nothing while a
    file-wide check saw the key as present.
    """
    produced: dict = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("put", "put_nowait") and node.args
                and isinstance(node.args[0], ast.Dict)):
            d = node.args[0]
            keys = {k.value for k in d.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)}
            mtype = next((v.value for k, v in zip(d.keys, d.values)
                          if isinstance(k, ast.Constant) and k.value == "type"
                          and isinstance(v, ast.Constant)), None)
            if mtype is not None:
                produced.setdefault(mtype, set()).update(keys)
    if not produced:
        return

    # Consumers: a branch guarded by `<x> == "<type>"` handles that message type.
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        cmp_ = node.test
        if not (isinstance(cmp_, ast.Compare) and len(cmp_.ops) == 1
                and isinstance(cmp_.ops[0], ast.Eq)
                and isinstance(cmp_.comparators[0], ast.Constant)
                and isinstance(cmp_.comparators[0].value, str)):
            continue
        mtype = cmp_.comparators[0].value
        if mtype not in produced:
            continue
        # Only this branch's body, not the elif chain hanging off orelse.
        for key, line in _msg_keys_read(ast.Module(body=node.body, type_ignores=[])).items():
            if key not in produced[mtype]:
                findings.append(
                    f"{path}:{line}: '{key}' is read from a '{mtype}' message, but "
                    f"no put() of type '{mtype}' supplies it — reads as absent, "
                    f"silently")


def check_status_symmetry(tree, path, findings):
    """Status strings that are read but never written."""
    written, read = set(), {}
    for node in ast.walk(tree):
        # written: status="x" / "status": "x" / status = "x"
        if isinstance(node, ast.keyword) and node.arg == "status" and \
                isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            written.add(node.value.value)
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if (isinstance(k, ast.Constant) and k.value == "status"
                        and isinstance(v, ast.Constant) and isinstance(v.value, str)):
                    written.add(v.value)
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            for t in node.targets:
                if isinstance(t, ast.Name) and "status" in t.id.lower():
                    written.add(node.value.value)
        # written via subscript: st["measure_failed"] = ... — how counters that
        # never travel as a status= value get populated.
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if (isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant)
                        and isinstance(t.slice.value, str)):
                    written.add(t.slice.value)
        # read: st.get("x") / st["x"] on a status-ish mapping
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get" and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in ("st", "status_by_bench", "bmark_status")):
            read.setdefault(node.args[0].value, node.lineno)
    for key, line in sorted(read.items(), key=lambda kv: kv[1]):
        if key not in written and key not in ("noop_count",):
            findings.append(
                f"{path}:{line}: status '{key}' is read but never written by any "
                f"status= argument — this counter can only ever be zero")


def main(argv) -> int:
    paths = [Path(a) for a in argv[1:]]
    if not paths:
        print(__doc__)
        return 2
    findings: list[str] = []
    for p in paths:
        try:
            tree = ast.parse(p.read_text())
        except SyntaxError as e:
            findings.append(f"{p}:{e.lineno}: SYNTAX ERROR: {e.msg}")
            continue
        check_use_before_assign(tree, p, findings)
        check_queue_contract(tree, p, findings)
        check_status_symmetry(tree, p, findings)

    if not findings:
        print(f"preflight: {len(paths)} file(s) clean")
        return 0
    for f in findings:
        print(f"preflight: {f}")
    print(f"\npreflight: {len(findings)} finding(s) — do NOT launch")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
