"""
Box-side check of hecbench._make's HARD timeout: a timed-out compile must take
its whole process tree with it (sh -> make -> the grandchild), and clean and
failing builds must still come back as a CompletedProcess with their exit code
and captured output.  Uses a dummy Makefile that ignores CC/CFLAGS: no
compiler, no benchmark tree, no GPU.  Needs the hecbench import (torch) and
`make`; works for either target (CUDA or HIP).

  python3 rl/test/test_make_timeout.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hecbench import ARCH, _make  # noqa: E402

fails = 0


def check(name: str, cond, detail: str = "") -> None:
    global fails
    cond = bool(cond)
    print(("PASS  " if cond else "FAIL  ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        fails += 1


with tempfile.TemporaryDirectory() as d:
    bdir = Path(d) / "fakebench"          # not *-hip: the HIP sibling helper is a no-op
    bdir.mkdir()
    marker = bdir / "grandchild_ran"

    # --- 1. timeout: the grandchild (sh -> sleep) must die with the tree -------------
    # The default target forks a shell that would touch the marker 3 s in; _make's
    # 1 s timeout must kill sh+make+that shell so the marker never appears.
    (bdir / "Makefile").write_text(
        "all:\n\tsh -c 'sleep 3; touch grandchild_ran'\n"
        "clean:\n\t@true\n")
    t0 = time.time()
    try:
        _make(bdir, extra_cflags="", arch=ARCH, timeout=1)
        check("timeout raises TimeoutExpired", False, "returned normally")
    except subprocess.TimeoutExpired:
        check("timeout raises TimeoutExpired", True)
    dt = time.time() - t0
    check("timeout kills promptly", dt < 5, f"{dt:.1f}s")
    time.sleep(3.5)                       # past when a SURVIVING grandchild would touch it
    check("timeout killed the whole tree (no marker)", not marker.exists())

    # --- 2. clean build: CompletedProcess, rc 0, stdout captured -------------------
    (bdir / "Makefile").write_text("all:\n\t@echo built\nclean:\n\t@true\n")
    r = _make(bdir, extra_cflags="", arch=ARCH, timeout=30)
    check("clean build returns rc 0", r.returncode == 0, str(r.returncode))
    check("clean build captures stdout", "built" in (r.stdout or ""), repr(r.stdout)[:60])

    # --- 3. failing build: rc and stderr come back (the error-signature path) --------
    (bdir / "Makefile").write_text("all:\n\t@echo boom >&2; exit 3\nclean:\n\t@true\n")
    r = _make(bdir, extra_cflags="", arch=ARCH, timeout=30)
    check("failing build returns its rc", r.returncode != 0, str(r.returncode))
    check("failing build captures stderr", "boom" in (r.stderr or ""), repr(r.stderr)[:60])

print(f"\n=== {fails} check(s) failed ===")
sys.exit(1 if fails else 0)
