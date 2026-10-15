#!/usr/bin/env bash
# Verification for the two UU pass changes of the nested-loop phase:
#   (1) the dominator-tree cycle cut in unmerge's DFS (UnrollAndUnmerge.cpp,
#       dfsAndDuplicate / unmerge), which fixes the non-terminating compile when
#       unmerging above an unmerged loop that had children;
#   (2) the pre-transform identity stamp (-uu-stamp-loops, OPT-IN, default off:
#       it changes codegen through SimplifyCFG's empty-block rule, see section 2)
#       read back by LoopCount as the uuIdx column.
#
# Runs on the CUDA dev box against multimaterial-cuda/compact.cu (-DFUSED), the
# nest idx 7 > 6 > 5 used for the original experiments.  Compile-only except the
# final runtime check.  Every check prints PASS/FAIL; the script exits non-zero
# if any check failed.
#
# Two-step use:
#   bash verify_nest_fix.sh pre     # BEFORE rebuilding clang: writes reference IR
#                                   # ($OUT/old_*.ll) for four single-action compiles
#   <rebuild clang and opt>
#   bash verify_nest_fix.sh         # all checks; section 3 diffs the new IR against
#                                   # the reference files if they exist
#
# Required env:  HECBENCH_SRC, IR2VEC_VOCAB, clang++ (this tree) first on PATH.
# Optional env:  TARGET_ARCH (else nvidia-smi), CUDA_HOME (/usr/local/cuda),
#                OUT (/tmp/uu_verify).
set -u
: "${HECBENCH_SRC:?export HECBENCH_SRC}"
: "${IR2VEC_VOCAB:?export IR2VEC_VOCAB as in the launch script}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
ARCH="${TARGET_ARCH:-sm_$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '. ')}"
OUT="${OUT:-/tmp/uu_verify}"; mkdir -p "$OUT"
B="$HECBENCH_SRC/multimaterial-cuda"; cd "$B" || exit 1
FILE=compact.cu
NEW="$(command -v clang++)"
OPT="$(dirname "$NEW")/opt"

BASE="-DFUSED -I$CUDA_HOME/include -std=c++17 -Wall -O3 --cuda-gpu-arch=$ARCH --cuda-device-only"
LC="-mllvm --enable-loopcount -mllvm --ir2vec-vocab-path=$IR2VEC_VOCAB"
T=nvptx64-nvidia-cuda
UU="-mllvm --enable-uu -mllvm -uu-match-targettriple=$T -mllvm -uu-match-filename=$FILE"
IN=6; OUT_IDX=7
STAMP="-mllvm -uu-stamp-loops=1"     # opt-in; never used in the oracle build or the regression compiles

fails=0
pass() { echo "PASS  $*"; }
fail() { echo "FAIL  $*"; fails=$((fails + 1)); }

# Columns after stripping "LOOPCOUNT::" (1-based): 1 loopIdx, 2 depth, 9 function,
# 10 numPaths, 12 loopSize, 18 containsChildLoops, 22 nBB, 28 isKernelFunction,
# 29 kernelParents, 30 uuIdx; embedding blocks follow from 31.
rows() { grep '^LOOPCOUNT::[0-9]' "$1" | sed 's/^LOOPCOUNT:://'; }
show() { rows "$1" | awk -F';' '{printf "  idx=%-2s depth=%s child=%s numPaths=%-3s loopSize=%-5s nBB=%-3s uu=%s\n",$1,$2,$18,$10,$12,$22,$30}'; }

# compile NAME  CLANG  "<extra flags>"   -> $OUT/NAME.csv (stderr), rc in $rc
compile() {
  local name=$1 cc=$2 extra=$3
  timeout 300 "$cc" $BASE $LC $extra -c $FILE -o /dev/null 2>"$OUT/$name.csv"
  rc=$?
}
# device IR after UU+LoopCount (what LoopCount saw), via -loopcount-emit-ir
emit_ir() {
  local name=$1 cc=$2 extra=$3
  timeout 300 "$cc" $BASE $LC $extra -mllvm -loopcount-emit-ir="$OUT/$name.ll" -c $FILE -o /dev/null 2>"$OUT/$name.emit.csv"
  rc=$?
}
# final device assembly (PTX): metadata-free, so a diff here is a codegen diff
emit_ptx() {
  local name=$1 cc=$2 extra=$3
  timeout 300 "$cc" $BASE $LC $extra -S -o "$OUT/$name.ptx" $FILE 2>/dev/null
  rc=$?
}
# final device IR (end of the whole -O3 pipeline), for the inertness diagnostic
emit_ll() {
  local name=$1 cc=$2 extra=$3
  timeout 300 "$cc" $BASE $LC $extra -S -emit-llvm -o "$OUT/$name.final.ll" $FILE 2>/dev/null
  rc=$?
}
# strip metadata lines/attachments so two .ll files can be compared structurally
strip_md() { sed -E 's/, ![a-zA-Z.]+ ![0-9]+//g; /^![^ ]+ = /d' "$1"; }

echo "== clang: $NEW"
echo "== assertion mode: $("$(dirname "$NEW")/llvm-config" --assertion-mode 2>/dev/null || echo unknown)"

# The four single-action compiles used as the regression reference (section 3):
# the exact post-feature configuration (P1), unmerge x2 on the inner, LLVM unroll
# x3 on the outer, unmerge x2 on the outer.  All ran on the pre-change clang.
REG_SPECS="P1:$IN:1:1 U_in:$IN:1:2 S_out:$OUT_IDX:0:3 U_out:$OUT_IDX:1:2"
if [ "${1:-}" = pre ]; then
  echo "== pre mode: writing reference IR with the CURRENT clang (run this before rebuilding)"
  for spec in $REG_SPECS; do
    IFS=: read -r name idx u f <<<"$spec"
    emit_ir "old_$name" "$NEW" "$UU -mllvm -uu-opt-loop-idx=$idx -mllvm -uu-opt-loop-unmerge=$u -mllvm -uu-opt-loop-unrollfactors=$f"
    echo "   $name rc=$rc -> $OUT/old_$name.ll"
  done
  exit 0
fi

# ---------------------------------------------------------------------------
echo; echo "### 1. base rows carry their own stamp (idx == uuIdx)"
compile base_stamped "$NEW" "$UU -mllvm -uu-opt-loop-idx=9999 $STAMP"      # all loops skipped, stamp only
if [ $rc -ne 0 ]; then fail "stamp-only compile rc=$rc"; else
  bad=$(rows "$OUT/base_stamped.csv" | awk -F';' '$1 != $30 {print $1":"$30}')
  [ -z "$bad" ] && pass "every base row: uuIdx == loopIdx" || fail "rows with uuIdx != loopIdx: $bad"
fi
compile base_plain "$NEW" ""                                          # no --enable-uu: nothing stamped
if [ $rc -ne 0 ]; then fail "plain loopcount compile rc=$rc"; else
  bad=$(rows "$OUT/base_plain.csv" | awk -F';' '$30 != -1 {print $1":"$30}')
  [ -z "$bad" ] && pass "plain --enable-loopcount rows report uuIdx == -1" || fail "unstamped rows with uuIdx != -1: $bad"
fi

# ---------------------------------------------------------------------------
echo; echo "### 2. inertness: the stamp leaves every feature column unchanged (the check that matters); PTX is reported, not judged"
compile base_nostamp "$NEW" "$UU -mllvm -uu-opt-loop-idx=9999"
# every column except uuIdx: structural features AND the embedding blocks must
# be identical, i.e. a stamped re-extraction compile yields the same feature
# vectors as an unstamped one (LoopCount runs right after the stamp, before any
# other pass can react to it).
if cmp -s <(rows "$OUT/base_stamped.csv" | cut -d';' -f1-29,31-) <(rows "$OUT/base_nostamp.csv" | cut -d';' -f1-29,31-); then
  pass "all feature columns (structural + embeddings) identical with stamps on vs off"; else fail "feature columns differ with stamps on vs off"; fi
# The stamp is KNOWN not to be inert for codegen: SimplifyCFG refuses to fold an
# empty block whose branch and a predecessor's terminator both carry !llvm.loop
# (Local.cpp, TryToSimplifyUncondBranchFromEmptyBlock), which stamping every
# terminator triggers throughout a loop body; LoopRotate then sees a different
# CFG (visible as for.body.lr.ph vs for.body.preheader block names).  That is
# why -uu-stamp-loops is opt-in and never used in a measured compile.  The PTX
# comparison below is therefore informational: it reports the diff sizes and
# whether the final IR already differs, but does not fail the run.
report_ptx() {   # label  flags
  local label=$1 flags=$2
  diff "$OUT/ptx_on$label.ptx" "$OUT/ptx_off$label.ptx" >"$OUT/ptx$label.diff"
  emit_ll "ll_on$label"  "$NEW" "$flags $STAMP"
  emit_ll "ll_off$label" "$NEW" "$flags"
  diff <(strip_md "$OUT/ll_on$label.final.ll") <(strip_md "$OUT/ll_off$label.final.ll") >"$OUT/ll$label.diff"
  echo "INFO  stamped vs unstamped${label:+ (B)}: PTX diff $(wc -l <"$OUT/ptx$label.diff") lines, final IR (metadata stripped) diff $(wc -l <"$OUT/ll$label.diff") lines — expected (SimplifyCFG empty-block rule); files in $OUT"
}
F0="$UU -mllvm -uu-opt-loop-idx=9999"
FB="$UU -mllvm -uu-opt-loop-idx=$IN,$OUT_IDX -mllvm -uu-opt-loop-unmerge=1,0 -mllvm -uu-opt-loop-unrollfactors=2,3"
emit_ptx ptx_on  "$NEW" "$F0 $STAMP"
emit_ptx ptx_off "$NEW" "$F0"
if cmp -s "$OUT/ptx_on.ptx" "$OUT/ptx_off.ptx"; then echo "INFO  PTX byte-identical with stamps on vs off (no transform)"; else report_ptx "" "$F0"; fi
emit_ptx ptx_on_B  "$NEW" "$FB $STAMP"
emit_ptx ptx_off_B "$NEW" "$FB"
if cmp -s "$OUT/ptx_on_B.ptx" "$OUT/ptx_off_B.ptx"; then echo "INFO  PTX byte-identical with stamps on vs off (B)"; else report_ptx "_B" "$FB"; fi

# ---------------------------------------------------------------------------
echo; echo "### 3. regression: single-action compiles unchanged by the cycle-cut change (production config vs pre-change reference IR)"
if [ -s "$OUT/old_P1.ll" ]; then
  for spec in $REG_SPECS; do
    IFS=: read -r name idx u f <<<"$spec"
    if [ ! -s "$OUT/old_$name.ll" ]; then fail "$name: reference $OUT/old_$name.ll missing (pre mode did not produce it)"; continue; fi
    emit_ir "new_$name" "$NEW" "$UU -mllvm -uu-opt-loop-idx=$idx -mllvm -uu-opt-loop-unmerge=$u -mllvm -uu-opt-loop-unrollfactors=$f"
    if [ $rc -ne 0 ]; then fail "$name: new compile rc=$rc"; continue; fi
    if cmp -s "$OUT/old_$name.ll" "$OUT/new_$name.ll"; then pass "$name: IR byte-identical to the pre-change reference"
    elif cmp -s <(strip_md "$OUT/old_$name.ll") <(strip_md "$OUT/new_$name.ll"); then pass "$name: IR identical modulo metadata numbering"
    else fail "$name: IR differs from the pre-change reference — diff $OUT/old_$name.ll $OUT/new_$name.ll"; fi
  done
else
  echo "SKIP  no reference IR in $OUT — run 'bash verify_nest_fix.sh pre' with the pre-change clang before rebuilding"
fi

# ---------------------------------------------------------------------------
echo; echo "### 4. the hang: unmerge above an unmerged loop with children now terminates"
compile C1 "$NEW" "$UU -mllvm -uu-opt-loop-idx=$IN,$OUT_IDX -mllvm -uu-opt-loop-unmerge=1,1 -mllvm -uu-opt-loop-unrollfactors=1,1"
case $rc in
  0)   pass "C1 (inner unmerge x1, outer unmerge x1) rc=0";;
  124) fail "C1 still does not terminate (timeout 300 s)";;
  *)   fail "C1 rc=$rc: $(grep -m1 -E 'Assertion|LLVM ERROR|error:' "$OUT/C1.csv")";;
esac
compile C2 "$NEW" "$UU -mllvm -uu-opt-loop-idx=$IN,$OUT_IDX -mllvm -uu-opt-loop-unmerge=1,1 -mllvm -uu-opt-loop-unrollfactors=2,1"
case $rc in
  0)   pass "C2 (inner unmerge x2, outer unmerge x1) rc=0";;
  124) fail "C2 does not terminate (timeout 300 s)";;
  *)   fail "C2 rc=$rc: $(grep -m1 -E 'Assertion|LLVM ERROR|error:' "$OUT/C2.csv")";;
esac
# the deeper nest in the same file: 11 > 10 > 9 > 8 ; unmerge 9 (has child 8) then 10 then 11
compile C3 "$NEW" "$UU -mllvm -uu-opt-loop-idx=9,10,11 -mllvm -uu-opt-loop-unmerge=1,1,1 -mllvm -uu-opt-loop-unrollfactors=1,1,1"
case $rc in
  0)   pass "C3 (three-level unmerge x1 chain 9,10,11) rc=0";;
  124) fail "C3 does not terminate (timeout 300 s)";;
  *)   fail "C3 rc=$rc: $(grep -m1 -E 'Assertion|LLVM ERROR|error:' "$OUT/C3.csv")";;
esac

# ---------------------------------------------------------------------------
echo; echo "### 5. IR verifier on the transformed modules"
if [ ! -x "$OPT" ]; then
  echo "SKIP  no opt binary next to clang++ ($OPT) — build the 'opt' target to enable this check"
else
  for spec in "C1:1,1:1,1" "C2:1,1:2,1" "B:1,0:2,3" "D1:0,1:2,1"; do
    IFS=: read -r name um fa <<<"$spec"
    emit_ir "v_$name" "$NEW" "$UU -mllvm -uu-opt-loop-idx=$IN,$OUT_IDX -mllvm -uu-opt-loop-unmerge=$um -mllvm -uu-opt-loop-unrollfactors=$fa $STAMP"
    if [ $rc -ne 0 ]; then fail "$name: emit-ir compile rc=$rc"; continue; fi
    if "$OPT" -passes=verify -disable-output "$OUT/v_$name.ll" 2>"$OUT/v_$name.verify"; then pass "$name: opt -passes=verify clean"
    else fail "$name: verifier errors — see $OUT/v_$name.verify"; fi
  done
fi

# ---------------------------------------------------------------------------
echo; echo "### 6. persistence: every clone reports its pre-transform index, no -1"
# Inside the nest every row must carry 5, 6 or 7 with the clone counts observed
# before the change.  Every other loop of the file must appear exactly once with
# its PRE-transform index as its stamp — their loopIdx shifts when the nest grows
# (that is the whole reason for the stamp), so the check is on the stamp set,
# not on loopIdx == uuIdx.
OTHERS_EXPECTED="0 1 2 3 4 8 9 10 11 "
check_persist() {   # name  flags  expect_count_of_5  expect_count_of_6  expect_count_of_7
  local name=$1 flags=$2 e5=$3 e6=$4 e7=$5
  compile "$name" "$NEW" "$flags $STAMP"
  if [ $rc -ne 0 ]; then fail "$name: rc=$rc"; return; fi
  local neg c5 c6 c7 others
  neg=$(rows "$OUT/$name.csv"   | awk -F';' '$30 == -1' | wc -l)
  c5=$(rows "$OUT/$name.csv"    | awk -F';' '$30 == 5'  | wc -l)
  c6=$(rows "$OUT/$name.csv"    | awk -F';' '$30 == 6'  | wc -l)
  c7=$(rows "$OUT/$name.csv"    | awk -F';' '$30 == 7'  | wc -l)
  others=$(rows "$OUT/$name.csv" | awk -F';' '$30 != 5 && $30 != 6 && $30 != 7 {print $30}' | sort -n | tr '\n' ' ')
  if [ "$neg" -eq 0 ] && [ "$c5" -eq "$e5" ] && [ "$c6" -eq "$e6" ] && [ "$c7" -eq "$e7" ] && [ "$others" = "$OTHERS_EXPECTED" ]; then
    pass "$name: uuIdx counts 5x$c5 6x$c6 7x$c7, no -1, other loops' stamps = {$OTHERS_EXPECTED}"
  else
    fail "$name: uuIdx counts 5x$c5 6x$c6 7x$c7 (expected $e5/$e6/$e7), -1 rows=$neg, other stamps='$others' (expected '$OTHERS_EXPECTED')"
    show "$OUT/$name.csv"
  fi
}
# counts come from the rows observed before the change (S_in: jx x3; U_in: jx x7; B: 3x(7 jx + 1 inner); D1: 7 jx + 1 inner)
check_persist S_in "$UU -mllvm -uu-opt-loop-idx=$IN -mllvm -uu-opt-loop-unmerge=0 -mllvm -uu-opt-loop-unrollfactors=3"                    3 1 1
check_persist U_in "$UU -mllvm -uu-opt-loop-idx=$IN -mllvm -uu-opt-loop-unmerge=1 -mllvm -uu-opt-loop-unrollfactors=2"                    7 1 1
check_persist Bst  "$UU -mllvm -uu-opt-loop-idx=$IN,$OUT_IDX -mllvm -uu-opt-loop-unmerge=1,0 -mllvm -uu-opt-loop-unrollfactors=2,3"     21 3 1
check_persist D1st "$UU -mllvm -uu-opt-loop-idx=$IN,$OUT_IDX -mllvm -uu-opt-loop-unmerge=0,1 -mllvm -uu-opt-loop-unrollfactors=2,1"      7 1 1

# ---------------------------------------------------------------------------
echo; echo "### 7. runtime oracle on the previously hanging combination (full build of multimat_F)"
rm -f multimat_F
make multimat_F CC="clang++" CFLAGS="-I$CUDA_HOME/include $UU -mllvm -uu-opt-loop-idx=$IN,$OUT_IDX -mllvm -uu-opt-loop-unmerge=1,1 -mllvm -uu-opt-loop-unrollfactors=2,1 -std=c++17 -Wall -O3 --cuda-gpu-arch=$ARCH" LDFLAGS="-L$CUDA_HOME/lib64 -lcudart -lcuda" >"$OUT/C2.build" 2>&1
if [ $? -ne 0 ]; then fail "C2 full build failed — see $OUT/C2.build"; else
  ./multimat_F 2000 2000 0.3 0.05 0.05 >"$OUT/C2.run" 2>&1
  # multimat_F runs exactly one result check (compact representation; the full-matrix
  # check lives in full_matrix_material_centric, which multimat.cu does not call).
  npass=$(grep -c 'All tests passed' "$OUT/C2.run"); nbad=$(grep -c 'not equal' "$OUT/C2.run")
  if [ "$npass" -eq 1 ] && [ "$nbad" -eq 0 ]; then pass "C2 binary: 'All tests passed!', no mismatches"
  else fail "C2 binary: passed=$npass (expected 1) mismatches=$nbad — see $OUT/C2.run"; fi
fi
rm -f multimat_F

echo; echo "=== $fails check(s) failed ==="
exit $(( fails > 0 ))
