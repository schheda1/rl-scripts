#!/usr/bin/env bash
# Diagnostic ONLY — writes nothing into the repo, changes no state.
#
# Answers, in order:
#   1. what compile command the Makefile actually issues
#   2. whether the DEVICE cc1 job is given -fsave-optimization-record, and
#      where its record is written        <- the whole question
#   3. whether a device-only compile works standalone and emits device records
#   4. whether UU actually fires at this loop index in a device-only compile
#
# Run from anywhere; edit BENCH if adamw-cuda is not convenient.
set -u

BENCH="${BENCH:-adamw-cuda}"
SRCROOT="${SRCROOT:?set SRCROOT=/path/to/HeCBench/src}"
ARCH="${TARGET_ARCH:-sm_80}"
CUDA="${CUDA_HOME:-/usr/local/cuda}"
FILE="main.cu"
LOOP=0; UNMERGE=1; FACTOR=8        # adamw-cuda loop 0, the oracle action

cd "$SRCROOT/$BENCH" || exit 1
echo "### benchmark: $PWD   arch=$ARCH   cuda=$CUDA"
echo

CFLAGS="-I$CUDA/include -std=c++17 -Wall -O3 --cuda-gpu-arch=$ARCH"
LDFLAGS="-L$CUDA/lib64 -lcudart -lcuda"

echo "=== 1. the command make actually runs ============================="
make clean >/dev/null 2>&1
make -n CC=clang++ CFLAGS="$CFLAGS -fsave-optimization-record" LDFLAGS="$LDFLAGS" \
  2>/dev/null | grep -m2 -- "$FILE"
echo

echo "=== 2. cc1 jobs: does the DEVICE job get an opt-record file? ======="
# -### prints the jobs without running them.
clang++ -### -x cuda $CFLAGS -fsave-optimization-record -c "$FILE" -o /tmp/probe.o 2>&1 \
  | grep -- '-cc1' \
  | while IFS= read -r line; do
      t=$(printf '%s' "$line" | grep -o '"-triple" "[^"]*"' | head -1 | cut -d'"' -f4)
      r=$(printf '%s' "$line" | grep -o '"-opt-record-file" "[^"]*"' | head -1 | cut -d'"' -f4)
      s=$(printf '%s' "$line" | grep -c -- '-fsave-optimization-record')
      echo "  triple=${t:-?}"
      echo "     save-optimization-record present: $s"
      echo "     opt-record-file: ${r:-<none>}"
    done
echo

echo "=== 3. device-only compile, explicit record path =================="
rm -f /tmp/dev.opt.yaml
clang++ -x cuda --cuda-device-only -S -O3 --cuda-gpu-arch="$ARCH" -std=c++17 \
    -I. -I"$CUDA/include" \
    -fsave-optimization-record -foptimization-record-file=/tmp/dev.opt.yaml \
    "$FILE" -o /tmp/base.ptx 2>/tmp/dev.err
echo "  exit=$?  ptx_lines=$(wc -l < /tmp/base.ptx 2>/dev/null || echo 0)"
echo "  first compile errors (if any):"; head -3 /tmp/dev.err | sed 's/^/     /'
if [ -s /tmp/dev.opt.yaml ]; then
  echo "  remark records: $(grep -c '^--- !' /tmp/dev.opt.yaml)"
  echo "  functions named in them:"
  grep -h '^Function:' /tmp/dev.opt.yaml | sort | uniq -c | sort -rn | head -8 | sed 's/^/     /'
  echo "  contains host 'main'? -> $(grep -c '^Function: *main$' /tmp/dev.opt.yaml)"
else
  echo "  NO RECORD FILE PRODUCED  <- remark-diff cannot work on the device side"
fi
echo

echo "=== 4. does UU fire here at all (device-only)? ====================="
UUF="-mllvm --enable-uu -mllvm -uu-match-filename=$FILE"
UUF="$UUF -mllvm -uu-match-targettriple=nvptx64-nvidia-cuda"
UUF="$UUF -mllvm -uu-opt-loop-idx=$LOOP -mllvm -uu-opt-loop-unrollfactors=$FACTOR"
UUF="$UUF -mllvm -uu-opt-loop-unmerge=$UNMERGE"
rm -f /tmp/uu.opt.yaml
clang++ -x cuda --cuda-device-only -S -O3 --cuda-gpu-arch="$ARCH" -std=c++17 \
    -I. -I"$CUDA/include" $UUF \
    -fsave-optimization-record -foptimization-record-file=/tmp/uu.opt.yaml \
    "$FILE" -o /tmp/uu.ptx 2>/tmp/uu.err
echo "  exit=$?  ptx_lines=$(wc -l < /tmp/uu.ptx 2>/dev/null || echo 0)"
head -3 /tmp/uu.err | sed 's/^/     /'
if cmp -s /tmp/base.ptx /tmp/uu.ptx; then
  echo "  PTX IDENTICAL  <- UU did NOT fire at this index in device-only mode"
else
  echo "  PTX DIFFERS    <- UU fired; device-only preserves the loop numbering"
  echo "  ptx line delta: $(( $(wc -l < /tmp/uu.ptx) - $(wc -l < /tmp/base.ptx) ))"
fi
if [ -s /tmp/uu.opt.yaml ] && [ -s /tmp/dev.opt.yaml ]; then
  echo "  remark records: baseline=$(grep -c '^--- !' /tmp/dev.opt.yaml) uu=$(grep -c '^--- !' /tmp/uu.opt.yaml)"
fi
