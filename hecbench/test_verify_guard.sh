#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# test_verify_guard.sh
#
# Compile and run all *-cuda benchmarks that have reference.h to verify
# that the VERIFY guard patches are correct.
#
# Two compilation modes are tested per benchmark:
#
#   1. Default (no -DVERIFY): CPU reference is skipped.
#      → Binary must compile and run without crashing.
#
#   2. VERIFY mode (-DVERIFY): CPU reference runs.
#      → Binary must compile and run, typically printing PASS.
#      → Only tested when --with-verify is passed.
#
# No UU flags are set — this tests the guard patch only.
#
# Usage:
#   bash test_verify_guard.sh [OPTIONS]
#
# Options:
#   --hecbench-src DIR    Path to HeCBench/src  (default: HeCBench/src)
#   --arch ARCH           GPU arch, e.g. sm_80   (default: auto-detect)
#   --bench NAME          Test only this benchmark (e.g. entropy-cuda)
#   --with-verify         Also compile+run with -DVERIFY
#   --compile-only        Skip execution, only check compilation
#   --compile-timeout N   Seconds before killing a make (default: 120)
#   --run-timeout N       Seconds before killing a run  (default: 60)
#   --jobs N              Parallel make jobs             (default: 4)
#   -h / --help           Show this help
#
# Output:
#   A per-benchmark result table is printed.
#   A summary CSV is written to verify_guard_results.csv in the current dir.
# ---------------------------------------------------------------------------

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
HECBENCH_SRC="HeCBench/src"
ARCH=""
FILTER_BENCH=""
WITH_VERIFY=0
COMPILE_ONLY=0
COMPILE_TIMEOUT=120
RUN_TIMEOUT=60
JOBS=4

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --hecbench-src) HECBENCH_SRC="$2"; shift 2 ;;
        --arch)         ARCH="$2";         shift 2 ;;
        --bench)        FILTER_BENCH="$2"; shift 2 ;;
        --with-verify)  WITH_VERIFY=1;     shift   ;;
        --compile-only) COMPILE_ONLY=1;    shift   ;;
        --compile-timeout) COMPILE_TIMEOUT="$2"; shift 2 ;;
        --run-timeout)  RUN_TIMEOUT="$2";  shift 2 ;;
        --jobs)         JOBS="$2";         shift 2 ;;
        -h|--help)
            sed -n '/^# Usage:/,/^# -----------/p' "$0" | head -n -1 | sed 's/^# \{0,3\}//'
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Detect GPU arch if not supplied
# ---------------------------------------------------------------------------
if [[ -z "$ARCH" ]]; then
    if command -v nvidia-smi &>/dev/null; then
        CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
        if [[ -n "$CAP" ]]; then
            ARCH="sm_${CAP}"
            echo "Auto-detected arch: ${ARCH}"
        fi
    fi
    if [[ -z "$ARCH" ]]; then
        echo "ERROR: Could not auto-detect GPU arch. Pass --arch sm_XX." >&2
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PASS="\033[0;32mPASS\033[0m"
FAIL="\033[0;31mFAIL\033[0m"
SKIP="\033[0;33mSKIP\033[0m"

CSV_OUT="verify_guard_results.csv"
echo "benchmark,compile_default,run_default,compile_verify,run_verify" > "$CSV_OUT"

n_pass=0; n_fail=0; n_skip=0

try_compile() {
    local bench_dir="$1"
    local extra_cflags="$2"
    local log_file="$3"

    # Clean first to ensure fresh build with new flags
    make -C "$bench_dir" clean -j"$JOBS" &>/dev/null || true

    timeout "$COMPILE_TIMEOUT" \
        make -C "$bench_dir" \
             EXTRA_CFLAGS="$extra_cflags" \
             ARCH="$ARCH" \
             -j"$JOBS" \
             &> "$log_file"
    return $?
}

find_binary() {
    local bench_dir="$1"
    # HeCBench binaries are typically named after the benchmark (without -cuda suffix)
    # or are just 'a.out'.  Try common patterns.
    local bench_name
    bench_name=$(basename "$bench_dir" | sed 's/-cuda$//')
    for candidate in \
        "$bench_dir/$bench_name" \
        "$bench_dir/$(basename "$bench_dir")" \
        "$bench_dir/a.out" \
        $(find "$bench_dir" -maxdepth 1 -type f -executable 2>/dev/null | head -1)
    do
        if [[ -f "$candidate" && -x "$candidate" ]]; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

try_run() {
    local bench_dir="$1"
    local log_file="$2"
    local binary

    if ! binary=$(find_binary "$bench_dir"); then
        echo "  [no binary found]" >> "$log_file"
        return 2   # special: compiled but binary not found
    fi

    # Run from benchmark directory so relative paths (data files etc.) work
    timeout "$RUN_TIMEOUT" \
        bash -c "cd '$bench_dir' && '$(realpath "$binary")'" \
        &> "$log_file"
    return $?
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
if [[ -n "$FILTER_BENCH" ]]; then
    bench_dirs=("$HECBENCH_SRC/$FILTER_BENCH")
else
    mapfile -t bench_dirs < <(find "$HECBENCH_SRC" -maxdepth 1 -name "*-cuda" -type d | sort)
fi

printf "\n%-40s %-14s %-14s %-14s %-14s\n" \
    "BENCHMARK" "COMPILE(def)" "RUN(def)" "COMPILE(VFY)" "RUN(VFY)"
printf "%s\n" "$(printf '%.0s-' {1..100})"

for bench_dir in "${bench_dirs[@]}"; do
    [[ -d "$bench_dir" ]] || continue
    [[ -f "$bench_dir/reference.h" ]] || continue

    bench=$(basename "$bench_dir")
    tmpdir=$(mktemp -d)
    trap "rm -rf '$tmpdir'" EXIT

    c_def="-"; r_def="-"; c_vfy="-"; r_vfy="-"
    row_pass=1

    # ------------------------------------------------------------------
    # Mode 1: default (no -DVERIFY)
    # ------------------------------------------------------------------
    log_c="$tmpdir/compile_default.log"
    if try_compile "$bench_dir" "" "$log_c"; then
        c_def="OK"
    else
        c_def="FAIL"
        row_pass=0
        # Save compile log excerpt
        echo "  !! compile error (no VERIFY) — last 10 lines:" >> "$log_c"
        tail -10 "$log_c" >&2 || true
    fi

    if [[ "$c_def" == "OK" && "$COMPILE_ONLY" -eq 0 ]]; then
        log_r="$tmpdir/run_default.log"
        exit_code=0
        try_run "$bench_dir" "$log_r" || exit_code=$?
        case $exit_code in
            0)   r_def="OK" ;;
            2)   r_def="NOBIN" ;;
            124) r_def="TIMEOUT"; row_pass=0 ;;
            *)   r_def="FAIL(${exit_code})"; row_pass=0 ;;
        esac
    fi

    # ------------------------------------------------------------------
    # Mode 2: with -DVERIFY (optional)
    # ------------------------------------------------------------------
    if [[ "$WITH_VERIFY" -eq 1 ]]; then
        log_c2="$tmpdir/compile_verify.log"
        if try_compile "$bench_dir" "-DVERIFY" "$log_c2"; then
            c_vfy="OK"
        else
            c_vfy="FAIL"
            row_pass=0
        fi

        if [[ "$c_vfy" == "OK" && "$COMPILE_ONLY" -eq 0 ]]; then
            log_r2="$tmpdir/run_verify.log"
            exit_code=0
            try_run "$bench_dir" "$log_r2" || exit_code=$?
            case $exit_code in
                0)   r_vfy="OK" ;;
                2)   r_vfy="NOBIN" ;;
                124) r_vfy="TIMEOUT" ;;
                *)   r_vfy="FAIL(${exit_code})" ;;
            esac
            # Check if output contains PASS or FAIL
            if [[ -f "$log_r2" ]]; then
                if grep -qi "^PASS" "$log_r2" 2>/dev/null; then
                    r_vfy="OK(PASS)"
                elif grep -qi "^FAIL\|FAIL$" "$log_r2" 2>/dev/null; then
                    r_vfy="${r_vfy}(FAIL)"
                fi
            fi
        fi
    fi

    # ------------------------------------------------------------------
    # Print row
    # ------------------------------------------------------------------
    if [[ $row_pass -eq 1 ]]; then
        status="$PASS"
        n_pass=$((n_pass + 1))
    else
        status="$FAIL"
        n_fail=$((n_fail + 1))
    fi

    printf "%-40s %-14s %-14s %-14s %-14s  %b\n" \
        "$bench" "$c_def" "$r_def" "$c_vfy" "$r_vfy" "$status"

    # CSV row (no colour codes)
    echo "${bench},${c_def},${r_def},${c_vfy},${r_vfy}" >> "$CSV_OUT"

    # Clean up temp dir for this benchmark
    rm -rf "$tmpdir"
    trap - EXIT
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
printf "\n%s\n" "$(printf '%.0s-' {1..100})"
printf "Results: %d passed, %d failed, %d skipped\n" "$n_pass" "$n_fail" "$n_skip"
printf "Full results written to: %s\n\n" "$CSV_OUT"
