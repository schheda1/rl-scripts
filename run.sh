benchmarks=("complex" "coordinates" "haccmk" "lavaMD" "mandelbrot" "rainflow" "libor" "bspline-vgh" "bn" "quicksort" "clink" "contract" "ccs" "qtclustering" "bezier-surface" "xsbench")

iterations=$1
logs_dir=$2
config=$3
build_dir=$4
unroll_factor=$5

mkdir -p ${logs_dir}
mkdir -p ${build_dir}

curr_dir=$(pwd)

for application in "${benchmarks[@]}"; do
    if [ ${config} == "unmerge" ] || [ ${config} == "uu-heuristic" ]; then
        # We do not provide an explicit unroll factor for unmerge or the heuristic
        python3 executors/executor_${application}_cuda.py --num_iterations ${iterations} --logs_dir "${curr_dir}/${logs_dir}" --target_triple nvptx64-nvidia-cuda --filter_branchless --configs ${config} --build-dir ${build_dir}
    else
        python3 executors/executor_${application}_cuda.py --unrolling ${unroll_factor} --num_iterations ${iterations} --logs_dir "${curr_dir}/${logs_dir}" --target_triple nvptx64-nvidia-cuda --filter_branchless --configs ${config} --build-dir ${build_dir}
    fi
done
