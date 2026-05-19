unroll_factor=$1
iterations=20
logs_dir=../logs/unroll/
config=unroll
build_dir=../build/build-unroll-${unroll_factor}/

bash run.sh ${iterations} ${logs_dir} ${config} ${build_dir} ${unroll_factor}
bash aggregate_logs.sh ${logs_dir} ${unroll_factor} unroll
