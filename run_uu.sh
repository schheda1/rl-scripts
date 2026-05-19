unroll_factor=$1
iterations=20
logs_dir=../logs/uu/
config=uu
build_dir=../build/build-uu-${unroll_factor}/

bash run.sh ${iterations} ${logs_dir} ${config} ${build_dir} ${unroll_factor}

bash aggregate_logs.sh ${logs_dir} ${unroll_factor} uu
