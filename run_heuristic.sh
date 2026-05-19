iterations=20
logs_dir=../logs/heuristic/
config=uu-heuristic
build_dir=../build/build-heuristic/

bash run.sh ${iterations} ${logs_dir} ${config} ${build_dir}
bash aggregate_logs.sh ${logs_dir} 1 heuristic
bash plot/create_table.sh
