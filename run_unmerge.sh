iterations=20
logs_dir=../logs/unmerge/
config=unmerge
build_dir=../build/build-unmerge/

bash run.sh ${iterations} ${logs_dir} ${config} ${build_dir}
bash aggregate_logs.sh ${logs_dir} 1 unmerge
