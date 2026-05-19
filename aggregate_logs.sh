path_to_logs=$1
unrolling=$2
config=$3
file="*_${unrolling}_*_${config}_times.txt"

if [[ $config == "heuristic" ]]; then
    file="*_1_*_uu-heuristic_times.txt"
fi

curr_dir=$(pwd)

data_dump_file="${curr_dir}/${path_to_logs}aggregated_${config}_${unrolling}.csv"

# Remove previously aggregated data
rm -f ${data_dump_file}

echo "Aggregating logs for ${config} ${unrolling}"
echo "Storing results in ${data_dump_file}"

for d in "${path_to_logs}"*/; do
    if [[ $d != *"-cuda/" ]]; then
        continue
    fi
    app_name=${d:0:${#d}-6}
    app_name=$(basename $app_name)

    cd ${curr_dir}/$d
    echo "Aggregating logs for $app_name"
    python3 ${curr_dir}/read_logs.py $file --cmp-size $data_dump_file --name $app_name
    cd ..
done

echo "Done aggregating logs"
