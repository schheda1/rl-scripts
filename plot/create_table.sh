curr_dir=$(pwd)
logs_dir=../logs/heuristic/
ouput_csv="${curr_dir}/../results/table1.txt"
mkdir -p "${curr_dir}/../results/"
touch $ouput_csv

unrolling=1
file="*_${unrolling}_*_uu-heuristic_times.txt"

# write header
echo "application;% C;Baseline Mean;Baseline RSD;Heuristic Mean;Heuristic RSD" > $ouput_csv

for d in "${logs_dir}"*/; do
    if [[ $d == *"-cuda/" ]]; then
        app_name=${d:0:${#d}-6}
        app_name=$(basename $app_name)
        echo "Creating table entry for: $app_name"
        cd ${curr_dir}/$d
        python3 ${curr_dir}/read_logs.py $file --table $ouput_csv --name $app_name
        cd ..
    fi
done

python3 ${curr_dir}/prettify_csv.py $ouput_csv $ouput_csv
cat $ouput_csv
echo "Done creating table"
echo "Table is stored in: $ouput_csv"
