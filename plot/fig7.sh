logs_dir=../logs
curr_dir=$(pwd)

aggr() {
  local unrolling=$1
  local config=$2
  echo "${curr_dir}/${logs_dir}/${config}/aggregated_${config}_${unrolling}.csv"
}

heuristic=${curr_dir}/${logs_dir}/heuristic/aggregated_heuristic_1.csv

mkdir -p ${curr_dir}/../results

box_plot=box_plots.py
echo "Creating Figure 7"
python3 $box_plot $(aggr 1 unmerge) $(aggr 2 uu) $(aggr 2 unroll) $(aggr 4 uu) $(aggr 4 unroll) $(aggr 8 uu) $(aggr 8 unroll) $heuristic --y speedup --name "${curr_dir}/../results/fig7" --disable-plot
echo "Results stored in: ${curr_dir}/../results/fig7.pdf"
