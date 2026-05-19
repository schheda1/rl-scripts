logs_dir=../logs
curr_dir=$(pwd)

aggr() {
  local unrolling=$1
  local config=$2
  echo "${curr_dir}/${logs_dir}/${config}/aggregated_${config}_${unrolling}.csv"
}

heuristic=${logs_dir}/heuristic/aggregated_heuristic_1.csv

mkdir -p ${curr_dir}/../results

scatter_plot=read_logs.py

echo "Creating Figure 8a"
python3 $scatter_plot $(aggr 2 uu) $(aggr 2 unroll) $(aggr 4 uu) $(aggr 4 unroll) $(aggr 8 uu) $(aggr 8 unroll) --plot-O3-scatter unroll --plot-O3-scatter-dir "${curr_dir}/../results/" --disable-plot
echo "Results stored in: ${curr_dir}/../results/fig8a.pdf"

echo "Creating Figure 8b"
python3 $scatter_plot $(aggr 2 uu) $(aggr 1 unmerge) $(aggr 4 uu) $(aggr 1 unmerge) $(aggr 8 uu) $(aggr 1 unmerge) --plot-O3-scatter unmerge --plot-O3-scatter-dir "${curr_dir}/../results/" --disable-plot
echo "Results stored in: ${curr_dir}/../results/fig8b.pdf"
