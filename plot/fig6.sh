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
echo "Creating Figure 6a"
python3 $box_plot $(aggr 2 uu) $(aggr 4 uu) $(aggr 8 uu) $heuristic --y speedup --name "${curr_dir}/../results/fig6a" --disable-plot
echo "Results stored in: ${curr_dir}/../results/fig6a.pdf"

echo "Creating Figure 6b"
python3 $box_plot $(aggr 2 uu) $(aggr 4 uu) $(aggr 8 uu) $heuristic --y codesize --name "${curr_dir}/../results/fig6b" --disable-plot
echo "Results stored in: ${curr_dir}/../results/fig6b.pdf"

echo "Creating Figure 6c"
python3 $box_plot $(aggr 2 uu) $(aggr 4 uu) $(aggr 8 uu) $heuristic --y comptime --name "${curr_dir}/../results/fig6c" --disable-plot
echo "Results stored in: ${curr_dir}/../results/fig6c.pdf"
