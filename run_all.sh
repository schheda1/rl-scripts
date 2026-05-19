# Run heuristic measurements
bash run_heuristic.sh

# Run unrolling & unmerging measurements
bash run_uu.sh 2
bash run_uu.sh 4
bash run_uu.sh 8

# Run unrolling only measurements
bash run_unroll.sh 2
bash run_unroll.sh 4
bash run_unroll.sh 8

# Run unmerging only measurements
bash run_unmerge.sh
