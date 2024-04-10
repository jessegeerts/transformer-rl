#/bin/bash

for K in 128 256 512 2048;
do
for B in 0 1 2 4;
do
  python -m reddy_replication_torch.experiment --B ${B} --K ${K} --eps 0.1 --apply_ln 0 --pB 1 --pC 0 --niters 200000
done
done
