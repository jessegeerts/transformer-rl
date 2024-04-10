#/bin/bash

for K in 128 256 512 2048;
do
for B in 0 1 2 4;
do
  python reddy_replication_torch.experiment --B ${B} --K ${K} --eps 0.2 --apply_ln 1 --pB 1 --pC 0
done
done
