#!/bin/sh
#command="python3 train_arps_SAM.py --task_type priority -task_json ntm/tasks/prioritysort_SAM.json -saved_model saved_models/saved_model_priority_sam.pt -batch_size 160 -num_iters 4000 -summarize_freq 40 -lr 1e-3 --device cuda"
command="python3 run_toys.py -task_json=./tasks/prioritysort.json -model_name=stm -mode=train"
for seq_len in 20 30 40
do
    for runid in `seq 1 2`
    do
        $command --seq_len $seq_len -runid $runid
    done
done