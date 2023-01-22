#!/bin/sh
command="python3 train_arps.py --task_type priority -task_json ntm/tasks/prioritysort.json -saved_model saved_models/saved_model_priority.pt -batch_size 160 -num_iters 4000 -summarize_freq 40 -lr 1e-3 --device cuda"
#$command -runid 1 --infer_type 1 --sort_flag
for runid in `seq 1 2`
do
    #$command -runid $runid --infer_type 1 --sort_flag
    $command --seq_len 30 -runid $runid
    $command --seq_len 30 -runid $runid --infer_type 1
done

#file="hello.py"
#inte=2
#python3 $file -testint $inte