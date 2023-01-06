#!/bin/sh
command="python3 train_arps.py --task_type associative -task_json ntm/tasks/associative.json -saved_model saved_models/saved_model_associative.pt -batch_size 250 -num_iters 12000 -summarize_freq 120 -lr 1e-3 --device cuda"

for runid in `seq 1 5`
do
    $command -runid $runid
    $command -runid $runid --infer_type 1
    $command -runid $runid --infer_type 1 --sort_flag
done

#file="hello.py"
#inte=2
#python3 $file -testint $inte