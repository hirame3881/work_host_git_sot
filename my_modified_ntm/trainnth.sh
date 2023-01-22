#!/bin/sh
command="python3 train_nth.py -task_json ntm/tasks/nth_farthest.json -saved_model saved_models/saved_model_nth.pt -batch_size 12000 -num_iters 30000 -summarize_freq 100 -lr 1e-4 --device cuda"
for runid in `seq 1 2`
do
#    $command -runid $runid
    $command -runid $runid --infer_type 1
#    $command -runid $runid --infer_type 1 --sort_flag
done

#file="hello.py"
#inte=2
#python3 $file -testint $inte