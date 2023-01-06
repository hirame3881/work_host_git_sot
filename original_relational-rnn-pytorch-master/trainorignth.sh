#!/bin/sh
command="python3 train_nth_farthest.py --cuda"
for runid in `seq 1 3`
do
    $command -runid $runid
done

#file="hello.py"
#inte=2
#python3 $file -testint $inte