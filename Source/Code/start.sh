 x=1;while [ $x -eq 1 ]; do   x=`nvidia-smi | grep python |wc -l` ;    done  ; python landcover_run.py
