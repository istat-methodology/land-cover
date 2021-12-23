#!/bin/bash
# file: loop_training_run.sh
until python landcover_run.py -c landcover_139_nasnet.ini -m 2 2>/dev/null
do
  sleep 0.1
done
#until python landcover_run.py -c landcover_139_inceptionresnet.ini -m 2 2>/dev/null
#do
#  sleep 0.1
#done
#until python landcover_run.py -c landcover_75.ini -m 2 2>/dev/null
#do
#  sleep 0.1
#done
#until python landcover_run.py -c landcover_139.ini -m 2 2>/dev/null
#do
#  sleep 0.1
#done
