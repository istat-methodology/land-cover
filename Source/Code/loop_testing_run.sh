#!/bin/bash
# file: loop_testing_run.sh
until python landcover_run.py -m 3 2>/dev/null
do
  sleep 0.1
done
#until python landcover_run.py -c landcover_75.ini -m 3 2>/dev/null
#do
#  sleep 0.1
#done
#until python landcover_run.py -c landcover_139.ini -m 3 2>/dev/null
#do
#  sleep 0.1
#done
