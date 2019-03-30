#!/bin/sh

BASE=/home/ubuntu/nucleus_prediction
MODEL_NAME=$1
PROC_DIR=$BASE/src
LOG_DIR=$BASE/logs/$MODEL_NAME

mkdir $BASE/logs
mkdir $BASE/logs/$MODEL_NAME

cd $PROC_DIR && nohup python3.5 run.py > $LOG_DIR/$MODEL_NAME.out 2> $LOG_DIR/$MODEL_NAME.err < /dev/null &