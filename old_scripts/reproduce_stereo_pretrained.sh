#!/bin/bash
NGPU=$1
CFG_NAME=$2
NAME=$3
EPOCHS=$4
WEIGHTS=$5

IMS_PER_GPU=32
#$((${NGPU} * ${IMS_PER_GPU}))

srun --gpus $NGPU --nodes=1 --cpus-per-gpu=10 --mem-per-cpu=8G bash core/denstereo_modeling/train_gdrn_slurm.sh configs/denstereo/denstereo/$CFG_NAME.py $NGPU --resume --pretrained --opts OUTPUT_DIR="output/denstereo/denstereo/${CFG_NAME}_$NAME" SOLVER.TOTAL_EPOCHS=$EPOCHS SOLVER.IMS_PER_BATCH=$((${NGPU} * ${IMS_PER_GPU})) SOLVER.MAX_TO_KEEP=100 SOLVER.CHECKPOINT_PERIOD=5 SOLVER.WEIGHTS=${WEIGHTS}
