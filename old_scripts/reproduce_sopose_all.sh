#!/bin/sh
NGPU=$1
CFG=$2
NAME=$3
EPOCHS=$4


IMS_PER_GPU=32
srun --gpus $NGPU --nodes=1 --cpus-per-gpu=10 --mem-per-cpu=8G bash core/gdrn_selfocc_modeling/train_gdrn_slurm.sh configs/gdrn_selfocc/denstereo/${CFG}.py $NGPU --opts OUTPUT_DIR="output/gdrn_selfocc/denstereo/${CFG}-$NAME" SOLVER.TOTAL_EPOCHS=$EPOCHS SOLVER.IMS_PER_BATCH=$(($NGPU * $IMS_PER_GPU)) SOLVER.MAX_TO_KEEP=100 SOLVER.CHECKPOINT_PERIOD=5
