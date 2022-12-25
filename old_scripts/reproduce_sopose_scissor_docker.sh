#!/bin/sh
NGPU=$1
NAME=$2
EPOCHS=$3

IMS_PER_GPU=64
BS=40

bash core/gdrn_selfocc_modeling/train_gdrn_slurm.sh configs/gdrn_selfocc/denstereo/gdrn_selfocc_multistep_ycb-arch_scissors.py $NGPU --opts OUTPUT_DIR="output/gdrn_selfocc/denstereo/reproduce-sopose-$NAME" SOLVER.TOTAL_EPOCHS=$EPOCHS SOLVER.IMS_PER_BATCH=$BS SOLVER.MAX_TO_KEEP=1
