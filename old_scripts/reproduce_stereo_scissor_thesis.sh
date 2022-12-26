#!/bin/bash
NGPU=8
CFG_NAME=denstereo_scissor_early
NAME=rr_stereo_scissor_early
EPOCHS=1000
BS=256

#IMS_PER_GPU=${5:-32}
#$((${NGPU} * ${IMS_PER_GPU}))
srun --gpus $NGPU --nodes=1 --cpus-per-gpu=10 --mem-per-cpu=8G bash core/denstereo_modeling/train_gdrn_slurm.sh configs/denstereo/denstereo/$CFG_NAME.py $NGPU --opts OUTPUT_DIR="output/denstereo/denstereo/${CFG_NAME}_$NAME" SOLVER.TOTAL_EPOCHS=$EPOCHS SOLVER.IMS_PER_BATCH=$BS SOLVER.MAX_TO_KEEP=5 SOLVER.CHECKPOINT_PERIOD=40
