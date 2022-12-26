#!/bin/bash
NGPU=1
CFG_NAME=denstereo_scissor_early
NAME=r4_stereo_scissor_early
EPOCHS=40
BS=40

#IMS_PER_GPU=${5:-32}
#$((${NGPU} * ${IMS_PER_GPU}))
bash core/denstereo_modeling/train_gdrn.sh configs/denstereo/denstereo/$CFG_NAME.py $NGPU --opts OUTPUT_DIR="output/denstereo/denstereo/${CFG_NAME}_$NAME" SOLVER.TOTAL_EPOCHS=$EPOCHS SOLVER.IMS_PER_BATCH=$BS SOLVER.MAX_TO_KEEP=5 SOLVER.CHECKPOINT_PERIOD=40
