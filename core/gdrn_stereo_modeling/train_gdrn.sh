#!/bin/sh
#!/usr/bin/env bash
set -x

echo $HOSTNAME

# commonly used opts:

# MODEL.WEIGHTS: resume or pretrained, or test checkpoint
CFG=$1
NGPU=$2

NCCL_DEBUG=INFO
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1

export RANK=0
export WORLD_SIZE=1

echo train_gdrn.sh: python core/gdrn_stereo_modeling/main_gdrn.py --config-file $CFG --num-gpus $NGPU ${@:3}

python core/gdrn_stereo_modeling/main_gdrn.py \
    --config-file $CFG --num-gpus $NGPU  ${@:3} #--launcher dataparallel