#!/usr/bin/env bash
set -x
# commonly used opts:

# MODEL.WEIGHTS: resume or pretrained, or test checkpoint
CFG=$1
NGPU=$2
#IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
# GPUS=($(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n'))
#NGPU=${#GPUS[@]}  # echo "${GPUS[0]}"
#echo "use gpu ids: $CUDA_VISIBLE_DEVICES num gpus: $NGPU"
# CUDA_LAUNCH_BLOCKING=1
#NCCL_DEBUG=INFO
#OMP_NUM_THREADS=1
#MKL_NUM_THREADS=1
echo train_gdrn.sh: python core/gdrn_selffocc_modeling/main_gdrn.py --config-file $CFG --num-gpus $NGPU ${@:3}

python core/gdrn_selfocc_modeling/main_gdrn.py \
    --config-file $CFG --num-gpus $NGPU  ${@:3}
