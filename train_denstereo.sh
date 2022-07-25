#!/usr/bin/env bash
# commonly used opts:

# MODEL.WEIGHTS: resume or pretrained, or test checkpoint
#--mem-per-gpu=64G  \
NODE=$1
CFG=$2
NGPU=$3

#export RANK=$NGPU

echo srun -w ${NODE} --nodes=1 --gpus ${NGPU} --cpus-per-gpu=${NGPU} --mem-per-gpu=64G bash core/gdrn_selfocc_modeling/train_gdrn.sh ${CFG} ${NGPU} ${@:4}
srun -w $NODE --gpus $NGPU  --cpus-per-gpu=10 --mem-per-cpu=8G\
    bash core/denstereo_modeling/train_gdrn_slurm.sh \
    $CFG $NGPU ${@:4} |& tee std_out.txt
#    --cpus-per-gpu=4 --mem-per-gpu=32G \
