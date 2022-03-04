#!/usr/bin/env bash
# commonly used opts:

# MODEL.WEIGHTS: resume or pretrained, or test checkpoint
#--mem-per-gpu=64G  \
NODE=$1
CFG=$2
NGPU=$3

export RANK=4

cd /opt/cache/
if [ ! -d denStereo-SO ] ; then
	git clone git@github.com:janemrich/denStereo-SO.git
	ln -s /opt/spool/jemrich datasets
else
	git pull
fi

echo srun -w ${NODE} --gpus ${NGPU} --cpus-per-gpu=${NGPU} --mem-per-gpu=64G bash core/gdrn_selfocc_modeling/train_gdrn.sh ${CFG} ${NGPU} ${@:4}
srun -w $NODE --gpus $NGPU  --cpus-per-gpu=4 --mem-per-cpu=8G\
    bash core/gdrn_selfocc_modeling/train_gdrn.sh \
    $CFG $NGPU ${@:4}
#    --cpus-per-gpu=4 --mem-per-gpu=32G \
