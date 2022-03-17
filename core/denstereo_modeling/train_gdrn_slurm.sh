#!/usr/bin/env bash
set -x
# commonly used opts:

# MODEL.WEIGHTS: resume or pretrained, or test checkpoint
CFG=$1
NGPU=$2

#NCCL_DEBUG=INFO
#OMP_NUM_THREADS=1
#MKL_NUM_THREADS=1

cd /opt/cache/
if [ -d denStereo-SO ] ; then
	git clone git@github.com:janemrich/denStereo-SO.git
fi
cd denStereo-SO
git checkout denstereo
git pull
ln -s /opt/spool/jemrich datasets

echo train_gdrn.sh: python core/gdrn_selffocc_modeling/main_gdrn.py --config-file $CFG --num-gpus $NGPU ${@:3}

python core/gdrn_selfocc_modeling/main_gdrn.py \
    --config-file $CFG --num-gpus $NGPU  ${@:3}
