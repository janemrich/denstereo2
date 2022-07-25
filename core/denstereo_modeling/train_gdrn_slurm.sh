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

BASE_PATH=/opt/spool/jemrich-denstereo

mkdir $BASE_PATH
cd $BASE_PATH
#if [ -d denStereo-SO ] ; then
git clone git@github.com:janemrich/denStereo-SO.git
#fi
cd denStereo-SO
git checkout denstereo
git pull
ln -s /opt/spool/jemrich datasets
rsync -aP pc3002:/opt/datasets/jemrich/cache/ .cache
rsync -aP pc3002:/opt/datasets/jemrich/VOCdevkit datasets

echo train_gdrn.sh: python core/denstereo_modeling/main_gdrn.py --config-file $CFG --num-gpus $NGPU ${@:3}

python core/denstereo_modeling/main_gdrn.py \
    --config-file $CFG --num-gpus $NGPU  ${@:3} #--launcher dataparallel

# sync output
rsync -aP $BASE_PATH/denStereo-SO/output pc3002:/opt/datasets/jemrich/
rsync -aP $BASE_PATH/denStereo-SO/.cache/ pc3002:/opt/datasets/jemrich/cache
