#!/usr/bin/env bash
#SBATCH --job-name=train_denstereo
#SBATCH --gpus 1
#SBATCH --cpus-per-gpu 8
#SBATCH --nodelist ampere2

# MODEL.WEIGHTS: resume or pretrained, or test checkpoint
CFG=$1
NGPU=$2
echo gpus $NGPU

#export RANK=4

cd /opt/cache/
if [ ! -d denStereo-SO ] ; then
	git clone git@github.com:janemrich/denStereo-SO.git
	ln -s /opt/spool/jemrich datasets
else
	cd denStereo-SO
	git pull
fi
nvidia-smi

bash core/gdrn_selfocc_modeling/train_gdrn.sh $CFG $NGPU ${@:4}
#SBATCH --mem-per-cpu 2G
