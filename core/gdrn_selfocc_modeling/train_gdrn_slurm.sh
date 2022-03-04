#!/usr/bin/env bash
# commonly used opts:

# MODEL.WEIGHTS: resume or pretrained, or test checkpoint

# NODE=$1
CFG=$2
NGPU=$3

#srun -w ampere5 --cpus-per-gpu=4 --gpus=4 --mem-per-gpu=64G nvidia-smi
#srun -w ampere5 --cpus-per-gpu=4 --gpus=4 --mem-per-gpu=64G free -m

export RANK=4

# srun -w $NODE --gpus $NGPU  \

if $NGPU > 1
	srun --gpus $NGPU  \
	    --cpus-per-gpu=4 --mem-per-gpu=16G  \
	    python core/gdrn_selfocc_modeling/main_gdrn.py \
	    --config-file $CFG --num-gpus $NGPU --launch none ${@:4}
else
	srun --gpus $NGPU  \
	    --cpus-per-gpu=4 --mem-per-gpu=16G  \
	    python core/gdrn_selfocc_modeling/main_gdrn.py \
	    --config-file $CFG --num-gpus $NGPU ${@:4}
fi
