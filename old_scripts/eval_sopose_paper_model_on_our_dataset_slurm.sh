
#CONFIG=$1
#WEIGHTS=$2

cd ~/denStereo-SO/
srun --gpus 1 --cpus-per-gpu=10 --mem-per-cpu=8G bash core/gdrn_selfocc_modeling/train_gdrn_slurm.sh configs/gdrn_selfocc/ycbv/gdrn_selfocc_multistep_10e.py 1 --eval-only --opts OUTPUT_DIR="output/gdrn_selfocc/denstereo/sopose_eval" MODEL.WEIGHTS="/igd/a4/homestud/jemrich/models/sopose.pth"
