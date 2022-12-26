
CONFIG=$1
WEIGHTS=$2

cd ~/denStereo-SO/
srun --gpus 1 --cpus-per-gpu=10 --mem-per-cpu=8G bash core/gdrn_selfocc_modeling/train_gdrn_slurm.sh configs/gdrn_selfocc/denstereo/${CONFIG} 1 --eval-only --opts OUTPUT_DIR="output/gdrn_selfocc/denstereo/${WEIGHTS}_eval" MODEL.WEIGHTS="/igd/a4/homestud/jemrich/models/${WEIGHTS}.pth"
