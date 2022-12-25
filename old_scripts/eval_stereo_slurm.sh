
CONFIG=$1
WEIGHTS=$2

cd ~/denStereo-SO/
srun --gpus 1 --cpus-per-gpu=10 --mem-per-cpu=8G bash core/denstereo_modeling/train_gdrn_slurm.sh configs/denstereo/denstereo/${CONFIG} 1 --eval-only --opts OUTPUT_DIR="output/denstereo/denstereo/${WEIGHTS}_eval" MODEL.WEIGHTS="/igd/a4/homestud/jemrich/models/${WEIGHTS}.pth"
