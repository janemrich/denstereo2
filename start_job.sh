
NGPUS=$1

echo "srun --gpus $NGPUS --nodes=1 --cpus-per-gpu=10 --mem-per-cpu=8G --pty bash run_gdrn_container.sh $NGPUS"
srun --gpus $NGPUS --nodes=1 --cpus-per-gpu=10 --mem-per-cpu=8G --pty bash run_gdrn_container.sh $NGPUS
