
CONFIG=$1
RUN=$2

echo $CONFIG
echo $RUN

bash core/gdrn_selfocc_modeling/train_gdrn_slurm.sh \
    configs/gdrn_selfocc/denstereo/${CONFIG} \
    1 \
    --eval-only \
    --opts \
    OUTPUT_DIR="output/gdrn_selfocc/denstereo/${WEIGHTS}_eval" \
    MODEL.WEIGHTS="output/gdrn_selfocc/denstereo/${WEIGHTS}/model_final.pth"
