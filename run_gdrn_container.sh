
NGPUS=$1
CONFIG=$2
RUN_ID=$3
METHOD=$4
DATASET=$5
EVAL=$6
TMUX=$7


### setup environment on spool
DATASET_DICT_CACHE="/opt/cache/jemrich/datadicts"
OUTPUT_PATH="/opt/spool/jemrich-denstereo/output"
IMAGE_CACHE="/opt/cache/jemrich/image.tar"

echo "setup environment on spool"

if [ ! -d $OUTPUT_PATH ]; then
    mkdir $OUTPUT_PATH
fi

### create dataset dict cache
if [ ! -d $DATASET_DICT_CACHE ]; then
    mkdir -p $DATASET_DICT_CACHE
fi

### sync cached dataset dicts
echo ""
echo "sync dataset dict cache from pc3002"
rsync -aP pc3002:/opt/datasets/jemrich/cache/ $DATASET_DICT_CACHE 

EVALUATE=""
if [ ! $EVAL == "False" ]; then
    echo ""
    echo "sync eval model from pc3002"
    rsync -aP --update --whole-file pc3002:/opt/datasets/jemrich/output/${METHOD}/${DATASET}/${RUN_ID}/model_final.pth ${OUTPUT_PATH}/${METHOD}/${DATASET}/${RUN_ID}/model_final.pth
    EVALUATE="--evaluate"
fi

# source rootless_docker_env.sh share-data
source ~/rootless_docker_env_tmux.sh $TMUX

### cache and load docker image
if [ ! -f $IMAGE_CACHE.gz ]; then
    mkdir -p "/opt/cache/jemrich"
fi
echo "sync image from pc3002"
rsync -aP --update --whole-file pc3002:~/den-env-image.tar.gz $IMAGE_CACHE.gz

if [[ $IMAGE_CACHE.gz -nt $IMAGE_CACHE ]]; then
    echo "newer image found on pc3002"
    echo "unzip image..."
    gzip -dv --force $IMAGE_CACHE.gz
fi

echo "load image..."
docker ps
docker images 
docker load -i $IMAGE_CACHE

docker run --shm-size=50G --rm --name prod --gpus $NGPUS -dit -v /denstereo -v $OUTPUT_PATH:/denstereo/output -v ~/denstereo-so/runs:/denstereo/runs -v /opt/spool/jemrich/:/denstereo/datasets -v $DATASET_DICT_CACHE:/denstereo/.cache denstereo-env:latest
docker exec -it prod sh -c "cd /denstereo; git clone git@git.igd-r.fraunhofer.de:jemrich/denstereo-so.git; cd denstereo-so; git checkout denstereo; ln -s ../.cache .cache; ln -s ../runs runs; ln -s ../datasets datasets; rm output/.gitignore; rmdir output; ln -s ../output output; python launch_main.py $CONFIG $RUN_ID $EVALUATE"
docker stop prod

echo "sync output..."
rsync -aP $OUTPUT_PATH/ pc3002:/opt/datasets/jemrich/output
rsync -aP $DATASET_DICT_CACHE/ pc3002:/opt/datasets/jemrich/cache

echo "run_id:"
echo $RUN_ID