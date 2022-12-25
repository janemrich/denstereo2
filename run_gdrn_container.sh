
NGPUS=$1
RUN=$2
METHOD=$3
DATASET=$4
EVAL=$5


### setup environment on spool
BASE_PATH="/opt/spool/jemrich-denstereo"
DATASET_DICT_CACHE="/opt/cache/jemrich/datadicts"
IMAGE_CACHE="/opt/cache/jemrich/image.tar"

if [ ! -d $BASE_PATH ]; then
    mkdir $BASE_PATH
fi

cd $BASE_PATH
if [ ! -d $BASE_PATH/denstereo-so ]; then
    git clone git@git.igd-r.fraunhofer.de:jemrich/denstereo-so.git
fi

cd denstereo-so
git checkout denstereo
git pull

### create dataset dict cache
if [ ! -d $DATASET_DICT_CACHE ]; then
    mkdir -p $DATASET_DICT_CACHE
fi

### share cached dataset dicts
echo ""
echo "sync dataset dict cache from pc3002"
rsync -aP pc3002:/opt/datasets/jemrich/cache/ $DATASET_DICT_CACHE 

EVALUATE = ""
if [ $EVAL == "True" ]; then
    echo ""
    echo "sync eval model from pc3002"
    rsync -a --update --whole-file pc3002:/opt/datasets/jemrich/output/${METHOD}/${DATASET}/${RUN}/model_final.pth output/${METHOD}/${DATASET}/${RUN}/model_final.pth
fi

source rootless_docker_env.sh

### cache and load docker image
if [ ! -f $IMAGE_CACHE.gz ]; then
    mkdir -p "/opt/cache/jemrich"
fi
echo "sync image from pc3002"
rsync -a --update --whole-file pc3002:~/den-env-image.tar.gz $IMAGE_CACHE.gz

if [ ! -f $IMAGE_CACHE ]; then
    echo "unzip image..."
    gzip -dv $IMAGE_CACHE.gz
fi

echo "load image..."
docker load -i $IMAGE_CACHE

docker run --shm-size=50G --rm --gpus $NGPUS -it -v $BASE_PATH/denstereo-so:/denstereo-so -v /opt/spool/jemrich/:/denstereo-so/datasets -v $DATASET_DICT_CACHE:/denstereo-so/.cache denstereo-env:latest "cd /denstereo-so; bash core/denstereo_modeling/train_gdrn.sh $NGPUS $EVAL"

echo "sync output..."
rsync -aP $BASE_PATH/denstereo-so/output pc3002:/opt/datasets/jemrich/
rsync -aP $DATASET_DICT_CACHE/ pc3002:/opt/datasets/jemrich/cache