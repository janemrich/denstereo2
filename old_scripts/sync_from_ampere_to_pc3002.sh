#!/bin/sh

BASE_PATH=/opt/spool/jemrich-denstereo

# sync output
rsync -aP $BASE_PATH/denStereo-SO/output pc3002:/opt/datasets/jemrich/
rsync -aP $BASE_PATH/denStereo-SO/.cache/ pc3002:/opt/datasets/jemrich/cache

