#!/usr/bin/env bash
srun -w ampere$1 rsync -aP /opt/spool/jemrich-denstereo/denStereo-SO/output pc3002:/opt/datasets/jemrich/
