#!/usr/bin/env python3
import logging
from loguru import logger as loguru_logger
from pathlib import Path
import os.path as osp
from setproctitle import setproctitle
from detectron2.engine import (
    default_argument_parser,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.config import LazyConfig, instantiate

import wandb

import cv2

cv2.setNumThreads(0)  # pytorch issue 1355: possible deadlock in dataloader
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

import sys

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../../../"))

from lib.utils.time_utils import get_time_str
import core.utils.my_comm as comm
from core.utils.my_checkpoint import MyCheckpointer
from det.yolox.engine.yolox_setup import default_yolox_setup
from det.yolox.engine.yolox_trainer import YOLOX_DefaultTrainer
from det.yolox.utils import fuse_model
from det.yolox.data.datasets.dataset_factory import register_datasets_in_cfg


logger = logging.getLogger("detectron2")


def setup(args):
    """Create configs and perform basic setups."""
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_yolox_setup(cfg, args)
    register_datasets_in_cfg(cfg)
    setproctitle("{}.{}".format(cfg.train.exp_name, get_time_str()))
    return cfg


@loguru_logger.catch
def main(args):
    cfg = setup(args)
    if comm.is_main_process():
        # if args.no_wandb:
            # mode = "disabled"
        # else:
            # mode = "online"
        wandb.init(project="yolox", entity="jemrich") #, mode=mode)
        wandb.config.update(cfg)
        wandb.config.update({ "method": 'yolox',
                              "dataset_0": cfg.DATASETS.TRAIN[0],
                              "seed": cfg.get("SEED", None),
                              })
        '''
        run_name = Path(cfg.OUTPUT_DIR).stem
        wandb.run.name = run_name
        path = Path(cfg.OUTPUT_DIR)
        run = path.parts[3]
        if run[-1] == 's':
            run = '_'.join(run.split('_')[:-1])
        run = run[:-12]
        wandb.config.update({ "method": path.parts[1],
                              "dataset_0": cfg.DATASETS.TRAIN[0],
                              "dataset": path.parts[2],
                              "run": run,
                              "batch_size": cfg.SOLVER.IMS_PER_BATCH,
                              "epochs": cfg.SOLVER.TOTAL_EPOCHS,
                              "weights": cfg.MODEL.WEIGHTS,
                              "seed": cfg.get("SEED", None),
                              })
        '''
    Trainer = YOLOX_DefaultTrainer
    if args.eval_only:  # eval
        model = Trainer.build_model(cfg)
        MyCheckpointer(model, save_dir=cfg.train.output_dir).resume_or_load(
            cfg.train.init_checkpoint, resume=args.resume
        )
        if cfg.test.fuse_conv_bn:
            logger.info("\tFusing conv bn...")
            model = fuse_model(model)
        res = Trainer.test(cfg, model)
        # import ipdb; ipdb.set_trace()
        return res
    # train
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
