from pathlib import Path
import os.path as osp
import argparse
import subprocess

import numpy as np
from tqdm import tqdm

    
def get_scenes(dataset_root):
    trainval_scenes = {}
    with open(osp.join("biolab_trainval_scenes.txt"), "r") as f:
        # read all lines without newline char
        trainval_scenes['biolab'] = f.read().splitlines()
    # extend with mechanics scenes
    with open(osp.join("mechanics_trainval_scenes.txt"), "r") as f:
        # read all lines without newline char
        trainval_scenes['mechanics'] = (f.read().splitlines())

    return trainval_scenes


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="gen denstereo train_pbr xyz")
    parser.add_argument("--bop_path", type=str, default="/opt/spool/jemrich/")
    parser.add_argument("--dataset", type=str, default="stereobj_1m", help="dataset")
    parser.add_argument("--node", type=str, default="ampere2", help="node name")
    parser.add_argument("--scenes", type=str, default="all", help="scene type")
    parser.add_argument("--threads", type=int, default=1, help="number of threads")
    args = parser.parse_args()

    dataset_path = Path(args.bop_path) / args.dataset
    scenes = get_scenes(dataset_path)

    if args.scenes == "all":
        scenes = scenes['biolab'] + scenes['mechanics']
    else:
        scenes = scenes[args.scenes]

    for scene in tqdm(scenes):

        s = (
            "srun -w {}".format(args.node),
            "--cpus-per-task={}".format(args.threads),
            "python ~/denstereo-so/tools/stereobj_1m/generate_pbr_P.py",
            "--bop_path {}".format(args.bop_path),
            "--dataset {}".format(args.dataset),
            "--scene {}".format(scene),
        )
        subprocess.Popen(" ".join(s), shell=True)