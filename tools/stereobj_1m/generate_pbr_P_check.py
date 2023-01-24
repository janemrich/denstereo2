import os
from pathlib import Path
import json

os.environ["PYOPENGL_PLATFORM"] = "egl"
import os.path as osp
import sys
from datetime import timedelta

from plyfile import PlyData
import mmcv
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from multiprocessing import Pool
from numba import jit

cur_dir = osp.abspath(osp.dirname(__file__))
PROJ_ROOT = osp.join(cur_dir, "../../../..")
sys.path.insert(0, PROJ_ROOT)

def get_scenes(dataset_root):
    trainval_scenes = {}
    with open(osp.join(dataset_root, "split", "biolab_trainval_scenes.txt"), "r") as f:
        # read all lines without newline char
        trainval_scenes['biolab'] = f.read().splitlines()
    # extend with mechanics scenes
    with open(osp.join(dataset_root, "split", "mechanics_trainval_scenes.txt"), "r") as f:
        # read all lines without newline char
        trainval_scenes['mechanics'] = (f.read().splitlines())

    return trainval_scenes

id2obj = {
    1: "blade_razor",
    2: "hammer",
    3: "screwdriver",
    4: "needle_nose_pliers",
    5: "side_cutters",
    6: "tape_measure",
    7: "wire_stripper",
    8: "wrench",
    9: "centrifuge_tube",
    10: "microplate",
    11: "pipette_0.5_10",
    12: "pipette_10_100",
    13: "pipette_100_1000",
    14: "sterile_tip_rack_10",
    15: "sterile_tip_rack_200",
    16: "sterile_tip_rack_1000",
    17: "tube_rack_1.5_2_ml",
    18: "tube_rack_50_ml",
}
obj2id = {v: k for k, v in id2obj.items()}

    
class XyzGen(object):
    def __init__(self, base_dir, dataset, scenes):
        self.base_dir = Path(base_dir)
        self.dataset_root = self.base_dir / dataset
        self.scenes = scenes
        cls_indexes = sorted(id2obj.keys())

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="gen denstereo train_pbr xyz")
    parser.add_argument("--bop_path", type=str, default="/opt/spool/jemrich/")
    parser.add_argument("--dataset", type=str, default="stereobj_1m", help="dataset")
    parser.add_argument("--scene", type=str, default="all", help="scene id")
    args = parser.parse_args()

    base_dir = args.bop_path
    dataset_root = Path(base_dir) / args.dataset


    scenes = get_scenes(dataset_root)
    scenes = scenes['biolab'] + scenes['mechanics']

    total_existing_xyz = 0
    total_should_exist = 0

    for scene in scenes:
        existing_xyz = 0
        should_exist = 0
        scene_root = dataset_root / scene

        for label_path in tqdm(scene_root.glob('*_rt_label.json'), leave=False, desc=f"Image in Scene {scene}"):
            
            mask_path = scene_root / (label_path.stem[:6] + '_mask_label.npz')

            if not label_path.exists():
                print(f"Label file does not exist: {label_path}")
                continue
            with open(label_path, 'r') as rt_f:
                rt_data = json.load(rt_f)

            for obj_number, obj in rt_data['class'].items():
                if obj not in obj2id:
                    # print('skip object')
                    continue
                should_exist += 1
                outpath = scene_root / '{}_{:06d}_xyz.npz'.format(label_path.stem[:6], int(obj2id[obj]))
                if osp.exists(outpath):
                    existing_xyz += 1
                    continue
            
        print('scene', scene)
        # print('existing_xyz', existing_xyz)
        # print('should_exist', should_exist)
        print('missing', should_exist - existing_xyz)
        print()

        total_existing_xyz += existing_xyz
        total_should_exist += should_exist
    
    print('total_existing_xyz', total_existing_xyz)
    print('total_should_exist', total_should_exist)
    print('total_missing', total_should_exist - total_existing_xyz)
    print('total_missing %', (total_should_exist - total_existing_xyz) / total_should_exist)