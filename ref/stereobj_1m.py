# encoding: utf-8
"""This file includes necessary params, info."""
import os
import mmcv
import os.path as osp

import numpy as np

# ---------------------------------------------------------------- #
# ROOT PATH INFO
# ---------------------------------------------------------------- #
cur_dir = osp.abspath(osp.dirname(__file__))
root_dir = osp.normpath(osp.join(cur_dir, ".."))
# directory storing experiment data (result, model checkpoints, etc).
output_dir = osp.join(root_dir, "output")

data_root = osp.join(root_dir, "datasets")
bop_root = osp.join(data_root, "BOP_DATASETS/")

# ---------------------------------------------------------------- #
# STEREOBJ_1M DATASET
# ---------------------------------------------------------------- #
dataset_root = osp.join(bop_root, "stereobj_1m/")

# train_pbr_dir = osp.join(dataset_root, "train_pbr_left")

# test_dir = osp.join(dataset_root, "test")
train_scenes = []
with open(osp.join(dataset_root, "split", "biolab_train_scenes.txt"), "r") as f:
    # read all lines without newline char
    train_scenes = f.read().splitlines()
# extend with mechanics scenes
with open(osp.join(dataset_root, "split", "mechanics_train_scenes.txt"), "r") as f:
    # read all lines without newline char
    train_scenes.extend(f.read().splitlines())

debug_scenes = ['mechanics_scene_8_08172020_1']

test_scenes = []
with open(osp.join(dataset_root, "split", "biolab_test_scenes.txt"), "r") as f:
    # read all lines without newline char
    test_scenes = f.read().splitlines()
# extend with mechanics scenes
with open(osp.join(dataset_root, "split", "mechanics_test_scenes.txt"), "r") as f:
    # read all lines without newline char
    test_scenes.extend(f.read().splitlines())

val_scenes = []
with open(osp.join(dataset_root, "split", "biolab_val_scenes.txt"), "r") as f:
    # read all lines without newline char
    val_scenes = f.read().splitlines()
# extend with mechanics scenes
with open(osp.join(dataset_root, "split", "mechanics_val_scenes.txt"), "r") as f:
    # read all lines without newline char
    val_scenes.extend(f.read().splitlines())

trainval_scenes = []
with open(osp.join(dataset_root, "split", "biolab_trainval_scenes.txt"), "r") as f:
    # read all lines without newline char
    trainval_scenes = f.read().splitlines()
# extend with mechanics scenes
with open(osp.join(dataset_root, "split", "mechanics_trainval_scenes.txt"), "r") as f:
    # read all lines without newline char
    trainval_scenes.extend(f.read().splitlines())

all_scenes = trainval_scenes + test_scenes

labeled_scenes = trainval_scenes
    
id2scene = {i: scene for i, scene in enumerate(sorted(all_scenes))}
scene2id = {scene: id for id, scene in id2scene.items()}

model_dir = osp.join(dataset_root, "models")
model_eval_dir = model_dir
# fine_model_dir = osp.join(dataset_root, "models_fine")
# model_eval_dir = osp.join(dataset_root, "models_eval")
# model_scaled_simple_dir = osp.join(dataset_root, "models_rescaled")  # m, .obj
vertex_scale = 1 # TODO

# object info
biolab_objects = [
    "centrifuge_tube",
    "microplate",
    "pipette_0.5_10",
    "pipette_10_100",
    "pipette_100_1000",
    "sterile_tip_rack_10",
    "sterile_tip_rack_200",
    "sterile_tip_rack_1000",
    "tube_rack_1.5_2_ml",
    "tube_rack_50_ml",
]

mechanics_objects = [
    "blade_razor",
    "hammer",
    "needle_nose_pliers",
    "screwdriver",
    "side_cutters",
    "tape_measure",
    "wire_stripper",
    "wrench",
]


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


objects = list(id2obj.values())

obj_num = len(id2obj)
obj2id = {_name: _id for _id, _name in id2obj.items()}

# model_paths = [osp.join(model_dir, "obj_{:06d}.ply").format(_id) for _id in id2obj]  # TODO: check this
# texture_paths = [osp.join(model_dir, "obj_{:06d}.png".format(_id)) for _id in id2obj]
# model_colors = [((i + 1) * 10, (i + 1) * 10, (i + 1) * 10) for i in range(obj_num)]  # for renderer

# yapf: disable
diameters = np.array([
    0.1461045767407543,
    0.25473471948537085,
    0.18378781695690202,
    0.16558196048735566,
    0.15653643902028683,
    0.07983673311480846,
    0.20011998350430626,
    0.1531628526543576,
    0.114393357175511,
    0.15148605326695933,
    0.2559583108512058,
    0.22460018049377958,
    0.23522909301607026,
    0.21167499334903236,
    0.21092084041466744,
    0.2527857901655345,
    0.22173089545663227,
    0.2175223953987267,
])
# yapf: enable
# Camera info
# width = 640
# height = 480
# zNear = 0.25
# zFar = 6.0
# center = (height / 2, width / 2)
# default: 0000~0059 and synt
# baseline = [0.05, 0, 0]
# camera_matrix = uw_camera_matrix = np.array([[1066.778, 0.0, 312.9869], [0.0, 1067.487, 241.3109], [0.0, 0.0, 1.0]])
# 0060~0091
# cmu_camera_matrix = np.array([[1077.836, 0.0, 323.7872], [0.0, 1078.189, 279.6921], [0.0, 0.0, 1.0]])

def get_models_info():
    """key is str(obj_id)"""
    models_info_path = osp.join(model_dir, "models_info.json")
    assert osp.exists(models_info_path), models_info_path
    models_info = mmcv.load(models_info_path)  # key is str(obj_id)
    return models_info


def get_fps_points():
    """key is str(obj_id) generated by tools/lm/lmo_1_compute_fps.py."""
    fps_points_path = osp.join(model_dir, "fps_points.pkl")
    assert osp.exists(fps_points_path), fps_points_path
    fps_dict = mmcv.load(fps_points_path)
    return fps_dict
