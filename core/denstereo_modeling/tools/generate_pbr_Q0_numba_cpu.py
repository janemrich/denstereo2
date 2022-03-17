# 产生Q0坐标，同时计算occlusion mask
from cgi import test
import sys
import os.path as osp
sys.path.append('../')
import argparse

import numpy as np
from PIL import Image, ImageFile
import os
import matplotlib.image as mp
from plyfile import PlyData
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mmcv
import ref
from tqdm import tqdm
from collections import OrderedDict

from multiprocessing import Pool
from numba import jit

# 需要知道物体的外接矩形信息，在那个models_info.json里面
def read_rec(model_info_path, obj_id):
    id = obj_id
    model_info = mmcv.load(model_info_path)
    diameter = model_info[id]["diameter"]
    x_min, x_size = model_info[id]["min_x"], model_info[id]["size_x"]
    y_min, y_size = model_info[id]["min_y"], model_info[id]["size_y"]
    z_min, z_size = model_info[id]["min_z"], model_info[id]["size_z"]
    return diameter, x_min, x_size, y_min, y_size, z_min, z_size

@jit(nopython=True)
def test_in_box(point, xmin, xmax, ymin, ymax, zmin, zmax, R_t, t):
    # 要先将点坐标变换回去
    point = (R_t @ point) - (R_t @ t)
    if xmin < point[0] < xmax and ymin < point[1] < ymax and zmin < point[2] < zmax:
        return (1, np.array((point[0,0], point[1,0], point[2,0]), dtype=np.float32))
    else:
        return (0, np.zeros((3), dtype=np.float32))

@jit(nopython=True)
def calc_Q0(height, width, mask, RnxTt, RnyTt, RnzTt,
            Q0_x, Q0_y, Q0_z, R, R_t, t, camK_inv,
            n_x, n_y, n_z, occ_mask_x, occ_mask_y, occ_mask_z,
            xmin, xmax, ymin, ymax, zmin, zmax):

    for i in range(height):
        for j in range(width):
            point = np.array([[j], [i], [1]], dtype=np.float32)
            if mask[i][j] < 1:
                continue
            else:
                Q0_x_v = (
                            RnxTt
                            /   (
                                    (R @ n_x).T
                                    @ (camK_inv @ point)
                                )
                         ) * (camK_inv @ point)
                occ_mask_x[i][j], Q_save = test_in_box(Q0_x_v, xmin, xmax, ymin, ymax, zmin, zmax, R_t, t)
                if occ_mask_x[i][j] > 0:
                    Q0_x[:, i, j] = Q_save

                Q0_y_v = (
                            RnyTt
                            /   (
                                    (R @ n_y).T
                                    @ (camK_inv @ point)
                                )
                         ) * (camK_inv @ point)
                occ_mask_y[i][j], Q_save = test_in_box(Q0_y_v, xmin, xmax, ymin, ymax, zmin, zmax, R_t, t)
                if occ_mask_y[i][j] > 0:
                    Q0_y[:, i, j] = Q_save

                Q0_z_v = (
                         RnzTt
                         /   (
                                (R @ n_z).T
                                @ (camK_inv @ point)
                             )
                         ) * (camK_inv @ point)
                occ_mask_z[i][j], Q_save = test_in_box(Q0_z_v, xmin, xmax, ymin, ymax, zmin, zmax, R_t, t)
                if occ_mask_z[i][j] > 0:
                    Q0_z[:, i, j] = Q_save

    return Q0_x, Q0_y, Q0_z


LM_13_OBJECTS = [
    "ape",
    "benchvise",
    "camera",
    "can",
    "cat",
    "driller",
    "duck",
    "eggbox",
    "glue",
    "holepuncher",
    "iron",
    "lamp",
    "phone",
]  # no bowl, cup
LM_OCC_OBJECTS = ["ape", "can", "cat", "driller", "duck", "eggbox", "glue", "holepuncher"]

class Q0_generator():
    def __init__(self, rootdir, modeldir, xyz_crop_dir, scenes):
        self.dataset_root = rootdir
        self.modeldir = modeldir
        # NOTE: careful! Only the selected objects
        self.objs = LM_OCC_OBJECTS
        self.cat_ids = [cat_id for cat_id, obj_name in ref.lm_full.id2obj.items() if obj_name in self.objs]
        # map selected objs to [0, num_objs-1]
        self.cat2label = {v: i for i, v in enumerate(self.cat_ids)}  # id_map
        self.label2cat = {label: cat for cat, label in self.cat2label.items()}
        self.obj2label = OrderedDict((obj, obj_id) for obj_id, obj in enumerate(self.objs))
        self.scenes = [f"{i:06d}" for i in scenes]
        self.xyz_root = xyz_crop_dir

    def run(self, scale=1000):
        for scene in self.scenes:
            scene_id = int(scene)
            scene_root = osp.join(self.dataset_root, scene)

            gt_dict = mmcv.load(osp.join(scene_root, "scene_gt.json"))
            gt_info_dict = mmcv.load(osp.join(scene_root, "scene_gt_info.json"))
            cam_dict = mmcv.load(osp.join(scene_root, "scene_camera.json"))

            Q0_path = osp.join(self.dataset_root, "Q0", scene)
            if not os.path.exists(Q0_path):
                os.makedirs(Q0_path)
            for str_im_id in gt_dict.keys():
                int_im_id = int(str_im_id)
                rgb_path = osp.join(scene_root, "rgb/{:06d}.jpg").format(int_im_id)
                assert osp.exists(rgb_path), rgb_path

                '''
                show image
                rgb = mmcv.imread(rgb_path, "unchanged")
                plt.imshow(rgb)
                plt.show()
                '''

                camK = np.array(cam_dict[str_im_id]["cam_K"], dtype=np.float32).reshape(3, 3)
                camK_inv = np.linalg.inv(camK)
                depth_factor = 1000.0 / cam_dict[str_im_id]["depth_scale"]  # 10000

                for anno_i, anno in enumerate(gt_dict[str_im_id]):
                    print("processing seq:{:06d} obj:{:06d}".format(scene_id, int_im_id))
                    obj_id = anno["obj_id"]
                    if obj_id not in self.cat_ids:
                        continue
                    outpath = os.path.join(Q0_path, "{:06d}_{:06d}-Q0.pkl".format(int_im_id, anno_i))
                    if osp.exists(outpath):
                        continue
                    R = np.array(anno["cam_R_m2c"], dtype="float32").reshape(3, 3)
                    R_t = R.T.astype(np.float32).copy()
                    t = (np.array(anno["cam_t_m2c"], dtype="float32") / 1000.0).reshape(3, 1)

                    # mask_file = osp.join(scene_root, "mask/{:06d}_{:06d}.png".format(int_im_id, anno_i))
                    mask_visib_file = osp.join(scene_root, "mask_visib/{:06d}_{:06d}.png".format(int_im_id, anno_i))
                    # assert osp.exists(mask_file), mask_file
                    assert osp.exists(mask_visib_file), mask_visib_file
                    # load mask visib  TODO: load both mask_visib and mask_full
                    mask = mmcv.imread(mask_visib_file, "unchanged")
                    mask = mask.astype(np.bool).astype(np.float)
                    area = mask.sum()
                    '''
                    show mask
                    plt.imshow(mask)
                    plt.show()
                    '''
                    xyz_path = osp.join(self.xyz_root, f"{scene_id:06d}/{int_im_id:06d}_{anno_i:06d}-xyz.pkl")
                    assert osp.exists(xyz_path), xyz_path
                    xyz = mmcv.load(xyz_path)
                    x1, y1, x2, y2 = xyz["xyxy"]
                    model_info_path = osp.join(self.modeldir, "models_info.json")
                    diameter, xmin, x_size, ymin, y_size, zmin, z_size = read_rec(model_info_path, str(12))
                    xmax = xmin + x_size
                    ymax = ymin + y_size
                    zmax = zmin + z_size
                    xmin = xmin / scale
                    xmax = xmax / scale
                    ymin = ymin / scale
                    ymax = ymax / scale
                    zmin = zmin / scale
                    zmax = zmax / scale
                    # 开始循环
                    height, width = mask.shape
                    #  存储遮挡mask
                    occ_mask_x = np.zeros((height, width))
                    occ_mask_y = np.zeros((height, width))
                    occ_mask_z = np.zeros((height, width))
                    # 存储Q0的坐标
                    Q0_x = np.zeros((3, height, width), dtype=np.float32)
                    Q0_y = np.zeros((3, height, width), dtype=np.float32)
                    Q0_z = np.zeros((3, height, width), dtype=np.float32)
                    n_x = np.array([[1], [0], [0]], dtype=np.float32)  # Q0_yz
                    n_y = np.array([[0], [1], [0]], dtype=np.float32)  # Q0_xz
                    n_z = np.array([[0], [0], [1]], dtype=np.float32)  # Q0_xy
                    # 计算一些必要的量
                    RnxTt = np.matmul(np.matmul(R, n_x).T, t).astype(np.float32)
                    RnyTt = np.matmul(np.matmul(R, n_y).T, t).astype(np.float32)
                    RnzTt = np.matmul(np.matmul(R, n_z).T, t).astype(np.float32)

                    Q0_x, Q0_y, Q0_z = calc_Q0(
                        height,
                        width,
                        mask,
                        RnxTt,
                        RnyTt,
                        RnzTt,
                        Q0_x,
                        Q0_y,
                        Q0_z,
                        R,
                        R_t,
                        t,
                        camK_inv,
                        n_x,
                        n_y,
                        n_z,
                        occ_mask_x,
                        occ_mask_y,
                        occ_mask_z,
                        xmin,
                        xmax,
                        ymin,
                        ymax,
                        zmin,
                        zmax
                    )

                    Q0 = np.concatenate((Q0_x[1:, :, :], Q0_y[0:1, :, :], Q0_y[2:, :, :], Q0_z[:2, :, :]), axis=0)
                    # 维度变一下CHW -  HWC
                    Q0 = Q0.transpose((1, 2, 0))
                    occ_crop = Q0[y1:y2 + 1, x1:x2 + 1, :]
                    xyxy = [x1, y1, x2, y2],
                    #  存储 Q0的坐标
                    np.savez_compressed(outpath, occ_crop=occ_crop, xyxy=xyxy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="gen lm train_pbr xyz")
    parser.add_argument("--dataset", type=str, default="lm", help="dataset")
    parser.add_argument("--split", type=str, default="train_pbr", help="split")
    parser.add_argument("--xyz_name", type=str, default="xyz_crop_lm", help="xyz folder name")
    args = parser.parse_args()

    base_dir = "/opt/spool/jemrich/BOP_DATASETS/"
    # base_dir = "/home/jemrich/datasets/BOP_DATASETS"
    model_dir = osp.join(base_dir, args.dataset, "models")
    root_dir = osp.join(base_dir, args.dataset, args.split)
    xyz_root = osp.join(root_dir, args.xyz_name)

    def gen_Q0(scenes):
        G_Q = Q0_generator(root_dir, model_dir, xyz_root, scenes)
        G_Q.run(scale=1000)

    scenes = np.array(range(50)).reshape((50,1))
    with Pool(50) as p:
        p.map(gen_Q0, scenes)