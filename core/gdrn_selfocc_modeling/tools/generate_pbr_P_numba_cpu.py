# generate Q0

import sys
import argparse

sys.path.append('../')
import numpy as np
from PIL import Image, ImageFile
import os
import matplotlib.image as mp
from plyfile import PlyData
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ref
from _collections import OrderedDict
import os.path as osp
import mmcv

from multiprocessing import Pool
from numba import jit

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

intrinsic_matrix = {
    'linemod': np.array([[572.4114, 0., 325.2611],
                         [0., 573.57043, 242.04899],
                         [0., 0., 1.]]),
}


@jit(nopython=True)
def transformer(P0, R, t):
    P0 = np.reshape(P0, (3, 1))
    P = (R @ P0) + t
    return P


@jit(nopython=True)
def transformer_back(P, R, t):  # calculateP0=RTP-RTt
    P0 = (R.T @ P) - (R.T @ t)
    return P0


@jit(nopython=True)
def projector(P0, K, R, t):  # Calculate the camera projection, and then projep it to the image after R, T transform
    p = (K  @ P0) / P0[2]
    p = p[0:2, :] / p[2]
    return p


@jit(nopython=True)
def pointintriangle(A, B, C, P):  # Judging whether a point is in the interior of the 3 corner, ABC is a triangular wrapper 3 points    P = np.expand_dims(P, 1)
    P = np.expand_dims(P, 1)
    v0 = C - A
    v1 = B - A
    v2 = P - A

    dot00 = (v0.T @ v0)
    dot01 = (v0.T @ v1)
    dot02 = (v0.T @ v2)
    dot11 = (v1.T @ v1)
    dot12 = (v1.T @ v2)

    down = dot00 * dot11 - dot01 * dot01
    if down < 1e-6:
        return False

    inverdeno = 1 / down

    u = (dot11 * dot02 - dot01 * dot12) * inverdeno
    if u < 0 or u > 1:
        return False
    v = (dot00 * dot12 - dot01 * dot02) * inverdeno
    if v < 0 or v > 1:
        return False
    if u + v <= 1: # necessary for numba
        return True
    else:
        return False

@jit(nopython=True)
def calc_xy_crop(vert_id, vert, camK, R, t, norm_d, height, width, camK_inv, pixellist, mask, xyxy):
    for i in range(vert_id.shape[0]):  # 行数
        P1 = transformer(vert[vert_id[i][0], :].T.copy(), R, t)
        P2 = transformer(vert[vert_id[i][1], :].T.copy(), R, t)
        P3 = transformer(vert[vert_id[i][2], :].T.copy(), R, t)
        p1 = projector(P1, camK, R, t)
        p2 = projector(P2, camK, R, t)  # col first
        p3 = projector(P3, camK, R, t)
        planenormal = norm_d[vert_id[i][0], :]
        planenormal = np.expand_dims(planenormal, 1)
        planenormal = R @ planenormal
        # Calculatep 1, p2, p3 Integer point in the triangle and initialize one for them candidate
        p_x_min = min([p1[0].item(), p2[0].item(), p3[0].item()])
        p_x_max = max([p1[0].item(), p2[0].item(), p3[0].item()])
        p_y_min = min([p1[1].item(), p2[1].item(), p3[1].item()])
        p_y_max = max([p1[1].item(), p2[1].item(), p3[1].item()])
        # inside the image
        if p_y_min < 0.: p_y_min = 0.
        if p_y_max >= height: p_y_max = height - 1.
        if p_x_min < 0.: p_x_min = 0.
        if p_x_max >= width: p_x_max = width - 1.
        for x in np.arange(int(p_x_min), int(p_x_max) + 1, 1):
            for y in np.arange(int(p_y_min), int(p_y_max) + 1, 1): # row
                if pointintriangle(p1, p2, p3, np.asarray([x, y], dtype=np.float32).T):
                    point = np.array([x, y, 1]).astype(np.float32)
                    point = np.expand_dims(point, 1)
                    Zp_upper = planenormal.T @ P1
                    Zp_lower = planenormal.T @ (camK_inv @ point)
                    Zp = np.abs(Zp_upper / Zp_lower)
                    pixellist[y, x] = min([Zp.item(), pixellist[y, x].item()])
    # 生成P0的图， 之前只存储了Zp， 现在计算值
    # pixellist is the result
    P0_output = np.zeros((height, width, 3), dtype=np.float32)
    x1, y1, x2, y2 = xyxy
    for i in range(y1, y2+1):
        for j in range(x1, x2+1):
            if mask[i][j] < 1 or pixellist[i, j] > 30:
                continue
            else:
                point = np.array([j, i, 1], dtype=np.float32)
                point = np.expand_dims(point, 1)
                P = pixellist[i, j] * (camK_inv @ point)
                P = P.astype(np.float32)
                P0 = transformer_back(P, R, t)
                # P0_3 = P0.reshape(3)
                P0_output[i, j, :] = P0.reshape(3)  # 边界上的点在计算的时候会出现错误， 没有完全包裹住

    return P0_output[y1:y2 + 1, x1:x2 + 1, :]


def modelload(model_dir, ids):
    modellist = {}
    for obj in ids:
        model_path = osp.join(model_dir, "obj_{:06d}.ply".format(obj))
        ply = PlyData.read(model_path)
        modellist[str(obj)] = ply
    return modellist


class estimate_coor_P0():
    def __init__(self, rootdir, modeldir, scenes):
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
        self.xyz_root = osp.join(self.dataset_root, "xyz_crop")
        self.model = modelload(modeldir, self.cat_ids)
        self.new_xyz_root = osp.join(self.dataset_root, "xyz_crop_lm")

    def run(self, scale=1000):
        camK = intrinsic_matrix["linemod"].astype(np.float32)
        camK_inv = np.linalg.inv(camK)
        height = 480
        width = 640
        for scene in self.scenes:
            scene_id = int(scene)
            scene_root = osp.join(self.dataset_root, scene)
            gt_dict = mmcv.load(osp.join(scene_root, "scene_gt.json"))
            # gt_info_dict = mmcv.load(osp.join(scene_root, "scene_gt_info.json"))
            # cam_dict = mmcv.load(osp.join(scene_root, "scene_camera.json"))
            basic_out_path = osp.join(self.new_xyz_root, f"{scene_id:06d}")
            if not os.path.exists(basic_out_path):
                os.makedirs(basic_out_path)
            for str_im_id in gt_dict.keys():
                int_im_id = int(str_im_id)
                print("processing seq:{:06d} img:{:06d}".format(scene_id, int_im_id))
                rgb_path = osp.join(scene_root, "rgb/{:06d}.jpg").format(int_im_id)
                assert osp.exists(rgb_path), rgb_path

                '''
                show image
                rgb = mmcv.imread(rgb_path, "unchanged")
                plt.imshow(rgb)
                plt.show()
                '''

                for anno_i, anno in enumerate(gt_dict[str_im_id]):
                    obj_id = anno["obj_id"]
                    if obj_id in self.cat_ids:
                        outpath = osp.join(self.new_xyz_root, f"{scene_id:06d}/{int_im_id:06d}_{anno_i:06d}-xyz.pkl")
                        if osp.exists(outpath):
                            continue
                        R = np.array(anno["cam_R_m2c"], dtype="float32").reshape(3, 3)
                        t = (np.array(anno["cam_t_m2c"], dtype="float32") / 1000.0).reshape(3, 1)
                        # mask_file = osp.join(scene_root, "mask/{:06d}_{:06d}.png".format(int_im_id, anno_i))
                        mask_visib_file = osp.join(scene_root, "mask/{:06d}_{:06d}.png".format(int_im_id, anno_i))
                        # assert osp.exists(mask_file), mask_file
                        assert osp.exists(mask_visib_file), mask_visib_file
                        # load mask visib  TODO: load both mask_visib and mask_full
                        mask = mmcv.imread(mask_visib_file, "unchanged")
                        mask = mask.astype(bool).astype(float)
                        if mask.sum() == 0:
                            P = {
                                "xyz_crop": np.zeros((height, width, 3), dtype=np.float16),
                                "xyxy": [0, 0, width - 1, height - 1],
                            }

                        else:

                            xyz_path = osp.join(self.xyz_root, f"{scene_id:06d}/{int_im_id:06d}_{anno_i:06d}-xyz.pkl")
                            assert osp.exists(xyz_path), xyz_path
                            xyz = mmcv.load(xyz_path)
                            # begin to estimate new xyz
                            ply = self.model[str(obj_id)]
                            vert = np.asarray(
                                [ply['vertex'].data['x']/scale, ply['vertex'].data['y']/scale, ply['vertex'].data['z']/scale]).transpose()
                            norm_d = np.asarray(
                                [ply['vertex'].data['nx'], ply['vertex'].data['ny'], ply['vertex'].data['nz']]).transpose()
                            vert_id = [id for id in ply['face'].data['vertex_indices']]
                            vert_id = np.asarray(vert_id, np.int64)
                            pixellist = np.full([height, width], 100, dtype=np.float32)  # 加一个大数
                            # 实际上就是将每个3角面片投影回来，查看其中包含的整点像素，为其提供一个估计，然后最后选择能看到的那个，Z值最小
                            import time
                            start = time.time()
                            xyz_crop = calc_xy_crop(vert_id, vert, camK, R, t, norm_d, height, width, camK_inv, pixellist, mask, xyz["xyxy"])
                            print(time.time() - start)
                            x1, y1, x2, y2 = xyz["xyxy"]
                            P =  {
                                "xyz_crop": xyz_crop,
                                "xyxy": [x1, y1, x2, y2],
                            }

                        mmcv.dump(P, outpath)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="gen lm train_pbr xyz")
    parser.add_argument("--dataset", type=str, default="lm", help="dataset")
    parser.add_argument("--split", type=str, default="train_pbr", help="split")
    parser.add_argument("--xyz_name", type=str, default="xyz_crop_lm", help="xyz folder name")
    parser.add_argument("--threads", type=int, default=50, help="number of threads")
    args = parser.parse_args()

    base_dir = "/opt/spool/jemrich/BOP_DATASETS/"
    # base_dir = "/home/jemrich/datasets/BOP_DATASETS"
    model_dir = osp.join(base_dir, args.dataset, "models")
    root_dir = osp.join(base_dir, args.dataset, args.split)
    xyz_root = osp.join(root_dir, args.xyz_name)

    def gen_Q0(scenes):
        G_P = estimate_coor_P0(root_dir, model_dir, scenes)
        G_P.run()

    # scenes = np.array(range(10)).reshape((10,1))
    scenes = [[10]]
    with Pool(args.threads) as p:
        p.map(gen_Q0, scenes)