import os
from pathlib import Path

os.environ["PYOPENGL_PLATFORM"] = "egl"
import os.path as osp
import sys
from datetime import timedelta
from plyfile import PlyData
import mmcv
import numpy as np
from tqdm import tqdm

from multiprocessing import Pool
from numba import jit

cur_dir = osp.abspath(osp.dirname(__file__))
PROJ_ROOT = osp.join(cur_dir, "../../../..")
sys.path.insert(0, PROJ_ROOT)

idx2class = {
    1: "002_master_chef_can",  # [1.3360, -0.5000, 3.5105]
    2: "003_cracker_box",  # [0.5575, 1.7005, 4.8050]
    3: "004_sugar_box",  # [-0.9520, 1.4670, 4.3645]
    4: "005_tomato_soup_can",  # [-0.0240, -1.5270, 8.4035]
    5: "006_mustard_bottle",  # [1.2995, 2.4870, -11.8290]
    6: "007_tuna_fish_can",  # [-0.1565, 0.1150, 4.2625]
    7: "008_pudding_box",  # [1.1645, -4.2015, 3.1190]
    8: "009_gelatin_box",  # [1.4460, -0.5915, 3.6085]
    9: "010_potted_meat_can",  # [2.4195, 0.3075, 8.0715]
    10: "011_banana",  # [-18.6730, 12.1915, -1.4635]
    11: "019_pitcher_base",  # [5.3370, 5.8855, 25.6115]
    12: "021_bleach_cleanser",  # [4.9290, -2.4800, -13.2920]
    13: "024_bowl",  # [-0.2270, 0.7950, -2.9675]
    14: "025_mug",  # [-8.4675, -0.6995, -1.6145]
    15: "035_power_drill",  # [9.0710, 20.9360, -2.1190]
    16: "036_wood_block",  # [1.4265, -2.5305, 17.1890]
    17: "037_scissors",  # [7.0535, -28.1320, 0.0420]
    18: "040_large_marker",  # [0.0460, -2.1040, 0.3500]
    19: "051_large_clamp",  # [10.5180, -1.9640, -0.4745]
    20: "052_extra_large_clamp",  # [-0.3950, -10.4130, 0.1620]
    21: "061_foam_brick",  # [-0.0805, 0.0805, -8.2435]
}

# DEPTH_FACTOR = 10000.
IM_H = 480
IM_W = 640


K = np.array([[1066.778, 0.0, 312.9869079589844], [0.0, 1067.487, 241.3108977675438], [0.0, 0.0, 1.0]], dtype=np.float32)

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
def calc_xy_crop(
        vert_id,
        vert,
        camK,
        R,
        t,
        norm_d,
        height,
        width,
        camK_inv,
        pixellist,
        mask,
        x1,
        y1,
        x2,
        y2
    ):
    for i in range(vert_id.shape[0]):  # 行数
        P1 = transformer(vert[vert_id[i][0], :].T.copy(), R, t)
        P2 = transformer(vert[vert_id[i][1], :].T.copy(), R, t)
        P3 = transformer(vert[vert_id[i][2], :].T.copy(), R, t)
        p1 = projector(P1, camK, R, t)
        p2 = projector(P2, camK, R, t)  # col first
        p3 = projector(P3, camK, R, t)
        planenormal = norm_d[vert_id[i][0], :].copy()
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


def modelload(model_dir, ids, scale=1000.):
    modellist = {}
    for obj in ids:
        print("loading model", f"{obj:06d}")
        model_path = osp.join(model_dir, "obj_{:06d}.ply".format(obj))
        ply = PlyData.read(model_path)
        vert = np.asarray(
            [ply['vertex'].data['x'] / scale, ply['vertex'].data['y'] / scale,
             ply['vertex'].data['z'] / scale]).transpose()
        norm_d = np.asarray(
            [ply['vertex'].data['nx'], ply['vertex'].data['ny'], ply['vertex'].data['nz']]).transpose()
        vert_id = [id for id in ply['face'].data['vertex_indices']]
        vert_id = np.asarray(vert_id, np.int64)
        modellist[str(obj)] = {
            "vert": np.array(vert.astype("float32")),
            "norm_d": np.array(norm_d.astype("float32")),
            "vert_id": np.array(vert_id.astype("int64"))
        }
    return modellist

def get_time_delta(sec):
    """Humanize timedelta given in seconds."""
    if sec < 0:
        return "{:.3g} seconds".format(sec)
    delta_time_str = str(timedelta(seconds=sec))
    return delta_time_str

class XyzGen(object):
    def __init__(self, root_dir, model_dir, xyz_root_in, xyz_root_out, split="train", scene="all"):
        self.dataset_root = root_dir
        self.modeldir = model_dir
        self.split = split
        self.scene = scene
        cls_indexes = sorted(idx2class.keys())
        self.model = modelload(model_dir, cls_indexes)
        self.xyz_root_in = xyz_root_in
        self.xyz_root_out = xyz_root_out

    def main(self):
        split = self.split
        scene = self.scene  # "all" or a single scene
        camK = K.astype(np.float32)
        camK_inv = np.linalg.inv(K)
        height = 480
        width = 640
        for scene_id in tqdm(scene, postfix=f"{split}_{scene}"):
            scene_id = int(scene_id)
            print("split: {} scene: {}".format(split, scene_id))
            scene_root = osp.join(self.dataset_root, f"{scene_id:06d}")

            xyz_scene_path = Path(self.xyz_root_out, f"{scene_id:06d}")
            xyz_scene_path.mkdir(parents=True, exist_ok=True)

            gt_dict = mmcv.load(osp.join(scene_root, "scene_gt.json"))

            for str_im_id in tqdm(gt_dict, postfix=f"{scene_id}"):
                int_im_id = int(str_im_id)

                for anno_i, anno in enumerate(gt_dict[str_im_id]):
                    obj_id = anno["obj_id"]
                    outpath = osp.join(self.xyz_root_out, f"{scene_id:06d}/{int_im_id:06d}_{anno_i:06d}-xyz.npz")
                    if osp.exists(outpath):
                        continue
                    R = np.array(anno["cam_R_m2c"], dtype="float32").reshape(3, 3)
                    t = (np.array(anno["cam_t_m2c"], dtype="float32") / 1000.0).reshape(3, 1)
                    # mask_file = osp.join(scene_root, "mask/{:06d}_{:06d}.png".format(int_im_id, anno_i))
                    mask_visib_file = osp.join(scene_root, "mask/{:06d}_{:06d}.png".format(int_im_id, anno_i))
                    # assert osp.exists(mask_file), mask_file
                    # load mask visib  TODO: load both mask_visib and mask_full
                    mask = mmcv.imread(mask_visib_file, "unchanged")
                    mask = mask.astype(bool).astype(float)
                    if np.sum(mask) == 0:
                        xyz_crop = np.zeros((height, width, 3), dtype=np.float16)
                        xyxy = [0, 0, width - 1, height - 1]

                    else:

                        xyz_path = osp.join(self.xyz_root_in, f"{scene_id:06d}/{int_im_id:06d}_{anno_i:06d}-xyz.pkl")
                        assert osp.exists(xyz_path), xyz_path
                        xyz = mmcv.load(xyz_path)
                        # begin to estimate new xyz
                        vert = self.model[str(obj_id)]["vert"]
                        norm_d = self.model[str(obj_id)]["norm_d"]
                        vert_id = self.model[str(obj_id)]["vert_id"]

                        pixellist = np.full([height, width], 100, dtype=np.float32)
                        x1, y1, x2, y2 = xyz["xyxy"]
                        xyz_crop = calc_xy_crop(
                            vert_id,
                            vert, camK,
                            R,
                            t,
                            norm_d,
                            height,
                            width,
                            camK_inv,
                            pixellist,
                            mask,
                            x1,
                            y1,
                            x2,
                            y2,
                            )
                        xyxy = xyz["xyxy"]
                    np.savez_compressed(outpath, xyz_crop=xyz_crop, xyxy=xyxy)


if __name__ == "__main__":
    import argparse
    import time

    import setproctitle

    parser = argparse.ArgumentParser(description="gen denstereo train_pbr xyz")
    parser.add_argument("--bop_path", type=str, default="/opt/spool/jemrich/BOP_DATSASETS")
    parser.add_argument("--dataset", type=str, default="denstereo-test", help="dataset")
    parser.add_argument("--split", type=str, default="train", help="split")
    parser.add_argument("--scene", type=str, default="all", help="scene id")
    parser.add_argument("--xyz_in", type=str, default="xyz_crop", help="xyz low fidelity input folder name")
    parser.add_argument("--xyz_out", type=str, default="xyz_crop_hd", help="xyz high fidelity output folder name")
    parser.add_argument("--threads", type=int, default=1, help="number of threads")
    args = parser.parse_args()

    height = IM_H
    width = IM_W

    base_dir = args.bop_path
    # base_dir = "/home/jemrich/datasets/BOP_DATASETS"
    # base_dir = "/igd/a4/homestud/jemrich/datasets/BOP_DATASETS"
    model_dir = osp.join(base_dir, args.dataset, "models")
    root_dir = osp.join(base_dir, args.dataset, args.split)
    xyz_root_in = osp.join(root_dir, args.xyz_in)
    xyz_root_out = osp.join(root_dir, args.xyz_out)

    def gen_P(scenes):
        T_begin = time.perf_counter()
        setproctitle.setproctitle(f"gen_xyz_{args.dataset}_{args.split}_{args.scene}")
        xyz_gen = XyzGen(root_dir, model_dir, xyz_root_in, xyz_root_out, args.split, scenes)
        xyz_gen.main()
        T_end = time.perf_counter() - T_begin
        print("split", args.split, "scene", args.scene, "total time: ", get_time_delta(T_end))

    scenes = np.array(range(50)).reshape((50, 1))
    with Pool(args.threads) as p:
        p.map(gen_P, scenes)
