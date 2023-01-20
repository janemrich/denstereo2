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

id2obj = {
    # 1: "blade_razor",
    # 10: "microplate",
    # 14: "sterile_tip_rack_10",
    15: "sterile_tip_rack_200",
    # 16: "sterile_tip_rack_1000",
    # 13: "pipette_100_1000",
}
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

    
def get_scenes(dataset_root):
    trainval_scenes = []
    with open(osp.join(dataset_root, "split", "biolab_trainval_scenes.txt"), "r") as f:
        # read all lines without newline char
        trainval_scenes = f.read().splitlines()
    # extend with mechanics scenes
    with open(osp.join(dataset_root, "split", "mechanics_trainval_scenes.txt"), "r") as f:
        # read all lines without newline char
        trainval_scenes.extend(f.read().splitlines())

    return sorted(trainval_scenes)

'''
def get_scene_dict(dataset_root):
    scenes = get_scenes(dataset_root)
    scene_ids = {i: scene for i, scene in enumerate(scenes)}
    return scene_ids
'''

# DEPTH_FACTOR = 10000.
IM_H = 1440
IM_W = 1440
ORIG_H = 1440
ORIG_W = 1440

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

def get_camera_params(dataset_dir):
    cam_param_filename = dataset_dir / 'camera.json'
    with open(cam_param_filename, 'r') as f:
        cam_param = json.load(f)

    return (
        np.array(cam_param['left']['P'], dtype=np.float32)[:3, :3],
        np.array(cam_param['right']['P'], dtype=np.float32)[:3, -1].reshape(3, 1),
    )
# K = np.array([[1066.778, 0.0, 312.9869079589844], [0.0, 1067.487, 241.3108977675438], [0.0, 0.0, 1.0]], dtype=np.float32)


def modelload(model_dir, ids, scale=1):
    modellist = {}
    for obj in ids:
        print("loading model", f"{obj}")
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
    def __init__(self, base_dir, dataset, scene):
        self.base_dir = Path(base_dir)
        self.dataset_root = self.base_dir / dataset
        model_dir = self.dataset_root / "models"
        self.scene = scene
        cls_indexes = sorted(id2obj.keys())
        self.model = modelload(model_dir, cls_indexes)
        self.camera, self.baseline = get_camera_params(self.dataset_root)
        # print(self.camera)

    def main(self):
        camK = self.camera
        # print("camK", camK)
        camK_inv = np.linalg.inv(camK)
        height = IM_H
        width = IM_W
        for scene in tqdm(self.scene, postfix=f"{self.scene}"):
            scene_root = self.dataset_root / scene

            for f in tqdm(scene_root.glob('*.jpg'), leave=False, desc=f"Image in Scene {scene}"):
                print(f.stem)

                label_path = scene_root / (f.stem + '_rt_label.json')
                # print('label_path', label_path)
                with open(label_path, 'r') as rt_f:
                    rt_data = json.load(rt_f)

                mask_path = scene_root / (f.stem + '_mask_label.npz')
                obj_masks = np.load(mask_path, allow_pickle=True)['masks'].item()
                for obj_number, obj in rt_data['class'].items():
                    if obj not in obj2id:
                        # print('skip object')
                        continue
                    outpath = scene_root / '{}_{:06d}_xyz.npz'.format(f.stem, int(obj2id[obj]))
                    # if osp.exists(outpath):
                        # print('skip, already exists')
                        # continue
                    xyz = {
                        'left': {},
                        'right': {}
                    }
                    for side in ['left', 'right']:

                        # print('obj', obj)
                        mask = np.zeros([1440, 1440], dtype='bool')
                        mask_in_bbox = obj_masks[side][obj_number]['mask']

                        R = np.array(rt_data['rt'][obj_number]['R'], dtype="float32")
                        t = np.array(rt_data['rt'][obj_number]['t'], dtype="float32").reshape((3, 1))
                        if side == 'right':
                            t += camK_inv @ self.baseline

                        mask_in_bbox = mask_in_bbox.astype(bool).astype(float)
                        pixellist = np.full([height, width], 100, dtype=np.float32)
                        x1 = obj_masks[side][obj_number]['x_min']
                        x2 = obj_masks[side][obj_number]['x_max']
                        y1 = obj_masks[side][obj_number]['y_min']
                        y2 = obj_masks[side][obj_number]['y_max']
                        if x1 is not None:
                            mask[y1:(y2+1), x1:(x2+1)] = mask_in_bbox
                        if np.sum(mask_in_bbox) == 0:
                            xyz[side]['xyz_crop'] = np.zeros((height, width, 3), dtype=np.float16)
                            xyz[side]['xyxy'] = [0, 0, width - 1, height - 1]
                        else:
                            # begin to estimate new xyz
                            obj_id = str(obj2id[obj])
                            vert = self.model[obj_id]["vert"]
                            norm_d = self.model[obj_id]["norm_d"]
                            vert_id = self.model[obj_id]["vert_id"]

                            xyz[side]['xyxy'] = [x1, y1, x2, y2]
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
                            xyz[side]['xyz_crop'] = xyz_crop
                    show = False
                    if show:
                        import matplotlib.pyplot as plt
                        _, axarr = plt.subplots(1,2)
                        axarr[0].imshow(xyz['left']['xyz_crop'])
                        axarr[1].imshow(xyz['right']['xyz_crop'])
                        plt.savefig(f"xyz_{obj}.png",  bbox_inches = 'tight', pad_inches = 0)
                    np.savez_compressed(outpath, xyz=xyz)


if __name__ == "__main__":
    import argparse
    import time

    import setproctitle

    parser = argparse.ArgumentParser(description="gen denstereo train_pbr xyz")
    parser.add_argument("--bop_path", type=str, default="../../datasets/BOP_DATASETS")
    parser.add_argument("--dataset", type=str, default="stereobj_1m", help="dataset")
    parser.add_argument("--scenes", type=str, default="all", help="scene id")
    # parser.add_argument("--xyz_out", type=str, default="xyz_crop_hd", help="xyz high fidelity output folder name")
    parser.add_argument("--threads", type=int, default=1, help="number of threads")
    args = parser.parse_args()

    base_dir = args.bop_path

    scenes = np.array(get_scenes(Path(base_dir) / args.dataset))
    if args.scenes != "all":
        # scenes = {args.scenes: scenes[args.scenes]}
        scenes = scenes[:int(args.scenes)]

    for scene in scenes:
        # if 'biolab' in scene: continue
        # print("scene: {}".format(scene))
        xyz_gen = XyzGen(base_dir, args.dataset, [scene])
        xyz_gen.main()

    def gen_P(scenes):
        T_begin = time.perf_counter()
        setproctitle.setproctitle(
            f"gen_xyz_{args.dataset}_{args.scene}"
        )

        xyz_gen = XyzGen(base_dir, args.dataset, scenes)
        xyz_gen.main()

        T_end = time.perf_counter() - T_begin
        print(
            "scene {}".format(args.scene),
            "total time: {}".format(get_time_delta(T_end)),
        )

    # scenes = np.array(scenes.keys()).reshape((len(scenes), 1))
    scenes = scenes.reshape((-1, 1))
    print(scenes)
    # with Pool(args.threads) as p:
        # p.map(gen_P, scenes)
