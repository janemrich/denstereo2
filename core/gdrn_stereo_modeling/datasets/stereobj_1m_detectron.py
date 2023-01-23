import hashlib
import logging
import os
import os.path as osp
import sys
import json
from pathlib import Path

# from lib.vis_utils.colormap import W

cur_dir = osp.dirname(osp.abspath(__file__))
PROJ_ROOT = osp.normpath(osp.join(cur_dir, "../../.."))
sys.path.insert(0, PROJ_ROOT)
import time
from collections import OrderedDict
import mmcv
import numpy as np
from tqdm import tqdm
from transforms3d.quaternions import mat2quat, quat2mat
import ref
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from lib.pysixd import inout, misc
from lib.utils.mask_utils import binary_mask_to_rle, cocosegm2mask
from lib.utils.utils import dprint, iprint, lazy_property
import scipy.ndimage as scin
import cv2

logger = logging.getLogger(__name__)
DATASETS_ROOT = osp.normpath(osp.join(PROJ_ROOT, "datasets"))


def resize_binary_map(binary_map, size):
    binary_map_tmp = []
    binary_map_shape = binary_map.shape
    if len(binary_map_shape) == 2:
        binary_map = np.expand_dims(binary_map, -1)
    for i in range(binary_map.shape[-1]):
        bm = binary_map[:, :, i]
        bm = cv2.resize(bm.astype('uint8'), size, \
                interpolation=cv2.INTER_NEAREST).astype('bool')
        binary_map_tmp.append(bm)
    binary_map = np.stack(binary_map_tmp, axis=-1)
    if len(binary_map_shape) == 2:
        binary_map = np.squeeze(binary_map)
    return binary_map


class STEREOBJ_1M_dataset:
    def __init__(self, data_cfg):
        """
        Set with_depth and with_masks default to True,
        and decide whether to load them into dataloader/network later
        with_masks:
        """
        self.name = data_cfg["name"]
        self.data_cfg = data_cfg

        self.objs = data_cfg["objs"]  # selected objects

        self.dataset_root = data_cfg.get(
            "dataset_root",
            osp.join(DATASETS_ROOT, "BOP_DATASETS/stereobj_1m/")
        )
        # self.xyz_root_l = data_cfg.get("xyz_root", osp.join(self.dataset_root, "xyz_crop_hd"))
        # self.xyz_root_r = data_cfg.get("xyz_root_right", osp.join(self.dataset_root_r, "xyz_crop_hd"))
        assert osp.exists(self.dataset_root), self.dataset_root
        self.models_root = data_cfg["models_root"]  # BOP_DATASETS/ycbv/models
        self.scale_to_meter = data_cfg["scale_to_meter"]  # 0.001

        self.with_masks = data_cfg["with_masks"]
        self.with_depth = data_cfg["with_depth"]

        self.height = data_cfg["height"]  # 480
        self.width = data_cfg["width"]  # 640

        self.cache_dir = data_cfg.get("cache_dir", osp.join(PROJ_ROOT, ".cache/gdrn_stereo/stereobj_1m/"))  # .cache
        self.use_cache = data_cfg.get("use_cache", True)
        self.num_to_load = data_cfg["num_to_load"]  # -1
        self.filter_invalid = data_cfg.get("filter_invalid", True)
        ##################################################

        # NOTE: careful! Only the selected objects
        self.cat_ids = [cat_id for cat_id, obj_name in ref.stereobj_1m.id2obj.items() if obj_name in self.objs]
        # map selected objs to [0, num_objs-1]
        self.cat2label = {v: i for i, v in enumerate(self.cat_ids)}  # id_map
        self.label2cat = {label: cat for cat, label in self.cat2label.items()}
        self.obj2label = OrderedDict((obj, obj_id) for obj_id, obj in enumerate(self.objs))
        ##########################################################

        self.scenes = data_cfg["scenes"]
        # self.scenes = [f"{i:d}" for i in data_cfg["scenes"]]
        self.id2scene = {i: scene for i, scene in enumerate(self.scenes)}
        self.scene2id = {scene: id for id, scene in self.id2scene.items()}
        self.debug_im_id = data_cfg.get("debug_im_id", None)
        self.max_im = data_cfg.get("max_im", None)

        cam_param_filename = Path(DATASETS_ROOT) / 'BOP_DATASETS' / 'stereobj_1m' / 'camera.json'
        with open(cam_param_filename, 'r') as f:
            cam_param = json.load(f)

        self.proj_matrix_l = np.array(cam_param['left']['P'])
        self.proj_matrix_r = np.array(cam_param['right']['P'])

        self.baseline = abs(self.proj_matrix_r[0, -1] / self.proj_matrix_r[0, 0])
            # self.baseline = np.array(json.load(cam_file)['baseline'], dtype=np.float32)

    def load_masks_and_bboxes(self, scene_root, int_im_id, obj):
        mask_visib_file = scene_root / "{:06d}_mask_label.npz".format(int_im_id)
        assert osp.exists(mask_visib_file), mask_visib_file
        # load mask visib  TODO: load both mask_visib and mask_full
        obj_mask = np.load(mask_visib_file, allow_pickle=True)['masks'].item()
        ##### decode instance mask
        mask_l = np.zeros([1440, 1440], dtype='bool')
        mask_r = np.zeros([1440, 1440], dtype='bool')

        mask_in_bbox_l = obj_mask['left'][obj]['mask']
        mask_in_bbox_r = obj_mask['right'][obj]['mask']
        x_min_l = obj_mask['left'][obj]['x_min']
        x_max_l = obj_mask['left'][obj]['x_max']
        y_min_l = obj_mask['left'][obj]['y_min']
        y_max_l = obj_mask['left'][obj]['y_max']
        x_min_r = obj_mask['right'][obj]['x_min']
        x_max_r = obj_mask['right'][obj]['x_max']
        y_min_r = obj_mask['right'][obj]['y_min']
        y_max_r = obj_mask['right'][obj]['y_max']

        if x_min_l is not None:
            mask_l[y_min_l:(y_max_l+1), x_min_l:(x_max_l+1)] = mask_in_bbox_l
        if x_min_r is not None:
            mask_r[y_min_r:(y_max_r+1), x_min_r:(x_max_r+1)] = mask_in_bbox_r

        mask_l = resize_binary_map(mask_l, (self.width, self.height))
        mask_r = resize_binary_map(mask_r, (self.width, self.height))
        mask_l = mask_l.astype('uint8')
        mask_r = mask_r.astype('uint8')

        visib_l = obj_mask['left'][obj]['percentage']
        visib_r = obj_mask['left'][obj]['percentage']

        return (
            mask_l,
            mask_r,
            [x_min_l, y_min_l, x_max_l, y_max_l],
            [x_min_r, y_min_r, x_max_r, y_max_r],
            visib_l,
            visib_r,
        )

    def __call__(self):
        """Load light-weight instance annotations of all images into a list of
        dicts in Detectron2 format.

        Do not load heavy data into memory in this file, since we will
        load the annotations of all images into memory.
        """
        # cache the dataset_dicts to avoid loading masks from files
        hashed_file_name = hashlib.md5(
            (
                "".join([str(fn) for fn in self.objs])
                + "dataset_dicts_{}_{}_{}_{}_{}".format(
                    self.name, self.dataset_root, self.with_masks, self.with_depth, __name__
                )
            ).encode("utf-8")
        ).hexdigest()
        cache_path = osp.join(self.cache_dir, "dataset_dicts_{}_{}.pkl".format(self.name, hashed_file_name))

        if osp.exists(cache_path) and self.use_cache:
            logger.info("load cached dataset dicts from {}".format(cache_path))
            return mmcv.load(cache_path)

        t_start = time.perf_counter()

        logger.info("loading dataset dicts: {}".format(self.name))
        self.num_instances_without_valid_segmentation = 0
        self.num_instances_without_valid_box = 0
        dataset_dicts = []  # ######################################################
        # it is slow because of loading and converting masks to rle
        for scene in tqdm(self.scenes):
            scene_id = self.scene2id[scene]
            # scene_root = osp.join(self.dataset_root, scene)
            scene_root = Path(self.dataset_root) / scene


            # gt_dict = mmcv.load(osp.join(scene_root, "scene_gt.json"))
            # load im_id list
            str_im_ids = [f.stem for f in scene_root.iterdir() if f.is_file() and f.suffix == '.jpg']


            # gt_info_dict = mmcv.load(osp.join(scene_root, "scene_gt_info.json"))
            # cam_dict_l = mmcv.load(osp.join(scene_root, "scene_camera.json"))
            # cam_dict_r = mmcv.load(osp.join(scene_root, "scene_camera.json"))

            for str_im_id in tqdm(str_im_ids, postfix=f"{scene_id}"):
                int_im_id = int(str_im_id)
                if self.max_im is not None:
                    if self.max_im < int_im_id:
                        continue

                gt_dict = json.load((
                    scene_root / "{}_rt_label.json".format(str_im_id)
                    ).open()
                )

                rgb_path = scene_root / "{}.jpg".format(str_im_id)
                assert osp.exists(rgb_path), rgb_path

                scene_im_id = f"{scene_id}/{int_im_id}"

                K = self.proj_matrix_l[:3, :3]
                # K = np.array(cam_dict_l[str_im_id]["cam_K"], dtype=np.float32).reshape(3, 3)
                # depth_factor = 1000.0 / cam_dict_l[str_im_id]["depth_scale"]  # 10000

                record = {
                    "dataset_name": self.name,
                    "file_name": str(rgb_path),
                    "height": self.height,
                    "width": self.width,
                    "image_id": int_im_id,
                    "scene_im_id": scene_im_id,  # for evaluation
                    "cam": K,
                    "baseline": [self.baseline, 0, 0],
                    # "depth_factor": depth_factor,
                    "img_type": "real",  # NOTE: has background
                }
                insts = []
                for anno_i, (obj_number, obj) in enumerate(gt_dict["class"].items()):
                    # anno_r = gt_dict_r[str_im_id][anno_i]
                    # obj_id = anno_l["obj_id"]
                    if obj not in self.objs:
                        continue

                    obj_label = self.obj2label[obj]
                    scene_im_anno_id = "{:d}/{:d}/{:d}".format(scene_id, int_im_id, obj_label)
                    if self.debug_im_id is not None:
                        if self.debug_im_id != scene_im_anno_id:
                            continue
                    
                    # cur_label = self.cat2label[obj_id]  # 0-based label
                    R = np.array(gt_dict['rt'][obj_number]['R'], dtype="float32")
                    # R_l = np.array(anno_l["cam_R_m2c"], dtype="float32").reshape(3, 3)
                    # R_r = np.array(anno_r["cam_R_m2c"], dtype="float32").reshape(3, 3)
                    t = np.array(gt_dict['rt'][obj_number]['t'], dtype="float32")
                    # t_l = np.array(anno_l["cam_t_m2c"], dtype="float32") / 1000.0
                    # t_r = np.array(anno_r["cam_t_m2c"], dtype="float32") / 1000.0
                    pose = np.hstack([R, t.reshape(3, 1)])

                    # P_l2 = np.vstack([self.proj_matrix_l, [0,0,0,1]])
                    # P_r2 = np.vstack([self.proj_matrix_r, [0,0,0,1]])
                    # P_r_inv = np.linalg.inv(P_r2)
                    # pose_r = (pose @ P_l2) @ P_r_inv

                    # R_r = pose_r[:3,:3]
                    # t_r = pose_r[:3,3]

                    quat = mat2quat(R).astype("float32")
                    # quat_r = mat2quat(R_r).astype("float32")

                    proj = (record["cam"] @ t.T).T
                    # proj_r = (record["cam"] @ t_r.T).T
                    proj = proj[:2] / proj[2]
                    # proj_r = proj_r[:2] / proj_r[2]

                    # bbox_obj = gt_info_dict[str_im_id][anno_i]["bbox_obj"]
                    # x1, y1, w, h = bbox_visib_l

                    (
                        mask_single_l,
                        mask_single_r,
                        bbox_visib_l,
                        bbox_visib_r,
                        visib_fract_l,
                        visib_fract_r,
                    ) = self.load_masks_and_bboxes(
                                scene_root,
                                int_im_id,
                                obj_number,
                            )

                    mask_single_erode_l = scin.binary_erosion(mask_single_l.astype(np.int))
                    mask_single_erode_r = scin.binary_erosion(mask_single_r.astype(np.int))
                    mask_single_l = mask_single_l.astype(np.bool).astype(np.int)
                    mask_single_r = mask_single_r.astype(np.bool).astype(np.int)
                    area_l = mask_single_erode_l.sum()
                    area_r = mask_single_erode_r.sum()
                    if (area_l <= 64) or (area_r <= 64):  # filter out too small or nearly invisible instances
                        self.num_instances_without_valid_segmentation += 1
                        continue

                    mask_rle_l = binary_mask_to_rle(mask_single_l, compressed=True)
                    mask_rle_r = binary_mask_to_rle(mask_single_r, compressed=True)

                    xyz_path = osp.join(
                        scene_root,
                        f"{int_im_id:06d}_{ref.stereobj_1m.obj2id[obj]:06d}_xyz.npz"
                    )
                    # xyz_path_r = osp.join(self.xyz_root_r, f"{scene_id:06d}/{int_im_id:06d}_{anno_i:06d}-xyz.npz")
                    assert osp.exists(xyz_path), xyz_path
                    # assert osp.exists(xyz_path_r), xyz_path_r
                    inst = {
                        "category_id": obj_label,  # 0-based label
                        "bbox_l": BoxMode.convert(bbox_visib_l, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS),  # TODO: load both bbox_obj and bbox_visib
                        "bbox_r": BoxMode.convert(bbox_visib_r, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS), # TODO: load both bbox_obj and bbox_visib
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "pose": pose,
                        # "pose_l": pose_l,
                        # "pose_r": pose_r,
                        "quat": quat,
                        # "quat_l": quat_l,
                        # "quat_r": quat_r,
                        "trans": t,
                        # "trans_l": t_l,
                        # "trans_r": t_r,
                        "centroid_2d": proj,  # absolute (cx, cy)
                        # "centroid_2d_l": proj_l,  # absolute (cx, cy)
                        # "centroid_2d_r": proj_r,  # absolute (cx, cy)
                        "segmentation_l": mask_rle_l, #500 by 500
                        "segmentation_r": mask_rle_r,
                        # "mask_full_file_l": mask_file_l,  # TODO: load as mask_full, rle
                        # "mask_full_file_r": mask_file_r,  # TODO: load as mask_full, rle
                        "visib_fract": (visib_fract_l + visib_fract_r) / 2,
                        "xyz_path": xyz_path,
                        # "occ_path_l": occ_path_l,
                        # "occ_path_r": occ_path_r,
                    }

                    model_info = self.models_info[str(obj_label)] #TODO
                    inst["model_info"] = model_info
                    # TODO: using full mask and full xyz
                    # for key in ["bbox3d_and_center"]:
                        # inst[key] = self.models[cur_label][key]
                    insts.append(inst)
                if len(insts) == 0:  # filter im without anno
                    continue
                record["annotations"] = insts
                dataset_dicts.append(record)

        if self.num_instances_without_valid_segmentation > 0:
            logger.warning(
                "Filtered out {} instances without valid segmentation. "
                "There might be issues in your dataset generation process.".format(
                    self.num_instances_without_valid_segmentation
                )
            )
        if self.num_instances_without_valid_box > 0:
            logger.warning(
                "Filtered out {} instances without valid box. "
                "There might be issues in your dataset generation process.".format(self.num_instances_without_valid_box)
            )
        ##########################################################################
        if self.num_to_load > 0:
            self.num_to_load = min(int(self.num_to_load), len(dataset_dicts))
            dataset_dicts = dataset_dicts[: self.num_to_load]
        logger.info("loaded {} dataset dicts, using {}s".format(len(dataset_dicts), time.perf_counter() - t_start))

        mmcv.mkdir_or_exist(osp.dirname(cache_path))
        mmcv.dump(dataset_dicts, cache_path, protocol=4)
        logger.info("Dumped dataset_dicts to {}".format(cache_path))
        return dataset_dicts

    @lazy_property
    def models_info(self):
        models_info_path = osp.join(self.models_root, "models_info.json")
        assert osp.exists(models_info_path), models_info_path
        models_info = mmcv.load(models_info_path)  # key is str(obj_id)
        return models_info

    @lazy_property
    def models(self):
        """Load models into a list."""
        cache_path = osp.join(self.models_root, "models_{}.pkl".format(self.name))
        if osp.exists(cache_path) and self.use_cache:
            # dprint("{}: load cached object models from {}".format(self.name, cache_path))
            return mmcv.load(cache_path)

        models = []
        for obj_name in self.objs:
            model = inout.load_ply(
                osp.join(self.models_root, f"obj_{ref.stereobj_1m.obj2id[obj_name]:06d}.ply"), vertex_scale=self.scale_to_meter
            )
            # NOTE: the bbox3d_and_center is not obtained from centered vertices
            # for BOP models, not a big problem since they had been centered
            model["bbox3d_and_center"] = misc.get_bbox3d_and_center(model["pts"])

            models.append(model)
        logger.info("cache models to {}".format(cache_path))
        mmcv.dump(models, cache_path, protocol=4)
        return models

    def image_aspect_ratio(self):
        return self.width / self.height  # 4/3


########### register datasets ############################################################


def get_stereobj_1m_metadata(obj_names, ref_key):
    """task specific metadata."""
    data_ref = ref.__dict__[ref_key]

    '''
    cur_sym_infos = {}  # label based key
    loaded_models_info = data_ref.get_models_info()

    for i, obj_name in enumerate(obj_names):
        obj_id = data_ref.obj2id[obj_name]
        model_info = loaded_models_info[str(obj_id)]
        if "symmetries_discrete" in model_info or "symmetries_continuous" in model_info:
            sym_transforms = misc.get_symmetry_transformations(model_info, max_sym_disc_step=0.01)
            sym_info = np.array([sym["R"] for sym in sym_transforms], dtype=np.float32)
        else:
            sym_info = None
        cur_sym_infos[i] = sym_info
    '''

    # meta = {"thing_classes": obj_names, "sym_infos": cur_sym_infos}
    meta = {"thing_classes": obj_names}
    return meta


stereobj_1m_model_root = "BOP_DATASETS/stereobj_1m/models/"
################################################################################


SPLITS_STEREOBJ_1M = dict(
    stereobj_1m_train =dict(
        name="stereobj_1m_train",
        objs=ref.stereobj_1m.objects,  # selected objects
        dataset_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/stereobj_1m/"),
        models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/stereobj_1m/models"),
        # xyz_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/stereobj_1m/train_pbr_left/xyz_crop_hd"),
        scale_to_meter=1.0,
        with_masks=True,  # (load masks but may not use it)
        with_depth=False,  # (load depth path here, but may not use it)
        height=1440,
        width=1440,
        use_cache=True,
        num_to_load=-1,
        filter_invalid=True,
        scenes=ref.stereobj_1m.train_scenes,
        ref_key="stereobj_1m",
    ),
    stereobj_1m_test_pbr =dict(
        name="stereobj_1m_test",
        objs=ref.stereobj_1m.objects,  # selected objects
        dataset_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/stereobj_1m"),
        models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/stereobj_1m/models"),
        # xyz_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/stereobj_1m/train_pbr_left/xyz_crop_hd"),
        scale_to_meter=1.0,
        with_masks=True,  # (load masks but may not use it)
        with_depth=False,  # (load depth path here, but may not use it)
        height=1440,
        width=1440,
        use_cache=False,
        num_to_load=-1,
        filter_invalid=True,
        scenes=ref.stereobj_1m.test_scenes,
        ref_key="stereobj_1m",
    ),
    stereobj_1m_debug_train =dict(
        name="stereobj_1m_debug_train",
        objs=ref.stereobj_1m.mechanics_objects,  # selected objects
        dataset_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/stereobj_1m"),
        models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/stereobj_1m/models"),
        # xyz_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/stereobj_1m/train_pbr_left/xyz_crop_hd"),
        scale_to_meter=1.0,
        with_masks=True,  # (load masks but may not use it)
        with_depth=True,  # (load depth path here, but may not use it)
        height=1440,
        width=1440,
        use_cache=False,
        num_to_load=-1,
        filter_invalid=True,
        scenes=ref.stereobj_1m.debug_scenes,
        ref_key="stereobj_1m",
    ),
    stereobj_1m_debug_test =dict(
        name="stereobj_1m_debug_test",
        objs=ref.stereobj_1m.mechanics_objects,  # selected objects
        dataset_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/stereobj_1m"),
        models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/stereobj_1m/models"),
        # xyz_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/stereobj_1m/train_pbr_left/xyz_crop_hd"),
        scale_to_meter=0.001,
        with_masks=True,  # (load masks but may not use it)
        with_depth=True,  # (load depth path here, but may not use it)
        height=1440,
        width=1440,
        use_cache=True,
        num_to_load=-1,
        filter_invalid=True,
        scenes=ref.stereobj_1m.debug_scenes,
        max_im = 100,
        ref_key="stereobj_1m",
    ),
)

# single obj splits
for obj in ref.stereobj_1m.objects:
    for split in ["train"]:
        name = "stereobj_1m_{}_{}".format(obj, split)
        if split in ["train"]:
            filter_invalid = True
        elif split in ["test"]:
            filter_invalid = False
        else:
            raise ValueError("{}".format(split))
        if name not in SPLITS_STEREOBJ_1M:
            SPLITS_STEREOBJ_1M[name] = dict(
                name=name,
                objs=[obj],  # only this obj
                dataset_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/stereobj_1m"),
                models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/stereobj_1m/models"),
                # xyz_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/stereobj_1m/train_pbr_left/xyz_crop_hd"),
                scale_to_meter=1,
                with_masks=True,  # (load masks but may not use it)
                with_depth=True,  # (load depth path here, but may not use it)
                height=1440,
                width=1440,
                use_cache=True,
                num_to_load=-1,
                filter_invalid=filter_invalid,
                scenes=ref.stereobj_1m.train_scenes,
                ref_key="stereobj_1m",
            )

for obj in ref.stereobj_1m.objects:
    for split in ["test_pbr"]:
        name = "stereobj_1m_{}_{}".format(obj, split)
        if split in ["train_pbr"]:
            filter_invalid = True
        elif split in ["test_pbr"]:
            filter_invalid = True
        elif split in ["test"]:
            filter_invalid = False
        else:
            raise ValueError("{}".format(split))
        if name not in SPLITS_STEREOBJ_1M:
            SPLITS_STEREOBJ_1M[name] = dict(
                name=name,
                objs=[obj],  # only this obj
                dataset_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/stereobj_1m/train_pbr_left"),
                dataset_root_right=osp.join(DATASETS_ROOT, "BOP_DATASETS/stereobj_1m/train_pbr_right"),
                models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/stereobj_1m/models"),
                xyz_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/stereobj_1m/train_pbr_left/xyz_crop_hd"),
                xyz_root_right=osp.join(DATASETS_ROOT, "BOP_DATASETS/stereobj_1m/train_pbr_right/xyz_crop_hd"),
                scale_to_meter=1.0,
                with_masks=True,  # (load masks but may not use it)
                with_depth=True,  # (load depth path here, but may not use it)
                height=1440,
                width=1440,
                use_cache=True,
                num_to_load=-1,
                filter_invalid=filter_invalid,
                scenes=ref.stereobj_1m.test_scenes,
                ref_key="stereobj_1m",
            )

'''
# ================ add single image dataset for debug =======================================
for split in ['train_pbr_left']: # TODO add train _right 'train_pbr_right']:
    for scene in ref.stereobj_1m.debug_pbr_scenes:
        scene_root = osp.join(DATASETS_ROOT, "BOP_DATASETS/stereobj_1m/{:s}/{:06d}".format(
                split,
                scene
            )
        )
        gt_dict = mmcv.load(osp.join(scene_root, "scene_gt.json"))
        for im_id in gt_dict.keys():
            int_im_id = int(im_id)
            obj_ids = [pose['obj_id'] for pose in gt_dict[im_id]]
            for obj_id in obj_ids:
                name = "stereobj_1m_single_{}_{}_{}_{}".format(scene, int_im_id, obj_id, split)
                if name not in SPLITS_STEREOBJ_1M:
                    scene_image_obj_id = "{:d}/{:d}/{:d}".format(scene, int_im_id, obj_id)
                    SPLITS_STEREOBJ_1M[name] = dict(
                        name=name,
                        dataset_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/stereobj_1m/{:s}".format(split)),
                        models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/stereobj_1m/models"),
                        objs=ref.stereobj_1m.objects,  # only this obj
                        image_prefixes=[
                            osp.join(DATASETS_ROOT, "BOP_DATASETS/stereobj_1m/{:s}/{:06d}").format(split, scene)
                        ],
                        xyz_prefixes=[
                            osp.join(DATASETS_ROOT, "BOP_DATASETS/stereobj_1m/{:s}/xyz_crop_hd/{:06d}".format(split, scene))
                        ],
                        scale_to_meter=0.001,
                        with_masks=True,  # (load masks but may not use it)
                        with_depth=True,  # (load depth path here, but may not use it)
                        height=480,
                        width=640,
                        cache_dir=osp.join(PROJ_ROOT, ".cache"),
                        use_cache=True,
                        num_to_load=-1,
                        filter_invalid=False,
                        filter_scene=True,
                        debug_im_id = scene_image_obj_id, # NOTE: debug im id
                        scenes = ref.stereobj_1m.debug_pbr_scenes,
                        ref_key="stereobj_1m",
                    )
'''

# single image stereo

'''
for scene in ref.stereobj_1m.debug_scenes:
    scene_root = osp.join(DATASETS_ROOT, "BOP_DATASETS/stereobj_1m/{:s}/{:06d}".format(
            'train_pbr_left',
            scene
        )
    )
    gt_dict = mmcv.load(osp.join(scene_root, "scene_gt.json"))
    for im_id in gt_dict.keys():
        int_im_id = int(im_id)
        obj_ids = [pose['obj_id'] for pose in gt_dict[im_id]]
        for obj_id in obj_ids:
            name = "stereobj_1m_single_{}_{}_{}_stereo".format(scene, int_im_id, obj_id)
            if name not in SPLITS_STEREOBJ_1M:
                scene_image_obj_id = "{:d}/{:d}/{:d}".format(scene, int_im_id, obj_id)
                SPLITS_STEREOBJ_1M[name] = dict(
                    name=name,
                    dataset_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/stereobj_1m/train_pbr_left"),
                    dataset_root_right=osp.join(DATASETS_ROOT, "BOP_DATASETS/stereobj_1m/train_pbr_right"),
                    models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/stereobj_1m/models"),
                    objs=ref.stereobj_1m.objects,  # only this obj
                    image_prefixes=[
                        osp.join(DATASETS_ROOT, "BOP_DATASETS/stereobj_1m/train_pbr_left/{:06d}").format(scene)
                    ],
                    image_prefixes_right=[
                        osp.join(DATASETS_ROOT, "BOP_DATASETS/stereobj_1m/train_pbr_right/{:06d}").format(scene)
                    ],
                    xyz_prefixes=[
                        osp.join(DATASETS_ROOT, "BOP_DATASETS/stereobj_1m/train_pbr_left/xyz_crop_hd/{:06d}".format(scene))
                    ],
                    xyz_prefixes_right=[
                        osp.join(DATASETS_ROOT, "BOP_DATASETS/stereobj_1m/train_pbr_right/xyz_crop_hd/{:06d}".format(scene))
                    ],
                    scale_to_meter=0.001,
                    with_masks=True,  # (load masks but may not use it)
                    with_depth=True,  # (load depth path here, but may not use it)
                    height=480,
                    width=640,
                    cache_dir=osp.join(PROJ_ROOT, ".cache"),
                    use_cache=True,
                    num_to_load=-1,
                    filter_invalid=False,
                    filter_scene=True,
                    debug_im_id = scene_image_obj_id, # NOTE: debug im id
                    scenes = ref.stereobj_1m.debug_pbr_scenes,
                    ref_key="stereobj_1m",
                )
'''

def register_with_name_cfg(name, data_cfg=None):
    """Assume pre-defined datasets live in `./datasets`.

    Args:
        name: datasnet_name,
        data_cfg: if name is in existing SPLITS, use pre-defined data_cfg
            otherwise requires data_cfg
            data_cfg can be set in cfg.DATA_CFG.name
    """
    dprint("register dataset: {}".format(name))
    if name in SPLITS_STEREOBJ_1M:
        used_cfg = SPLITS_STEREOBJ_1M[name]
    else:
        assert data_cfg is not None, f"dataset name {name} is not registered"
        used_cfg = data_cfg
    DatasetCatalog.register(name, STEREOBJ_1M_dataset(used_cfg))
    # something like eval_types
    MetadataCatalog.get(name).set(
        id="stereobj_1m",  # NOTE: for pvnet to determine module
        ref_key=used_cfg["ref_key"],
        objs=used_cfg["objs"],
        eval_error_types=["ad", "rete", "proj"],
        evaluator_type="bop",
        **get_stereobj_1m_metadata(obj_names=used_cfg["objs"], ref_key=used_cfg["ref_key"]),
    )


def get_available_datasets():
    return list(SPLITS_STEREOBJ_1M.keys())



def test_vis():
    dset_name = sys.argv[1]
    assert dset_name in DatasetCatalog.list()

    meta = MetadataCatalog.get(dset_name)
    dprint("MetadataCatalog: ", meta)
    objs = meta.objs

    t_start = time.perf_counter()
    dicts = DatasetCatalog.get(dset_name)
    logger.info("Done loading {} samples with {:.3f}s.".format(len(dicts), time.perf_counter() - t_start))

    dirname = "output/{}-data-vis".format(dset_name)
    os.makedirs(dirname, exist_ok=True)
    for d in dicts:
        img = read_image_mmcv(d["file_name"], format="BGR")
        depth = mmcv.imread(d["depth_file"], "unchanged") / 10000.0

        imH, imW = img.shape[:2]
        annos = d["annotations"]
        masks = [cocosegm2mask(anno["segmentation"], imH, imW) for anno in annos]
        bboxes = [anno["bbox"] for anno in annos]
        bbox_modes = [anno["bbox_mode"] for anno in annos]
        bboxes_xyxy = np.array(
            [BoxMode.convert(box, box_mode, BoxMode.XYXY_ABS) for box, box_mode in zip(bboxes, bbox_modes)]
        )
        kpts_3d_list = [anno["bbox3d_and_center"] for anno in annos]
        quats = [anno["quat"] for anno in annos]
        transes = [anno["trans"] for anno in annos]
        Rs = [quat2mat(quat) for quat in quats]
        # 0-based label
        cat_ids = [anno["category_id"] for anno in annos]
        K = d["cam"]
        kpts_2d = [misc.project_pts(kpt3d, K, R, t) for kpt3d, R, t in zip(kpts_3d_list, Rs, transes)]

        labels = [objs[cat_id] for cat_id in cat_ids]
        for _i in range(len(annos)):
            img_vis = vis_image_mask_bbox_cv2(
                img, masks[_i : _i + 1], bboxes=bboxes_xyxy[_i : _i + 1], labels=labels[_i : _i + 1]
            )
            img_vis_kpts2d = misc.draw_projected_box3d(img_vis.copy(), kpts_2d[_i])
            xyz_path = annos[_i]["xyz_path"]
            xyz_info = mmcv.load(xyz_path)
            x1, y1, x2, y2 = xyz_info["xyxy"]
            xyz_crop = xyz_info["xyz_crop"].astype(np.float32)
            xyz = np.zeros((imH, imW, 3), dtype=np.float32)
            xyz[y1 : y2 + 1, x1 : x2 + 1, :] = xyz_crop
            xyz_show = get_emb_show(xyz)
            xyz_crop_show = get_emb_show(xyz_crop)
            img_xyz = img.copy() / 255.0
            mask_xyz = ((xyz[:, :, 0] != 0) | (xyz[:, :, 1] != 0) | (xyz[:, :, 2] != 0)).astype("uint8")
            fg_idx = np.where(mask_xyz != 0)
            img_xyz[fg_idx[0], fg_idx[1], :] = xyz_show[fg_idx[0], fg_idx[1], :3]
            img_xyz_crop = img_xyz[y1 : y2 + 1, x1 : x2 + 1, :]
            img_vis_crop = img_vis[y1 : y2 + 1, x1 : x2 + 1, :]
            # diff mask
            diff_mask_xyz = np.abs(masks[_i] - mask_xyz)[y1 : y2 + 1, x1 : x2 + 1]

            grid_show(
                [
                    img[:, :, [2, 1, 0]],
                    img_vis[:, :, [2, 1, 0]],
                    img_vis_kpts2d[:, :, [2, 1, 0]],
                    depth,
                    # xyz_show,
                    diff_mask_xyz,
                    xyz_crop_show,
                    img_xyz[:, :, [2, 1, 0]],
                    img_xyz_crop[:, :, [2, 1, 0]],
                    img_vis_crop,
                ],
                [
                    "img",
                    "vis_img",
                    "img_vis_kpts2d",
                    "depth",
                    "diff_mask_xyz",
                    "xyz_crop_show",
                    "img_xyz",
                    "img_xyz_crop",
                    "img_vis_crop",
                ],
                row=3,
                col=3,
            )


if __name__ == "__main__":
    """Test the  dataset loader.

    Usage:
        python -m core/datasets/ycbv_pbr.py ycbv_pbr_train
    """
    from lib.vis_utils.image import grid_show
    from lib.utils.setup_logger import setup_my_logger

    import detectron2.data.datasets  # noqa # add pre-defined metadata
    from lib.vis_utils.image import vis_image_mask_bbox_cv2
    from core.utils.utils import get_emb_show
    from core.utils.data_utils import read_image_mmcv

    print("sys.argv:", sys.argv)
    logger = setup_my_logger(name="core")
    register_with_name_cfg(sys.argv[1])
    print("dataset catalog: ", DatasetCatalog.list())

    test_vis()
