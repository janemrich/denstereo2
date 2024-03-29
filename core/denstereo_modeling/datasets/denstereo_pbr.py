import hashlib
import logging
import os
import os.path as osp
import sys
import json

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

logger = logging.getLogger(__name__)
DATASETS_ROOT = osp.normpath(osp.join(PROJ_ROOT, "datasets"))


class DENSTEREO_PBR_Dataset:
    def __init__(self, data_cfg):
        """
        Set with_depth and with_masks default to True,
        and decide whether to load them into dataloader/network later
        with_masks:
        """
        self.name = data_cfg["name"]
        self.data_cfg = data_cfg

        self.objs = data_cfg["objs"]  # selected objects

        self.dataset_root_l = data_cfg.get(
            "dataset_root",
            osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/train_pbr_left")
        )
        self.dataset_root_r = data_cfg.get(
            "dataset_root_right",
            osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/train_pbr_right")
        )
        self.xyz_root_l = data_cfg.get("xyz_root", osp.join(self.dataset_root_l, "xyz_crop_hd"))
        self.xyz_root_r = data_cfg.get("xyz_root_right", osp.join(self.dataset_root_r, "xyz_crop_hd"))
        self.occ_root_l = data_cfg.get("occ_root", osp.join(self.dataset_root_l, "Q0"))
        self.occ_root_r = data_cfg.get("occ_root", osp.join(self.dataset_root_r, "Q0"))
        assert osp.exists(self.dataset_root_l), self.dataset_root_l
        self.models_root = data_cfg["models_root"]  # BOP_DATASETS/ycbv/models
        self.scale_to_meter = data_cfg["scale_to_meter"]  # 0.001

        self.with_masks = data_cfg["with_masks"]
        self.with_depth = data_cfg["with_depth"]

        self.height = data_cfg["height"]  # 480
        self.width = data_cfg["width"]  # 640

        self.cache_dir = data_cfg.get("cache_dir", osp.join(PROJ_ROOT, ".cache"))  # .cache
        self.use_cache = data_cfg.get("use_cache", True)
        self.num_to_load = data_cfg["num_to_load"]  # -1
        self.filter_invalid = data_cfg.get("filter_invalid", True)
        ##################################################

        # NOTE: careful! Only the selected objects
        self.cat_ids = [cat_id for cat_id, obj_name in ref.denstereo.id2obj.items() if obj_name in self.objs]
        # map selected objs to [0, num_objs-1]
        self.cat2label = {v: i for i, v in enumerate(self.cat_ids)}  # id_map
        self.label2cat = {label: cat for cat, label in self.cat2label.items()}
        self.obj2label = OrderedDict((obj, obj_id) for obj_id, obj in enumerate(self.objs))
        ##########################################################

        self.scenes = [f"{i:06d}" for i in data_cfg["scenes"]]
        self.debug_im_id = data_cfg.get("debug_im_id", None)
        self.max_im = data_cfg.get("max_im", None)

        cam_path = osp.join(DATASETS_ROOT, 'BOP_DATASETS/denstereo/camera.json')
        with open(cam_path, 'r') as cam_file:
            self.baseline = np.array(json.load(cam_file)['baseline'], dtype=np.float32)

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
                    self.name, self.dataset_root_l, self.with_masks, self.with_depth, __name__
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
            scene_id = int(scene)
            scene_root_l = osp.join(self.dataset_root_l, scene)
            scene_root_r = osp.join(self.dataset_root_r, scene)

            gt_dict_l = mmcv.load(osp.join(scene_root_l, "scene_gt.json"))
            gt_dict_r = mmcv.load(osp.join(scene_root_r, "scene_gt.json"))
            gt_info_dict_l = mmcv.load(osp.join(scene_root_l, "scene_gt_info.json"))
            gt_info_dict_r = mmcv.load(osp.join(scene_root_r, "scene_gt_info.json"))
            cam_dict_l = mmcv.load(osp.join(scene_root_l, "scene_camera.json"))
            cam_dict_r = mmcv.load(osp.join(scene_root_r, "scene_camera.json"))

            for str_im_id in tqdm(gt_dict_l, postfix=f"{scene_id}"):
                int_im_id = int(str_im_id)
                if self.max_im is not None:
                    if self.max_im < int_im_id:
                        continue

                rgb_path_l = osp.join(scene_root_l, "rgb/{:06d}.jpg").format(int_im_id)
                rgb_path_r = osp.join(scene_root_r, "rgb/{:06d}.jpg").format(int_im_id)
                assert osp.exists(rgb_path_l), rgb_path_l
                assert osp.exists(rgb_path_r), rgb_path_r

                depth_path_l = osp.join(scene_root_l, "depth/{:06d}.png".format(int_im_id))
                depth_path_r = osp.join(scene_root_r, "depth/{:06d}.png".format(int_im_id))

                scene_im_id = f"{scene_id}/{int_im_id}"

                K = np.array(cam_dict_l[str_im_id]["cam_K"], dtype=np.float32).reshape(3, 3)
                depth_factor = 1000.0 / cam_dict_l[str_im_id]["depth_scale"]  # 10000

                record = {
                    "dataset_name": self.name,
                    "file_name_l": rgb_path_l,
                    "file_name_r": rgb_path_r,
                    "depth_file_l": depth_path_l,
                    "depth_file_r": depth_path_r,
                    "height": self.height,
                    "width": self.width,
                    "image_id": int_im_id,
                    "scene_im_id": scene_im_id,  # for evaluation
                    "cam": K,
                    "baseline": self.baseline,
                    "depth_factor": depth_factor,
                    "img_type": "syn_pbr",  # NOTE: has background
                }
                insts = []
                for anno_i, anno_l in enumerate(gt_dict_l[str_im_id]):
                    anno_r = gt_dict_r[str_im_id][anno_i]
                    obj_id = anno_l["obj_id"]
                    if obj_id not in self.cat_ids:
                        continue

                    scene_im_anno_id = "{:d}/{:d}/{:d}".format(scene_id, int_im_id, obj_id)
                    if self.debug_im_id is not None:
                        if self.debug_im_id != scene_im_anno_id:
                            continue
                    
                    cur_label = self.cat2label[obj_id]  # 0-based label
                    R_l = np.array(anno_l["cam_R_m2c"], dtype="float32").reshape(3, 3)
                    R_r = np.array(anno_r["cam_R_m2c"], dtype="float32").reshape(3, 3)
                    t_l = np.array(anno_l["cam_t_m2c"], dtype="float32") / 1000.0
                    t_r = np.array(anno_r["cam_t_m2c"], dtype="float32") / 1000.0
                    pose_l = np.hstack([R_l, t_l.reshape(3, 1)])
                    pose_r = np.hstack([R_r, t_r.reshape(3, 1)])
                    quat_l = mat2quat(R_l).astype("float32")
                    quat_r = mat2quat(R_r).astype("float32")

                    proj_l = (record["cam"] @ t_l.T).T
                    proj_r = (record["cam"] @ t_r.T).T
                    proj_l = proj_l[:2] / proj_l[2]
                    proj_r = proj_r[:2] / proj_r[2]

                    # bbox_visib_l = gt_info_dict_l[str_im_id][anno_i]["bbox_visib"]
                    try:
                        bbox_visib_l = gt_info_dict_l[str_im_id][anno_i]["bbox_visib"]
                    except IndexError as e:
                        raise IndexError('list index out of range: scene:{} str_im_id:{} anno_i:{}'.format(scene, str_im_id, anno_i)) 
                    bbox_visib_r = gt_info_dict_r[str_im_id][anno_i]["bbox_visib"]
                    bbox_obj_l = gt_info_dict_l[str_im_id][anno_i]["bbox_obj"]
                    bbox_obj_r = gt_info_dict_r[str_im_id][anno_i]["bbox_obj"]
                    x1, y1, w, h = bbox_visib_l
                    if self.filter_invalid:
                        if h <= 1 or w <= 1:
                            self.num_instances_without_valid_box += 1
                            continue
                    x1, y1, w, h = bbox_visib_r
                    if self.filter_invalid:
                        if h <= 1 or w <= 1:
                            self.num_instances_without_valid_box += 1
                            continue

                    mask_file_l = osp.join(scene_root_l, "mask/{:06d}_{:06d}.png".format(int_im_id, anno_i))
                    mask_file_r = osp.join(scene_root_r, "mask/{:06d}_{:06d}.png".format(int_im_id, anno_i))
                    mask_visib_file_l = osp.join(scene_root_l, "mask_visib/{:06d}_{:06d}.png".format(int_im_id, anno_i))
                    mask_visib_file_r = osp.join(scene_root_r, "mask_visib/{:06d}_{:06d}.png".format(int_im_id, anno_i))
                    assert osp.exists(mask_file_l), mask_file_l
                    assert osp.exists(mask_file_r), mask_file_r
                    assert osp.exists(mask_visib_file_l), mask_visib_file_l
                    assert osp.exists(mask_visib_file_r), mask_visib_file_r
                    # load mask visib  TODO: load both mask_visib and mask_full
                    mask_single_l = mmcv.imread(mask_visib_file_l, "unchanged")
                    mask_single_r = mmcv.imread(mask_visib_file_r, "unchanged")
                    mask_single_l = mask_single_l.astype(np.bool).astype(np.int)
                    mask_single_r = mask_single_r.astype(np.bool).astype(np.int)
                    mask_single_erode_l = scin.binary_erosion(mask_single_l.astype(np.int))
                    mask_single_erode_r = scin.binary_erosion(mask_single_r.astype(np.int))
                    area_l = mask_single_erode_l.sum()
                    area_r = mask_single_erode_r.sum()
                    if (area_l <= 64) or (area_r <= 64):  # filter out too small or nearly invisible instances
                        self.num_instances_without_valid_segmentation += 1
                        continue

                    visib_fract_l = gt_info_dict_l[str_im_id][anno_i].get("visib_fract", 1.0)
                    visib_fract_r = gt_info_dict_r[str_im_id][anno_i].get("visib_fract", 1.0)

                    mask_rle_l = binary_mask_to_rle(mask_single_l, compressed=True)
                    mask_rle_r = binary_mask_to_rle(mask_single_r, compressed=True)

                    xyz_path_l = osp.join(self.xyz_root_l, f"{scene_id:06d}/{int_im_id:06d}_{anno_i:06d}-xyz.npz")
                    xyz_path_r = osp.join(self.xyz_root_r, f"{scene_id:06d}/{int_im_id:06d}_{anno_i:06d}-xyz.npz")
                    assert osp.exists(xyz_path_l), xyz_path_l
                    assert osp.exists(xyz_path_r), xyz_path_r
                    occ_path_l = osp.join(self.occ_root_l, f"{scene_id:06d}/{int_im_id:06d}_{anno_i:06d}-Q0.npz")
                    occ_path_r = osp.join(self.occ_root_r, f"{scene_id:06d}/{int_im_id:06d}_{anno_i:06d}-Q0.npz")
                    inst = {
                        "category_id": cur_label,  # 0-based label
                        "bbox_l": bbox_visib_l,  # TODO: load both bbox_obj and bbox_visib
                        "bbox_r": bbox_visib_r,  # TODO: load both bbox_obj and bbox_visib
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "pose_l": pose_l,
                        "pose_r": pose_r,
                        "quat_l": quat_l,
                        "quat_r": quat_r,
                        "trans_l": t_l,
                        "trans_r": t_r,
                        "centroid_2d_l": proj_l,  # absolute (cx, cy)
                        "centroid_2d_r": proj_r,  # absolute (cx, cy)
                        "segmentation_l": mask_rle_l,
                        "segmentation_r": mask_rle_r,
                        "mask_full_file_l": mask_file_l,  # TODO: load as mask_full, rle
                        "mask_full_file_r": mask_file_r,  # TODO: load as mask_full, rle
                        "visib_fract": (visib_fract_l + visib_fract_r) / 2,
                        "xyz_path_l": xyz_path_l,
                        "xyz_path_r": xyz_path_r,
                        "occ_path_l": occ_path_l,
                        "occ_path_r": occ_path_r,
                    }

                    model_info = self.models_info[str(obj_id)]
                    inst["model_info"] = model_info
                    # TODO: using full mask and full xyz
                    for key in ["bbox3d_and_center"]:
                        inst[key] = self.models[cur_label][key]
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
                osp.join(self.models_root, f"obj_{ref.denstereo.obj2id[obj_name]:06d}.ply"), vertex_scale=self.scale_to_meter
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


def get_denstereo_metadata(obj_names, ref_key):
    """task specific metadata."""
    data_ref = ref.__dict__[ref_key]

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

    meta = {"thing_classes": obj_names, "sym_infos": cur_sym_infos}
    return meta


denstereo_model_root = "BOP_DATASETS/denstereo/models/"
################################################################################


SPLITS_DENSTEREO_PBR = dict(
    denstereo_train_pbr =dict(
        name="denstereo_train_pbr",
        objs=ref.denstereo.objects,  # selected objects
        dataset_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/train_pbr_left"),
        dataset_root_right=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/train_pbr_right"),
        models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/models"),
        xyz_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/train_pbr_left/xyz_crop_hd"),
        xyz_root_right=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/train_pbr_right/xyz_crop_hd"),
        occ_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/train_pbr_left/Q0"),
        occ_root_right=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/train_pbr_right/Q0"),
        scale_to_meter=0.001,
        with_masks=True,  # (load masks but may not use it)
        with_depth=True,  # (load depth path here, but may not use it)
        height=480,
        width=640,
        use_cache=True,
        num_to_load=-1,
        filter_invalid=True,
        scenes=ref.denstereo.train_pbr_scenes,
        ref_key="denstereo",
    ),
    denstereo_train_pbr_left =dict(
        name="denstereo_train_pbr_left",
        objs=ref.denstereo.objects,  # selected objects
        dataset_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/train_pbr_left"),
        models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/models"),
        xyz_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/train_pbr_left/xyz_crop_hd"),
        occ_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/train_pbr_left/Q0"),
        scale_to_meter=0.001,
        with_masks=True,  # (load masks but may not use it)
        with_depth=True,  # (load depth path here, but may not use it)
        height=480,
        width=640,
        use_cache=True,
        num_to_load=-1,
        filter_invalid=True,
        scenes=ref.denstereo.train_pbr_scenes,
        ref_key="denstereo",
    ),
    denstereo_train_pbr_right =dict(
        name="denstereo_train_pbr_right",
        objs=ref.denstereo.objects,  # selected objects
        dataset_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/train_pbr_right"),
        models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/models"),
        xyz_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/train_pbr_right/xyz_crop_hd"),
        occ_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/train_pbr_right/Q0"),
        scale_to_meter=0.001,
        with_masks=True,  # (load masks but may not use it)
        with_depth=True,  # (load depth path here, but may not use it)
        height=480,
        width=640,
        use_cache=True,
        num_to_load=-1,
        filter_invalid=True,
        scenes=ref.denstereo.train_pbr_scenes,
        ref_key="denstereo",
    ),
    denstereo_test_pbr =dict(
        name="denstereo_test_pbr",
        objs=ref.denstereo.objects,  # selected objects
        dataset_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/train_pbr_left"),
        dataset_root_right=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/train_pbr_right"),
        models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/models"),
        xyz_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/train_pbr_left/xyz_crop_hd"),
        xyz_root_right=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/train_pbr_right/xyz_crop_hd"),
        occ_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/train_pbr_left/Q0"),
        occ_root_right=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/train_pbr_right/Q0"),
        scale_to_meter=0.001,
        with_masks=True,  # (load masks but may not use it)
        with_depth=True,  # (load depth path here, but may not use it)
        height=480,
        width=640,
        use_cache=True,
        num_to_load=-1,
        filter_invalid=True,
        scenes=ref.denstereo.test_pbr_scenes,
        ref_key="denstereo",
    ),
    denstereo_test_pbr_left =dict(
        name="denstereo_test_pbr_left",
        objs=ref.denstereo.objects,  # selected objects
        dataset_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/train_pbr_left"),
        models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/models"),
        xyz_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/train_pbr_left/xyz_crop_hd"),
        occ_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/train_pbr_left/Q0"),
        scale_to_meter=0.001,
        with_masks=True,  # (load masks but may not use it)
        with_depth=True,  # (load depth path here, but may not use it)
        height=480,
        width=640,
        use_cache=True,
        num_to_load=-1,
        filter_invalid=True,
        scenes=ref.denstereo.test_pbr_scenes,
        ref_key="denstereo",
    ),
    denstereo_test_pbr_right =dict(
        name="denstereo_test_pbr_right",
        objs=ref.denstereo.objects,  # selected objects
        dataset_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/train_pbr_right"),
        models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/models"),
        xyz_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/train_pbr_right/xyz_crop_hd"),
        occ_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/train_pbr_right/Q0"),
        scale_to_meter=0.001,
        with_masks=True,  # (load masks but may not use it)
        with_depth=True,  # (load depth path here, but may not use it)
        height=480,
        width=640,
        use_cache=True,
        num_to_load=-1,
        filter_invalid=True,
        scenes=ref.denstereo.test_pbr_scenes,
        ref_key="denstereo",
    ),
    denstereo_debug_train_pbr =dict(
        name="denstereo_debug_train_pbr",
        objs=[ref.denstereo.objects[0]],  # selected objects
        dataset_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/train_pbr_left"),
        dataset_root_right=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/train_pbr_right"),
        models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/models"),
        xyz_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/train_pbr_left/xyz_crop_hd"),
        xyz_root_right=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/train_pbr_right/xyz_crop_hd"),
        occ_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/train_pbr_left/Q0"),
        occ_root_right=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/train_pbr_right/Q0"),
        scale_to_meter=0.001,
        with_masks=True,  # (load masks but may not use it)
        with_depth=True,  # (load depth path here, but may not use it)
        height=480,
        width=640,
        use_cache=True,
        num_to_load=-1,
        filter_invalid=True,
        scenes=ref.denstereo.debug_pbr_scenes,
        max_im = 100,
        ref_key="denstereo",
    ),
    denstereo_debug_test_pbr =dict(
        name="denstereo_debug_test_pbr",
        objs=[ref.denstereo.objects[0]],  # selected objects
        dataset_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/train_pbr_left"),
        dataset_root_right=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/train_pbr_right"),
        models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/models"),
        xyz_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/train_pbr_left/xyz_crop_hd"),
        xyz_root_right=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/train_pbr_right/xyz_crop_hd"),
        occ_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/train_pbr_left/Q0"),
        occ_root_right=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/train_pbr_right/Q0"),
        scale_to_meter=0.001,
        with_masks=True,  # (load masks but may not use it)
        with_depth=True,  # (load depth path here, but may not use it)
        height=480,
        width=640,
        use_cache=True,
        num_to_load=-1,
        filter_invalid=True,
        scenes=ref.denstereo.debug_test_pbr_scenes,
        max_im = 100,
        ref_key="denstereo",
    ),
    denstereo_debug_train_pbr_left =dict(
        name="denstereo_debug_train_pbr_left",
        # objs=[ref.denstereo.objects[1]],  # selected objects
        objs=ref.denstereo.objects,  # selected objects
        dataset_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/train_pbr_left"),
        models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/models"),
        xyz_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/train_pbr_left/xyz_crop_hd"),
        occ_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/train_pbr_left/Q0"),
        scale_to_meter=0.001,
        with_masks=True,  # (load masks but may not use it)
        with_depth=True,  # (load depth path here, but may not use it)
        height=480,
        width=640,
        use_cache=True,
        num_to_load=-1,
        filter_invalid=True,
        scenes=ref.denstereo.debug_pbr_scenes,
        max_im = 100,
        ref_key="denstereo",
    ),
)

# single obj splits
for obj in ref.denstereo.objects:
    for split in ["train_pbr"]:
        name = "denstereo_{}_{}".format(obj, split)
        if split in ["train_pbr"]:
            filter_invalid = True
        elif split in ["test_pbr"]:
            filter_invalid = True
        elif split in ["test"]:
            filter_invalid = False
        else:
            raise ValueError("{}".format(split))
        if name not in SPLITS_DENSTEREO_PBR:
            SPLITS_DENSTEREO_PBR[name] = dict(
                name=name,
                objs=[obj],  # only this obj
                dataset_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/train_pbr_left"),
                dataset_root_right=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/train_pbr_right"),
                models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/models"),
                xyz_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/train_pbr_left/xyz_crop_hd"),
                xyz_root_right=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/train_pbr_right/xyz_crop_hd"),
                scale_to_meter=0.001,
                with_masks=True,  # (load masks but may not use it)
                with_depth=True,  # (load depth path here, but may not use it)
                height=480,
                width=640,
                use_cache=True,
                num_to_load=-1,
                filter_invalid=filter_invalid,
                scenes=ref.denstereo.train_pbr_scenes,
                ref_key="denstereo",
            )

for obj in ref.denstereo.objects:
    for split in ["test_pbr"]:
        name = "denstereo_{}_{}".format(obj, split)
        if split in ["train_pbr"]:
            filter_invalid = True
        elif split in ["test_pbr"]:
            filter_invalid = True
        elif split in ["test"]:
            filter_invalid = False
        else:
            raise ValueError("{}".format(split))
        if name not in SPLITS_DENSTEREO_PBR:
            SPLITS_DENSTEREO_PBR[name] = dict(
                name=name,
                objs=[obj],  # only this obj
                dataset_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/train_pbr_left"),
                dataset_root_right=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/train_pbr_right"),
                models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/models"),
                xyz_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/train_pbr_left/xyz_crop_hd"),
                xyz_root_right=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/train_pbr_right/xyz_crop_hd"),
                scale_to_meter=0.001,
                with_masks=True,  # (load masks but may not use it)
                with_depth=True,  # (load depth path here, but may not use it)
                height=480,
                width=640,
                use_cache=True,
                num_to_load=-1,
                filter_invalid=filter_invalid,
                scenes=ref.denstereo.test_pbr_scenes,
                ref_key="denstereo",
            )

# ================ add single image dataset for debug =======================================
for split in ['train_pbr_left']: # TODO add train _right 'train_pbr_right']:
    for scene in ref.denstereo.debug_pbr_scenes:
        scene_root = osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/{:s}/{:06d}".format(
                split,
                scene
            )
        )
        gt_dict = mmcv.load(osp.join(scene_root, "scene_gt.json"))
        for im_id in gt_dict.keys():
            int_im_id = int(im_id)
            obj_ids = [pose['obj_id'] for pose in gt_dict[im_id]]
            for obj_id in obj_ids:
                name = "denstereo_single_{}_{}_{}_{}".format(scene, int_im_id, obj_id, split)
                if name not in SPLITS_DENSTEREO_PBR:
                    scene_image_obj_id = "{:d}/{:d}/{:d}".format(scene, int_im_id, obj_id)
                    SPLITS_DENSTEREO_PBR[name] = dict(
                        name=name,
                        dataset_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/{:s}".format(split)),
                        models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/models"),
                        objs=ref.denstereo.objects,  # only this obj
                        image_prefixes=[
                            osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/{:s}/{:06d}").format(split, scene)
                        ],
                        xyz_prefixes=[
                            osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/{:s}/xyz_crop_hd/{:06d}".format(split, scene))
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
                        scenes = ref.denstereo.debug_pbr_scenes,
                        ref_key="denstereo",
                    )

# single image stereo

for scene in ref.denstereo.debug_pbr_scenes:
    scene_root = osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/{:s}/{:06d}".format(
            'train_pbr_left',
            scene
        )
    )
    gt_dict = mmcv.load(osp.join(scene_root, "scene_gt.json"))
    for im_id in gt_dict.keys():
        int_im_id = int(im_id)
        obj_ids = [pose['obj_id'] for pose in gt_dict[im_id]]
        for obj_id in obj_ids:
            name = "denstereo_single_{}_{}_{}_stereo".format(scene, int_im_id, obj_id)
            if name not in SPLITS_DENSTEREO_PBR:
                scene_image_obj_id = "{:d}/{:d}/{:d}".format(scene, int_im_id, obj_id)
                SPLITS_DENSTEREO_PBR[name] = dict(
                    name=name,
                    dataset_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/train_pbr_left"),
                    dataset_root_right=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/train_pbr_right"),
                    models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/models"),
                    objs=ref.denstereo.objects,  # only this obj
                    image_prefixes=[
                        osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/train_pbr_left/{:06d}").format(scene)
                    ],
                    image_prefixes_right=[
                        osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/train_pbr_right/{:06d}").format(scene)
                    ],
                    xyz_prefixes=[
                        osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/train_pbr_left/xyz_crop_hd/{:06d}".format(scene))
                    ],
                    xyz_prefixes_right=[
                        osp.join(DATASETS_ROOT, "BOP_DATASETS/denstereo/train_pbr_right/xyz_crop_hd/{:06d}".format(scene))
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
                    scenes = ref.denstereo.debug_pbr_scenes,
                    ref_key="denstereo",
                )

def register_with_name_cfg(name, data_cfg=None):
    """Assume pre-defined datasets live in `./datasets`.

    Args:
        name: datasnet_name,
        data_cfg: if name is in existing SPLITS, use pre-defined data_cfg
            otherwise requires data_cfg
            data_cfg can be set in cfg.DATA_CFG.name
    """
    dprint("register dataset: {}".format(name))
    if name in SPLITS_DENSTEREO_PBR:
        used_cfg = SPLITS_DENSTEREO_PBR[name]
    else:
        assert data_cfg is not None, f"dataset name {name} is not registered"
        used_cfg = data_cfg
    DatasetCatalog.register(name, DENSTEREO_PBR_Dataset(used_cfg))
    # something like eval_types
    MetadataCatalog.get(name).set(
        id="denstereo",  # NOTE: for pvnet to determine module
        ref_key=used_cfg["ref_key"],
        objs=used_cfg["objs"],
        eval_error_types=["ad", "rete", "proj"],
        evaluator_type="bop",
        **get_denstereo_metadata(obj_names=used_cfg["objs"], ref_key=used_cfg["ref_key"]),
    )


def get_available_datasets():
    return list(SPLITS_DENSTEREO_PBR.keys())


#### tests ###############################################
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
