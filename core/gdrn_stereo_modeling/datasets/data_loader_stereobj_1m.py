# -*- coding: utf-8 -*-
import copy
import logging
import os.path as osp
import pickle

import cv2
from PIL import Image
import numpy as np
import ref
import torch
from core.gdrn_stereo_modeling.datasets.base_data_loader import Base_DatasetFromList
from core.gdrn_stereo_modeling.tools.data_utils import crop_resize_by_warp_affine, get_2d_coord_np, read_image_mmcv, xyz_to_region
from core.gdrn_stereo_modeling.tools.dataset_utils import (
    filter_empty_dets,
    filter_invalid_in_dataset_dicts,
    flat_dataset_dicts,
    load_detections_into_dataset,
    my_build_batch_data_loader,
    trivial_batch_collator,
)
from core.gdrn_stereo_modeling.tools.my_distributed_sampler import InferenceSampler, RepeatFactorTrainingSampler, TrainingSampler
from core.gdrn_stereo_modeling.tools.ssd_color_transform import ColorAugSSDTransform
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import get_detection_dataset_dicts
from detectron2.data import transforms as T
from detectron2.structures import BoxMode
from detectron2.utils.logger import log_first_n
from lib.pysixd import inout, misc
from lib.utils.mask_utils import cocosegm2mask, get_edge, rle2mask

from .dataset_factory import register_datasets
from .data_loader_online import GDRN_Online_DatasetFromList
import scipy.ndimage as scin
logger = logging.getLogger(__name__)


def transform_instance_annotations(annotation, transforms, image_size, *, keypoint_hflip_indices=None):
    """
    NOTE: Adapted from detection_utils.
    Apply transforms to box, segmentation, keypoints, etc. of annotations of a single instance.

    It will use `transforms.apply_box` for the box, and
    `transforms.apply_coords` for segmentation polygons & keypoints.
    If you need anything more specially designed for each data structure,
    you'll need to implement your own version of this function or the transforms.

    Args:
        annotation (dict): dict of instance annotations for a single instance.
        transforms (TransformList):
        image_size (tuple): the height, width of the transformed image
        keypoint_hflip_indices (ndarray[int]): see `create_keypoint_hflip_indices`.

    Returns:
        dict:
            the same input dict with fields "bbox", "segmentation", "keypoints"
            transformed according to `transforms`.
            The "bbox_mode" field will be set to XYXY_ABS.
    """
    im_H, im_W = image_size
    bbox_l = BoxMode.convert(annotation["bbox_l"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
    bbox_r = BoxMode.convert(annotation["bbox_r"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
    bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
    # Note that bbox is 1d (per-instance bounding box)
    annotation["bbox_l"] = np.array(transforms.apply_box([bbox_l])[0])
    annotation["bbox_r"] = np.array(transforms.apply_box([bbox_r])[0])
    annotation["bbox"] = np.array(transforms.apply_box([bbox])[0])
    annotation["bbox_mode"] = BoxMode.XYXY_ABS

    if "segmentation_l" in annotation:
        # NOTE: here we transform segms to binary masks (interp is nearest by default)
        mask_l = transforms.apply_segmentation(cocosegm2mask(annotation["segmentation_l"], h=im_H, w=im_W))
        mask_r = transforms.apply_segmentation(cocosegm2mask(annotation["segmentation_r"], h=im_H, w=im_W))
        annotation["segmentation_l"] = mask_l
        annotation["segmentation_r"] = mask_r

    if "keypoints_l" in annotation:
        keypoints_l = utils.transform_keypoint_annotations(
            annotation["keypoints_l"], transforms, image_size, keypoint_hflip_indices
        )
        keypoints_r = utils.transform_keypoint_annotations(
            annotation["keypoints_r"], transforms, image_size, keypoint_hflip_indices
        )
        annotation["keypoints_l"] = keypoints_l
        annotation["keypoints_r"] = keypoints_r

    if "centroid_2d" in annotation:
        annotation["centroid_2d"] = transforms.apply_coords(
            np.array(annotation["centroid_2d"]).reshape(1, 2)
        ).flatten()

    return annotation


def build_gdrn_augmentation(cfg, is_train):
    """Create a list of :class:`Augmentation` from config. when training 6d
    pose, cannot flip.

    Returns:
        list[Augmentation]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(len(min_size))

    augmentation = []
    augmentation.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
    if is_train:
        # augmentation.append(T.RandomFlip())
        logger.info("Augmentations used in training: " + str(augmentation))
    return augmentation


class GDRN_Stereobj_1m_DatasetFromList(Base_DatasetFromList):
    """NOTE: we can also use the default DatasetFromList and
    implement a similar custom DataMapper,
    but it is harder to implement some features relying on other dataset dicts
    # https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/common.py
    Wrap a list to a torch Dataset. It produces elements of the list as data.
    """

    def __init__(self, cfg, split, lst: list, copy: bool = True, serialize: bool = True, flatten=True):
        """
        Args:
            lst (list): a list which contains elements to produce.
            copy (bool): whether to deepcopy the element when producing it,
                so that the result can be modified in place without affecting the
                source in the list.
            serialize (bool): whether to hold memory using serialized objects, when
                enabled, data loader workers can use shared RAM from master
                process instead of making a copy.
        """
        self.augmentation = build_gdrn_augmentation(cfg, is_train=(split == "train"))
        if cfg.INPUT.COLOR_AUG_PROB > 0 and cfg.INPUT.COLOR_AUG_TYPE.lower() == "ssd":
            self.augmentation.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
            logging.getLogger(__name__).info("Color augmentation used in training: " + str(self.augmentation[-1]))
        # fmt: off
        self.img_format = cfg.INPUT.FORMAT  # default BGR
        self.with_depth = cfg.INPUT.WITH_DEPTH
        self.aug_depth = cfg.INPUT.AUG_DEPTH
        # NOTE: color augmentation config
        self.color_aug_prob = cfg.INPUT.COLOR_AUG_PROB
        self.color_aug_type = cfg.INPUT.COLOR_AUG_TYPE
        self.color_aug_code = cfg.INPUT.COLOR_AUG_CODE
        # fmt: on
        self.cfg = cfg
        self.split = split  # train | val | test
        if split == "train" and self.color_aug_prob > 0:
            self.color_augmentor = self._get_color_augmentor(aug_type=self.color_aug_type, aug_code=self.color_aug_code)
        else:
            self.color_augmentor = None
        # ------------------------
        # common model infos
        self.fps_points = {}
        self.model_points = {}
        self.extents = {}
        self.sym_infos = {}
        # ----------------------------------------------------
        self.flatten = flatten
        self._lst = flat_dataset_dicts(lst) if flatten else lst
        # ----------------------------------------------------
        self._copy = copy
        self._serialize = serialize

        def _serialize(data):
            buffer = pickle.dumps(data, protocol=-1)
            return np.frombuffer(buffer, dtype=np.uint8)

        if self._serialize:
            logger.info("Serializing {} elements to byte tensors and concatenating them all ...".format(len(self._lst)))
            self._lst = [_serialize(x) for x in self._lst]
            self._addr = np.asarray([len(x) for x in self._lst], dtype=np.int64)
            self._addr = np.cumsum(self._addr)
            self._lst = np.concatenate(self._lst)
            logger.info("Serialized dataset takes {:.2f} MiB".format(len(self._lst) / 1024 ** 2))

    def __len__(self):
        if self._serialize:
            return len(self._addr)
        else:
            return len(self._lst)

    def _get_fps_points(self, dataset_name, with_center=False):
        """convert to label based keys.

        # TODO: get models info similarly
        """
        if dataset_name in self.fps_points:
            return self.fps_points[dataset_name]

        dset_meta = MetadataCatalog.get(dataset_name)
        ref_key = dset_meta.ref_key
        data_ref = ref.__dict__[ref_key]
        objs = dset_meta.objs
        cfg = self.cfg
        num_fps_points = cfg.MODEL.POSE_NET.GEO_HEAD.NUM_REGIONS
        cur_fps_points = {}
        loaded_fps_points = data_ref.get_fps_points()
        for i, obj_name in enumerate(objs):
            obj_id = data_ref.obj2id[obj_name]
            if with_center:
                cur_fps_points[i] = loaded_fps_points[str(obj_id)][f"fps{num_fps_points}_and_center"]
            else:
                cur_fps_points[i] = loaded_fps_points[str(obj_id)][f"fps{num_fps_points}_and_center"][:-1]
        self.fps_points[dataset_name] = cur_fps_points
        return self.fps_points[dataset_name]

    def _get_model_points(self, dataset_name):
        """convert to label based keys."""
        if dataset_name in self.model_points:
            return self.model_points[dataset_name]

        dset_meta = MetadataCatalog.get(dataset_name)
        ref_key = dset_meta.ref_key
        data_ref = ref.__dict__[ref_key]
        objs = dset_meta.objs
        cfg = self.cfg

        cur_model_points = {}
        num = np.inf
        for i, obj_name in enumerate(objs):
            obj_id = data_ref.obj2id[obj_name]
            model_path = osp.join(data_ref.model_dir, f"obj_{obj_id:06d}.ply")
            model = inout.load_ply(model_path, vertex_scale=data_ref.vertex_scale)
            cur_model_points[i] = pts = model["pts"]
            if pts.shape[0] < num:
                num = pts.shape[0]

        num = min(num, cfg.MODEL.POSE_NET.LOSS_CFG.NUM_PM_POINTS)
        for i in range(len(cur_model_points)):
            keep_idx = np.arange(num)
            np.random.shuffle(keep_idx)  # random sampling
            cur_model_points[i] = cur_model_points[i][keep_idx, :]

        self.model_points[dataset_name] = cur_model_points
        return self.model_points[dataset_name]

    def _get_extents(self, dataset_name):
        """label based keys."""
        if dataset_name in self.extents:
            return self.extents[dataset_name]

        dset_meta = MetadataCatalog.get(dataset_name)
        try:
            ref_key = dset_meta.ref_key
        except:
            # FIXME: for some reason, in distributed training, this need to be re-registered
            register_datasets([dataset_name])
            dset_meta = MetadataCatalog.get(dataset_name)
            ref_key = dset_meta.ref_key

        data_ref = ref.__dict__[ref_key]
        objs = dset_meta.objs
        cfg = self.cfg

        cur_extents = {}
        for i, obj_name in enumerate(objs):
            obj_id = data_ref.obj2id[obj_name]
            model_path = osp.join(data_ref.model_dir, f"obj_{obj_id:06d}.ply")
            model = inout.load_ply(model_path, vertex_scale=data_ref.vertex_scale)
            pts = model["pts"]
            xmin, xmax = np.amin(pts[:, 0]), np.amax(pts[:, 0])
            ymin, ymax = np.amin(pts[:, 1]), np.amax(pts[:, 1])
            zmin, zmax = np.amin(pts[:, 2]), np.amax(pts[:, 2])
            size_x = xmax - xmin
            size_y = ymax - ymin
            size_z = zmax - zmin
            cur_extents[i] = np.array([size_x, size_y, size_z], dtype="float32")

        self.extents[dataset_name] = cur_extents
        return self.extents[dataset_name]

    def _get_sym_infos(self, dataset_name):
        """label based keys."""
        if dataset_name in self.sym_infos:
            return self.sym_infos[dataset_name]

        dset_meta = MetadataCatalog.get(dataset_name)
        ref_key = dset_meta.ref_key
        data_ref = ref.__dict__[ref_key]
        objs = dset_meta.objs
        cfg = self.cfg

        cur_sym_infos = {}
        loaded_models_info = data_ref.get_models_info()
        for i, obj_name in enumerate(objs):
            obj_id = data_ref.obj2id[obj_name]
            model_info = loaded_models_info[str(obj_id)]
            if "symmetries_discrete" in model_info or "symmetries_continuous" in model_info:
                sym_transforms = misc.get_symmetry_transformations(model_info, max_sym_disc_step=0.01)
                sym_info = np.array([sym["R"] for sym in sym_transforms], dtype=np.float32)
            else:
                sym_info = None
            cur_sym_infos[i] = sym_info

        self.sym_infos[dataset_name] = cur_sym_infos
        return self.sym_infos[dataset_name]

    def read_data(self, dataset_dict):
        """load image and annos random shift & scale bbox; crop, rescale."""
        cfg = self.cfg
        net_cfg = cfg.MODEL.POSE_NET
        g_head_cfg = net_cfg.GEO_HEAD
        loss_cfg = net_cfg.LOSS_CFG

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        dataset_name = dataset_dict["dataset_name"]

        inp = Image.open(dataset_dict["file_name"])
        inp = np.asarray(inp)
        _, w, _ = inp.shape
        image_l = inp[:, :w//2]
        image_r = inp[:, w//2:]

        utils.check_image_size(dataset_dict, image_l)
        utils.check_image_size(dataset_dict, image_r)

        im_H_ori, im_W_ori = image_l.shape[:2]

        # currently only replace bg for train ###############################
        if self.split == "train":
            # some synthetic data already has bg, img_type should be real or something else but not syn
            img_type = dataset_dict.get("img_type", "real")
            if img_type == "syn":
                log_first_n(logging.WARNING, "replace bg", n=10)
                assert "segmentation_l" in dataset_dict["inst_infos"]
                assert "segmentation_r" in dataset_dict["inst_infos"]
                mask_l = cocosegm2mask(dataset_dict["inst_infos"]["segmentation_l"], im_H_ori, im_W_ori)
                mask_r = cocosegm2mask(dataset_dict["inst_infos"]["segmentation_r"], im_H_ori, im_W_ori)
                image_l, mask_trunc_l = self.replace_bg(
                    image_l.copy(), mask_l, return_mask=True, truncate_fg=cfg.INPUT.get("TRUNCATE_FG", False)
                )
                image_r, mask_trunc_r = self.replace_bg(
                    image_r.copy(), mask_r, return_mask=True, truncate_fg=cfg.INPUT.get("TRUNCATE_FG", False)
                )
            else:  # real image
                if np.random.rand() < cfg.INPUT.CHANGE_BG_PROB:
                    log_first_n(logging.WARNING, "replace bg for real", n=10)
                    assert "segmentation_l" in dataset_dict["inst_infos"]
                    assert "segmentation_r" in dataset_dict["inst_infos"]
                    mask_l = cocosegm2mask(dataset_dict["inst_infos"]["segmentation_l"], im_H_ori, im_W_ori)
                    mask_r = cocosegm2mask(dataset_dict["inst_infos"]["segmentation_r"], im_H_ori, im_W_ori)
                    image_l, mask_trunc_l = self.replace_bg(
                        image_l.copy(), mask_l, return_mask=True, truncate_fg=cfg.INPUT.get("TRUNCATE_FG", False)
                    )
                    image_r, mask_trunc_r = self.replace_bg(
                        image_r.copy(), mask_r, return_mask=True, truncate_fg=cfg.INPUT.get("TRUNCATE_FG", False)
                    )
                else:
                    mask_trunc_l = None
                    mask_trunc_r = None

        # NOTE: maybe add or change color augment here ===================================
        if self.split == "train" and self.color_aug_prob > 0 and self.color_augmentor is not None:
            if np.random.rand() < self.color_aug_prob:
                if cfg.INPUT.COLOR_AUG_SYN_ONLY and img_type not in ["real"]:
                    image_l = self._color_aug(image_l, self.color_aug_type)
                    image_r = self._color_aug(image_r, self.color_aug_type)
                else:
                    image_l = self._color_aug(image_l, self.color_aug_type)
                    image_r = self._color_aug(image_r, self.color_aug_type)

        # other transforms (mainly geometric ones);
        # for 6d pose task, flip is not allowed in general except for some 2d keypoints methods
        image_l, transforms = T.apply_augmentations(self.augmentation, image_l)
        image_r, transforms = T.apply_augmentations(self.augmentation, image_r)
        im_H, im_W = image_shape = image_l.shape[:2] # h, w

        # NOTE: scale camera intrinsic if necessary ================================
        scale_x = im_W / im_W_ori
        scale_y = im_H / im_H_ori  # NOTE: generally scale_x should be equal to scale_y
        if "cam" in dataset_dict:
            if im_W != im_W_ori or im_H != im_H_ori:
                dataset_dict["cam"][0] *= scale_x
                dataset_dict["cam"][1] *= scale_y
            K = dataset_dict["cam"].astype("float32")
            dataset_dict["cam"] = torch.as_tensor(K)

        input_res = cfg.MODEL.POSE_NET.INPUT_RES
        out_res = cfg.MODEL.POSE_NET.OUTPUT_RES

        # CHW -> HWC
        coord_2d_l = get_2d_coord_np(im_W, im_H, low=0, high=1).transpose(1, 2, 0)
        coord_2d_r = get_2d_coord_np(im_W, im_H, low=0, high=1).transpose(1, 2, 0)

        #################################################################################
        if self.split != "train":
            # don't load annotations at test time
            test_bbox_type = cfg.TEST.TEST_BBOX_TYPE
            if test_bbox_type == "gt":
                bbox_key = "bbox"
            else:
                bbox_key = f"bbox_{test_bbox_type}"
            assert not self.flatten, "Do not use flattened dicts for test!"
            # here get batched rois
            roi_infos = {}
            # yapf: disable
            roi_keys = [
                "scene_im_id",
                "file_name_l",
                "file_name_r",
                "cam",
                "baseline",
                "im_H",
                "im_W",
                "roi_img",
                "inst_id",
                "roi_coord_2d",
                "roi_cls",
                "score",
                "roi_extent",
                bbox_key,
                "bbox_mode",
                "bbox_center",
                "roi_wh",
                "scale",
                "resize_ratio",
                "model_info",
            ]
            for _key in roi_keys:
                roi_infos[_key] = []
            # yapf: enable
            # TODO: how to handle image without detections
            #   filter those when load annotations or detections, implement a function for this
            # "annotations" means detections
            for inst_i, inst_infos in enumerate(dataset_dict["annotations"]):
                # inherent image-level infos
                roi_infos["scene_im_id"].append(dataset_dict["scene_im_id"])
                roi_infos["file_name_l"].append(dataset_dict["file_name_l"])
                roi_infos["file_name_r"].append(dataset_dict["file_name_r"])
                roi_infos["im_H"].append(im_H)
                roi_infos["im_W"].append(im_W)
                roi_infos["cam"].append(dataset_dict["cam"].cpu().numpy())
                roi_infos["baseline"].append(dataset_dict["baseline"])

                # roi-level infos
                roi_infos["inst_id"].append(inst_i)
                roi_infos["model_info"].append(inst_infos["model_info"])

                roi_cls = inst_infos["category_id"]
                roi_infos["roi_cls"].append(roi_cls)
                roi_infos["score"].append(inst_infos["score"])

                # extent
                roi_extent = self._get_extents(dataset_name)[roi_cls]
                roi_infos["roi_extent"].append(roi_extent)

                bbox = BoxMode.convert(
                    inst_infos[bbox_key],
                    inst_infos["bbox_mode"],
                    BoxMode.XYXY_ABS
                )
                bbox = np.array(transforms.apply_box([bbox])[0])
                roi_infos[bbox_key].append(bbox)
                roi_infos["bbox_mode"].append(BoxMode.XYXY_ABS)

                x1, y1, x2, y2 = bbox
                bbox_center = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2)], dtype=np.float32)
                bw = max(x2 - x1, 1)
                bh = max(y2 - y1, 1)
                scale = max(bh, bw) * cfg.INPUT.DZI_PAD_SCALE
                scale = min(scale, max(im_H, im_W)) * 1.0

                roi_infos["bbox_center"].append(bbox_center)
                roi_infos["scale"].append(scale)
                roi_infos["roi_wh"].append(
                    np.array([bw, bh], dtype=np.float32)
                )
                roi_infos["resize_ratio"].append(out_res / scale)

                # CHW, float32 tensor
                # roi_image
                roi_img_l = crop_resize_by_warp_affine(
                    image_l, bbox_center, scale, input_res, interpolation=cv2.INTER_LINEAR
                ).transpose(2, 0, 1)
                roi_img_r = crop_resize_by_warp_affine(
                    image_r, bbox_center, scale, input_res, interpolation=cv2.INTER_LINEAR
                ).transpose(2, 0, 1)

                roi_img_l = self.normalize_image(cfg, roi_img_l)
                roi_img_r = self.normalize_image(cfg, roi_img_r)
                roi_infos["roi_img"].append(
                    np.stack([roi_img_l, roi_img_r], axis=0).astype("float32")
                )

                # roi_coord_2d
                roi_coord_2d_l = crop_resize_by_warp_affine(
                    coord_2d_l, bbox_center, scale, out_res, interpolation=cv2.INTER_LINEAR
                ).transpose(
                    2, 0, 1
                )  # HWC -> CHW
                roi_coord_2d_r = crop_resize_by_warp_affine(
                    coord_2d_r, bbox_center, scale, out_res, interpolation=cv2.INTER_LINEAR
                ).transpose(
                    2, 0, 1
                )  # HWC -> CHW
                roi_infos["roi_coord_2d"].append(
                    np.stack([roi_coord_2d_l, roi_coord_2d_r], axis=0).astype("float32")
                )

            for _key in roi_keys:
                if _key in ["roi_img", "roi_coord_2d"]:
                    dataset_dict[_key] = torch.as_tensor(roi_infos[_key]).contiguous()
                elif _key in ["model_info", "scene_im_id", "file_name_l", "file_name_r"]:  # "im_H", "im_W"
                    # can not convert to tensor
                    dataset_dict[_key] = roi_infos[_key]
                else:
                    dataset_dict[_key] = torch.tensor(roi_infos[_key])

            return dataset_dict
        #######################################################################################
        # NOTE: currently assume flattened dicts for train
        assert self.flatten, "Only support flattened dicts for train now"
        inst_infos = dataset_dict.pop("inst_infos")
        dataset_dict["roi_cls"] = roi_cls = inst_infos["category_id"]

        # extent
        roi_extent = self._get_extents(dataset_name)[roi_cls]
        dataset_dict["roi_extent"] = torch.tensor(roi_extent, dtype=torch.float32)

        # load xyz =======================================================
        xyz_path = inst_infos["xyz_path"]
        xyz_info = np.load(xyz_path, allow_pickle=True)['xyz'].item()
        xyz_info_l = xyz_info['left']
        xyz_info_r = xyz_info['right']
        # xyz_info_l = np.load(inst_infos["xyz_path_l"]) # also load npz files
        # xyz_info_r = np.load(inst_infos["xyz_path_r"]) # also load npz files

        x1, y1, x2, y2 = xyz_info_l["xyxy"]
        # float16 does not affect performance (classification/regresion)
        xyz_crop = xyz_info_l["xyz_crop"]
        xyz_l = np.zeros((im_H_ori, im_W_ori, 3), dtype=np.float32)

        xyz_l[y1 : y2 + 1, x1 : x2 + 1, :] = xyz_crop
        xyz_l = cv2.resize(xyz_l, (im_W, im_H), interpolation=cv2.INTER_AREA)

        x1, y1, x2, y2 = xyz_info_r["xyxy"]
        # float16 does not affect performance (classification/regresion)
        xyz_crop = xyz_info_r["xyz_crop"]
        xyz_r = np.zeros((im_H_ori, im_W_ori, 3), dtype=np.float32)
        xyz_r[y1 : y2 + 1, x1 : x2 + 1, :] = xyz_crop
        xyz_r = cv2.resize(xyz_r, (im_W, im_H), interpolation=cv2.INTER_AREA)

        # NOTE: full mask
        mask_obj_l = ((xyz_l[:, :, 0] != 0) | (xyz_l[:, :, 1] != 0) | (xyz_l[:, :, 2] != 0)).astype(np.bool).astype(np.float32)
        mask_obj_r = ((xyz_r[:, :, 0] != 0) | (xyz_r[:, :, 1] != 0) | (xyz_r[:, :, 2] != 0)).astype(np.bool).astype(np.float32)

        mask_obj_erode_l = scin.binary_erosion(mask_obj_l.astype(np.int)).astype(np.float32)
        mask_obj_erode_r = scin.binary_erosion(mask_obj_r.astype(np.int)).astype(np.float32)
        if cfg.INPUT.SMOOTH_XYZ:
            xyz_l = self.smooth_xyz(xyz_l)
            xyz_r = self.smooth_xyz(xyz_r)

        if cfg.TRAIN.VIS:
            xyz_l = self.smooth_xyz(xyz_l)
            xyz_r = self.smooth_xyz(xyz_r)

        # merge bboxes
        x1_l, y1_l, x2_l, y2_l = xyz_info_l['xyxy']
        x1_r, y1_r, x2_r, y2_r = xyz_info_r['xyxy']
        inst_infos["bbox"] = [min(x1_l, x1_r), min(y1_l, y1_r), max(x2_l, x2_r), max(y2_l, y2_r)]
        
        inst_infos["bbox_l"] = [x1_l, y1_l, x2_l, y2_l]
        inst_infos["bbox_r"] = [x1_r, y1_r, x2_r, y2_r]

        inst_infos["bbox_mode"] = BoxMode.XYXY_ABS

        # USER: Implement additional transformations if you have other types of data
        # inst_infos.pop("segmentation")  # NOTE: use mask from xyz
        anno = transform_instance_annotations(inst_infos, transforms, image_shape, keypoint_hflip_indices=None)

        # augment bbox ===================================================
        bbox_xyxy = anno["bbox"]
        bbox_center, scale = self.aug_bbox_DZI(cfg, bbox_xyxy, im_H, im_W)
        bh = max(bbox_xyxy[3] - bbox_xyxy[1], 1)
        bw = max(bbox_xyxy[2] - bbox_xyxy[0], 1)

        # CHW, float32 tensor
        ## roi_image ------------------------------------
        roi_img_l = crop_resize_by_warp_affine(
            image_l, bbox_center, scale, input_res, interpolation=cv2.INTER_LINEAR
        ).transpose(2, 0, 1)
        # import matplotlib.pyplot as plt
        # plt.imshow(image_l)
        # plt.savefig('image.png')
        # plt.imshow(roi_img_l.transpose(1,2,0))
        # plt.savefig('image.png')
        roi_img_r = crop_resize_by_warp_affine(
            image_r, bbox_center, scale, input_res, interpolation=cv2.INTER_LINEAR
        ).transpose(2, 0, 1)

        roi_img_l = self.normalize_image(cfg, roi_img_l)
        roi_img_r = self.normalize_image(cfg, roi_img_r)

        # roi_coord_2d ----------------------------------------------------
        roi_coord_2d_l = crop_resize_by_warp_affine(
            coord_2d_l, bbox_center, scale, out_res, interpolation=cv2.INTER_NEAREST
        ).transpose(2, 0, 1)
        roi_coord_2d_r = crop_resize_by_warp_affine(
            coord_2d_r, bbox_center, scale, out_res, interpolation=cv2.INTER_NEAREST
        ).transpose(2, 0, 1)

        ## roi_mask ---------------------------------------
        # (mask_trunc < mask_visib < mask_obj)
        mask_visib_l = anno["segmentation_l"].astype("float32") * mask_obj_l
        # mask_visib_r = rle2mask(anno["segmentation_r"], im_H, im_W).astype("float32") * mask_obj_l
        mask_visib_r = anno["segmentation_r"].astype("float32") * mask_obj_r # this was a bug in the original code #todo test result
        if mask_trunc_l is None:
            mask_trunc_l = mask_visib_l
            mask_trunc_r = mask_visib_r
        else:
            mask_trunc_l = mask_visib_l * mask_trunc_l.astype("float32")
            mask_trunc_r = mask_visib_r * mask_trunc_r.astype("float32")

        if cfg.TRAIN.VIS:
            mask_xyz_interp = cv2.INTER_LINEAR
        else:
            mask_xyz_interp = cv2.INTER_NEAREST

        # maybe truncated mask (true mask for rgb)
        roi_mask_trunc_l = crop_resize_by_warp_affine(
            mask_trunc_l[:, :, None], bbox_center, scale, out_res, interpolation=mask_xyz_interp
        )
        roi_mask_trunc_r = crop_resize_by_warp_affine(
            mask_trunc_r[:, :, None], bbox_center, scale, out_res, interpolation=mask_xyz_interp
        )

        # use original visible mask to calculate xyz loss (try full obj mask?)
        roi_mask_visib_l = crop_resize_by_warp_affine(
            mask_visib_l[:, :, None], bbox_center, scale, out_res, interpolation=mask_xyz_interp
        )
        roi_mask_visib_r = crop_resize_by_warp_affine(
            mask_visib_r[:, :, None], bbox_center, scale, out_res, interpolation=mask_xyz_interp
        )

        roi_mask_obj_l = crop_resize_by_warp_affine(
            mask_obj_l[:, :, None], bbox_center, scale, out_res, interpolation=mask_xyz_interp
        )
        roi_mask_obj_r = crop_resize_by_warp_affine(
            mask_obj_r[:, :, None], bbox_center, scale, out_res, interpolation=mask_xyz_interp
        )

        roi_mask_obj_erode_l = crop_resize_by_warp_affine(
            mask_obj_erode_l[:, :, None], bbox_center, scale, out_res, interpolation=mask_xyz_interp
        )
        roi_mask_obj_erode_r = crop_resize_by_warp_affine(
            mask_obj_erode_r[:, :, None], bbox_center, scale, out_res, interpolation=mask_xyz_interp
        )

        ## roi_xyz ----------------------------------------------------
        roi_xyz_l = crop_resize_by_warp_affine(xyz_l, bbox_center, scale, out_res, interpolation=mask_xyz_interp)
        roi_xyz_r = crop_resize_by_warp_affine(xyz_r, bbox_center, scale, out_res, interpolation=mask_xyz_interp)

        ## depth ------------------------------------------------------
        # depth_l = crop_resize_by_warp_affine(depth_l, bbox_center, scale, out_res, interpolation=mask_xyz_interp)
        # depth_r = crop_resize_by_warp_affine(depth_r, bbox_center, scale, out_res, interpolation=mask_xyz_interp)

        # focal_len = dataset_dict["cam"][0][0].item() # f_x [m]
        # baseline = dataset_dict["baseline"][0] # x baseline [m]
        if scale == 0:
            import json
            scale_error_path = '/igd/a4/homestud/jemrich/scale_error.json'
            try:
                with open(scale_error_path , 'r') as scale_file:
                    scale_error = json.load(scale_file)
            except:
                scale_error = []

            scale_error.append(dataset_dict['scene_im_id'])

            with open(scale_error_path, 'w') as scale_file:
                json.dump(scale_error, scale_file)
            print('scale == 0 error, scene and image id written to', scale_error_path)
            print(bbox_xyxy, im_W, im_H, dataset_dict['scene_im_id'], inst_infos)
        resize_ratio = out_res / scale
        # zero_mask_l = (depth_l == 0.0)
        # zero_mask_r = (depth_r == 0.0)
        # depth_l[zero_mask_l] = 1
        # depth_r[zero_mask_r] = 1
        # disparity_l = (((baseline * focal_len) / (depth_l)) * resize_ratio).astype(np.uint8)
        # disparity_r = (((baseline * focal_len) / (depth_r)) * resize_ratio).astype(np.uint8)
        # disparity_l[zero_mask_l] = 0
        # disparity_r[zero_mask_r] = 0

        # region label
        if g_head_cfg.NUM_REGIONS > 1:
            fps_points = self._get_fps_points(dataset_name)[roi_cls]
            roi_region_l = xyz_to_region(roi_xyz_l, fps_points)  # HW
            roi_region_r = xyz_to_region(roi_xyz_r, fps_points)  # HW
            dataset_dict["roi_region"] = torch.as_tensor(
                np.stack([roi_region_l, roi_region_r], axis=0),
                dtype=torch.int32
            ).contiguous()

        roi_xyz_l = roi_xyz_l.transpose(2, 0, 1)  # HWC-->CHW
        roi_xyz_r = roi_xyz_r.transpose(2, 0, 1)  # HWC-->CHW
        # normalize xyz to [0, 1] using extent
        roi_xyz_l[0] = roi_xyz_l[0] / roi_extent[0] + 0.5
        roi_xyz_l[1] = roi_xyz_l[1] / roi_extent[1] + 0.5
        roi_xyz_l[2] = roi_xyz_l[2] / roi_extent[2] + 0.5

        roi_xyz_r[0] = roi_xyz_r[0] / roi_extent[0] + 0.5
        roi_xyz_r[1] = roi_xyz_r[1] / roi_extent[1] + 0.5
        roi_xyz_r[2] = roi_xyz_r[2] / roi_extent[2] + 0.5

        xyz_loss_type = loss_cfg.XYZ_LOSS_TYPE
        if ("CE" in xyz_loss_type) or ("cls" in cfg.MODEL.POSE_NET.NAME):  # convert target to int for cls
            n_xyz_bin = g_head_cfg.XYZ_BIN
            # assume roi_xyz has been normalized in [0, 1]
            roi_xyz_bin_l = np.zeros_like(roi_xyz_l)
            roi_xyz_bin_r = np.zeros_like(roi_xyz_r)
            roi_x_norm_l = roi_xyz_l[0]
            roi_x_norm_r = roi_xyz_r[0]
            roi_x_norm_l[roi_x_norm_l < 0] = 0  # clip
            roi_x_norm_r[roi_x_norm_r < 0] = 0  # clip
            roi_x_norm_l[roi_x_norm_l > 0.999999] = 0.999999
            roi_x_norm_r[roi_x_norm_r > 0.999999] = 0.999999
            # [0, BIN-1]
            roi_xyz_bin_l[0] = np.asarray(roi_x_norm_l * n_xyz_bin, dtype=np.uint8)
            roi_xyz_bin_r[0] = np.asarray(roi_x_norm_r * n_xyz_bin, dtype=np.uint8)

            roi_y_norm_l = roi_xyz_l[1]
            roi_y_norm_r = roi_xyz_r[1]
            roi_y_norm_l[roi_y_norm_l < 0] = 0
            roi_y_norm_r[roi_y_norm_r < 0] = 0
            roi_y_norm_l[roi_y_norm_l > 0.999999] = 0.999999
            roi_y_norm_r[roi_y_norm_r > 0.999999] = 0.999999
            roi_xyz_bin_l[1] = np.asarray(roi_y_norm_l * n_xyz_bin, dtype=np.uint8)
            roi_xyz_bin_r[1] = np.asarray(roi_y_norm_r * n_xyz_bin, dtype=np.uint8)

            roi_z_norm_l = roi_xyz_l[2]
            roi_z_norm_r = roi_xyz_r[2]
            roi_z_norm_l[roi_z_norm_l < 0] = 0
            roi_z_norm_r[roi_z_norm_r < 0] = 0
            roi_z_norm_l[roi_z_norm_l > 0.999999] = 0.999999
            roi_z_norm_r[roi_z_norm_r > 0.999999] = 0.999999
            roi_xyz_bin_l[2] = np.asarray(roi_z_norm_l * n_xyz_bin, dtype=np.uint8)
            roi_xyz_bin_r[2] = np.asarray(roi_z_norm_r * n_xyz_bin, dtype=np.uint8)

            # the last bin is for bg
            roi_masks_l = {"trunc": roi_mask_trunc_l, "visib": roi_mask_visib_l, "obj": roi_mask_obj_l}
            roi_masks_r = {"trunc": roi_mask_trunc_r, "visib": roi_mask_visib_r, "obj": roi_mask_obj_r}
            roi_mask_xyz_l = roi_masks_l[loss_cfg.XYZ_LOSS_MASK_GT]
            roi_mask_xyz_r = roi_masks_r[loss_cfg.XYZ_LOSS_MASK_GT]
            roi_xyz_bin_l[0][roi_mask_xyz_l == 0] = n_xyz_bin
            roi_xyz_bin_r[0][roi_mask_xyz_r == 0] = n_xyz_bin
            roi_xyz_bin_l[1][roi_mask_xyz_l == 0] = n_xyz_bin
            roi_xyz_bin_r[1][roi_mask_xyz_r == 0] = n_xyz_bin
            roi_xyz_bin_l[2][roi_mask_xyz_l == 0] = n_xyz_bin
            roi_xyz_bin_r[2][roi_mask_xyz_r == 0] = n_xyz_bin

            if "CE" in xyz_loss_type:
                dataset_dict["roi_xyz_bin"] = torch.as_tensor(
                    np.stack([roi_xyz_bin_l, roi_xyz_bin_r], axis=0),
                    dtype=torch.uint8
                ).contiguous()
            if "/" in xyz_loss_type and len(xyz_loss_type.split("/")[1]) > 0:
                dataset_dict["roi_xyz"] = torch.as_tensor(
                    np.stack([roi_xyz_l, roi_xyz_r], axis=0),
                    dtype=torch.float32
                ).contiguous()
        else:
            dataset_dict["roi_xyz"] = torch.as_tensor(
                np.stack([roi_xyz_l, roi_xyz_r], axis=0),
                dtype=torch.float32
            ).contiguous()

        # depth


        # pose targets ----------------------------------------------------------------------
        pose_l = inst_infos["pose"] # left or right pose
        pose_r = np.full((3, 3), np.nan) # no right pose in stereobj for now
        dataset_dict["ego_rot"] = torch.as_tensor(
            np.stack([pose_l[:3, :3], pose_r[:3, :3]], axis=0).astype("float32")
        )
        trans_l = inst_infos['trans']
        trans_r = np.full((3,), np.nan)
        dataset_dict["trans"] = torch.as_tensor(
            np.stack([trans_l, trans_r], axis=0).astype("float32")
        )

        dataset_dict["roi_points"] = torch.as_tensor(
            self._get_model_points(dataset_name)[roi_cls].astype("float32")
        )
        dataset_dict["sym_info"] = self._get_sym_infos(dataset_name)[roi_cls]

        dataset_dict["roi_img"] = torch.as_tensor(
            np.stack([roi_img_l, roi_img_r], axis=0).astype("float32")
        ).contiguous()
        dataset_dict["roi_coord_2d"] = torch.as_tensor(
            np.stack([roi_coord_2d_l, roi_coord_2d_r], axis=0).astype("float32")
        ).contiguous()

        dataset_dict["roi_mask_trunc"] = torch.as_tensor(
            np.stack([roi_mask_trunc_l, roi_mask_trunc_r], axis=0).astype("float32")
        ).contiguous()
        dataset_dict["roi_mask_visib"] = torch.as_tensor(
            np.stack([roi_mask_visib_l, roi_mask_visib_r], axis=0).astype("float32")
        ).contiguous()
        dataset_dict["roi_mask_obj"] = torch.as_tensor(
            np.stack([roi_mask_obj_l, roi_mask_obj_r], axis=0).astype("float32")
        ).contiguous()
        dataset_dict["roi_mask_erode"] = torch.as_tensor(
            np.stack([roi_mask_obj_erode_l, roi_mask_obj_erode_r], axis=0).astype("float32")
        ).contiguous()

        dataset_dict["bbox_center"] = torch.as_tensor(bbox_center, dtype=torch.float32)
        dataset_dict["scale"] = scale
        dataset_dict["im_W"] = torch.as_tensor(im_W, dtype=torch.float32)
        dataset_dict["im_H"] = torch.as_tensor(im_H, dtype=torch.float32)
        baseline = dataset_dict["baseline"]
        dataset_dict["baseline"] = torch.as_tensor(baseline, dtype=torch.float32)
        dataset_dict["bbox"] = [anno["bbox_l"], anno["bbox_r"]]  # NOTE: original bbox
        dataset_dict["roi_wh"] = torch.as_tensor(np.array([bw, bh], dtype=np.float32))
        dataset_dict["resize_ratio"] = resize_ratio
        z_ratio_l = inst_infos["trans"][2] / resize_ratio
        # z_ratio_r = inst_infos["trans_r"][2] / resize_ratio
        obj_center_l = anno["centroid_2d"]
        # obj_center_r = anno["centroid_2d_r"]
        delta_c_l = obj_center_l - bbox_center
        # delta_c_r = obj_center_r - bbox_center
        trans_ratio_l = torch.as_tensor([delta_c_l[0] / bw, delta_c_l[1] / bh, z_ratio_l]).to(torch.float32)
        # trans_ratio_r = torch.as_tensor([delta_c_r[0] / bw, delta_c_r[1] / bh, z_ratio_r]).to(torch.float32)
        # depth
        # dataset_dict["disparity"] = torch.as_tensor(np.stack([disparity_l, disparity_r], axis=0)).contiguous()
        trans_ratio_r = torch.full((3,), np.nan)
        dataset_dict["trans_ratio"] = torch.stack([trans_ratio_l, trans_ratio_r], dim=0)
        return dataset_dict

    def smooth_xyz(self, xyz):
        """smooth the edge areas to reduce noise."""
        xyz = np.asarray(xyz, np.float32)
        xyz_blur = cv2.medianBlur(xyz, 3)
        edges = get_edge(xyz)
        xyz[edges != 0] = xyz_blur[edges != 0]
        return xyz

    def __getitem__(self, idx):
        if self.split != "train":
            dataset_dict = self._get_sample_dict(idx)
            return self.read_data(dataset_dict)

        while True:  # return valid data for train
            dataset_dict = self._get_sample_dict(idx)
            processed_data = self.read_data(dataset_dict)
            if processed_data is None:
                idx = self._rand_another(idx)
                continue
            return processed_data


def build_gdrn_train_loader(cfg, dataset_names):
    """A data loader is created by the following steps:

    1. Use the dataset names in config to query :class:`DatasetCatalog`, and obtain a list of dicts.
    2. Coordinate a random shuffle order shared among all processes (all GPUs)
    3. Each process spawn another few workers to process the dicts. Each worker will:
       * Map each metadata dict into another format to be consumed by the model.
       * Batch them by simply putting dicts into a list.

    The batched ``list[mapped_dict]`` is what this dataloader will yield.

    Args:
        cfg (CfgNode): the config

    Returns:
        an infinite iterator of training data
    """
    dataset_dicts = get_detection_dataset_dicts(
        dataset_names,
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
        min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE if cfg.MODEL.KEYPOINT_ON else 0,
        proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
    )

    dataset_dicts = filter_invalid_in_dataset_dicts(dataset_dicts, visib_thr=cfg.DATALOADER.FILTER_VISIB_THR)

    if cfg.MODEL.POSE_NET.XYZ_ONLINE:
        dataset = GDRN_Online_DatasetFromList(cfg, split="train", lst=dataset_dicts, copy=False)
    else:
        dataset = GDRN_DatasetFromList(cfg, split="train", lst=dataset_dicts, copy=False)

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger = logging.getLogger(__name__)
    logger.info("Using training sampler {}".format(sampler_name))
    # TODO avoid if-else?
    if sampler_name == "TrainingSampler":
        sampler = TrainingSampler(len(dataset))
    elif sampler_name == "RepeatFactorTrainingSampler":
        repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
            dataset_dicts, cfg.DATALOADER.REPEAT_THRESHOLD
        )
        sampler = RepeatFactorTrainingSampler(repeat_factors)
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))
    return my_build_batch_data_loader(
        dataset,
        sampler,
        cfg.SOLVER.IMS_PER_BATCH,
        aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )


def build_gdrn_test_loader(cfg, dataset_name, train_objs=None):
    """Similar to `build_detection_train_loader`. But this function uses the
    given `dataset_name` argument (instead of the names in cfg), and uses batch
    size 1.

    Args:
        cfg: a detectron2 CfgNode
        dataset_name (str): a name of the dataset that's available in the DatasetCatalog

    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
        dataset, with test-time transformation and batching.
    """
    dataset_dicts = get_detection_dataset_dicts(
        [dataset_name],
        filter_empty=False,
        proposal_files=[cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(dataset_name)]]
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )

    # load test detection results
    if cfg.MODEL.LOAD_DETS_TEST:
        det_files = cfg.DATASETS.DET_FILES_TEST
        assert len(cfg.DATASETS.TEST) == len(det_files)
        dataset_dicts = load_detections_into_dataset(
            dataset_name,
            dataset_dicts,
            det_file=det_files[cfg.DATASETS.TEST.index(dataset_name)],
            top_k_per_obj=cfg.DATASETS.DET_TOPK_PER_OBJ,
            score_thr=cfg.DATASETS.DET_THR,
            train_objs=train_objs,
        )
        if cfg.DATALOADER.FILTER_EMPTY_DETS:
            dataset_dicts = filter_empty_dets(dataset_dicts)

    dataset = GDRN_DatasetFromList(cfg, split="test", lst=dataset_dicts, flatten=False)

    sampler = InferenceSampler(len(dataset))
    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers.
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    # Horovod: limit # of CPU threads to be used per worker.
    # if num_workers > 0:
    #     torch.set_num_threads(num_workers)

    kwargs = {"num_workers": num_workers}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    # https://github.com/horovod/horovod/blob/master/examples/pytorch/pytorch_imagenet_resnet50.py
    # if (num_workers > 0 and hasattr(mp, '_supports_context') and
    #         mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
    #     kwargs['multiprocessing_context'] = 'forkserver'
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=batch_sampler, collate_fn=trivial_batch_collator, **kwargs
    )
    return data_loader