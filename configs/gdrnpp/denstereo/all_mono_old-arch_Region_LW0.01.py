_base_ = ["../../_base_/gdrn_base.py"]

OUTPUT_DIR = "output/gdrn/ycbvPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/002_master_chef_can"
INPUT = dict(
    DZI_PAD_SCALE=1.5,
    # TRUNCATE_FG=True,
    CHANGE_BG_PROB=0.5,
    COLOR_AUG_PROB=0.8,
    COLOR_AUG_TYPE="code",
    COLOR_AUG_CODE=(
        "Sequential(["
        # Sometimes(0.5, PerspectiveTransform(0.05)),
        # Sometimes(0.5, CropAndPad(percent=(-0.05, 0.1))),
        # Sometimes(0.5, Affine(scale=(1.0, 1.2))),
        "Sometimes(0.5, CoarseDropout( p=0.2, size_percent=0.05) ),"
        "Sometimes(0.4, GaussianBlur((0., 3.))),"
        "Sometimes(0.3, pillike.EnhanceSharpness(factor=(0., 50.))),"
        "Sometimes(0.3, pillike.EnhanceContrast(factor=(0.2, 50.))),"
        "Sometimes(0.5, pillike.EnhanceBrightness(factor=(0.1, 6.))),"
        "Sometimes(0.3, pillike.EnhanceColor(factor=(0., 20.))),"
        "Sometimes(0.5, Add((-25, 25), per_channel=0.3)),"
        "Sometimes(0.3, Invert(0.2, per_channel=True)),"
        "Sometimes(0.5, Multiply((0.6, 1.4), per_channel=0.5)),"
        "Sometimes(0.5, Multiply((0.6, 1.4))),"
        "Sometimes(0.1, AdditiveGaussianNoise(scale=10, per_channel=True)),"
        "Sometimes(0.5, iaa.contrast.LinearContrast((0.5, 2.2), per_channel=0.3)),"
        "Sometimes(0.5, Grayscale(alpha=(0.0, 1.0))),"  # maybe remove for det
        "], random_order=True)"
        # cosy+aae
    ),
)

SEED = 0

SOLVER = dict(
    IMS_PER_BATCH=36,
    TOTAL_EPOCHS=100,
    LR_SCHEDULER_NAME="flat_and_anneal",
    ANNEAL_METHOD="cosine",  # "cosine"
    ANNEAL_POINT=0.72,
    OPTIMIZER_CFG=dict(_delete_=True, type="Ranger", lr=8e-4, weight_decay=0.01),
    WEIGHT_DECAY=0.0,
    WARMUP_FACTOR=0.001,
    WARMUP_ITERS=1000,
)

DATASETS = dict(
    TRAIN=("denstereo_train_pbr_left",),
    TEST=("denstereo_test_pbr_left",),
    DET_FILES_TEST=("datasets/BOP_DATASETS/denstereo/test_bboxes/test_pbr_left.json",),
    SYM_OBJS=[
        "024_bowl",
        "036_wood_block",
        "051_large_clamp",
        "052_extra_large_clamp",
        "061_foam_brick",
    ],  # used for custom evalutor
)

DATALOADER = dict(
    # Number of data loading threads
    NUM_WORKERS=8,
    FILTER_VISIB_THR=0.3,
)

MODEL = dict(
    LOAD_DETS_TEST=True,
    PIXEL_MEAN=[0.0, 0.0, 0.0],
    PIXEL_STD=[255.0, 255.0, 255.0],
    BBOX_TYPE="AMODAL_CLIP",  # VISIB or AMODAL
    POSE_NET=dict(
        NAME="GDRN",
        XYZ_ONLINE=False,
        BACKBONE=dict(
            FREEZE=False,
            PRETRAINED="timm",
            INIT_CFG=dict(
                type="timm/resnest50d",
                # type="timm/convnext_base",
                pretrained=True,
                in_chans=3,
                features_only=True,
                out_indices=(4,), # resnest
                # out_indices=(3,), # convnext
            ),
        ),
        ## geo head: Mask, XYZ, Region
        GEO_HEAD=dict(
            FREEZE=False,
            INIT_CFG=dict(
                type="TopDownMaskXyzRegionHead",
                # in_dim=1024, # convnext
                in_dim=2048,  # this is num out channels of backbone conv feature
            ),
            NUM_REGIONS=64,
        ),
        SELFOCC_HEAD=dict(
            OCCMASK_AWARE=False,
            Q0_CLASS_AWARE=False,
            MASK_CLASS_AWARE=False,
            FREEZE=True,
            INIT_CFG=dict(
                type="ConvSelfoccHead",
                # in_dim=1024, # convnext
                in_dim=2048,
                feat_dim=256,
                feat_kernel_size=3,
                norm="GN",
                num_gn_groups=32,
                act="GELU",  # relu | lrelu | silu (swish) | gelu | mish
                out_kernel_size=1,
                out_layer_shared=False,
                Q0_num_classes=1,
                mask_num_classes=1,
            ),
            MIN_Q0_REGION=20,
            LR_MULT=1.0,

            REGION_CLASS_AWARE=False,
            MASK_THR_TEST=0.5,
        ),
        PNP_NET=dict(
            INIT_CFG=dict(norm="GN", act="gelu"),
            REGION_ATTENTION=True,
            WITH_2D_COORD=True,
            ROT_TYPE="allo_rot6d",
            TRANS_TYPE="centroid_z",
        ),
        LOSS_CFG=dict(
            # xyz loss ----------------------------
            XYZ_LOSS_TYPE="L1",  # L1 | CE_coor
            XYZ_LOSS_MASK_GT="visib",  # trunc | visib | obj
            XYZ_LW=1.0,
            # mask loss ---------------------------
            MASK_LOSS_TYPE="L1",  # L1 | BCE | CE
            MASK_LOSS_GT="trunc",  # trunc | visib | gt
            MASK_LW=1.0,
            # full mask loss ---------------------------
            FULL_MASK_LOSS_TYPE="L1",  # L1 | BCE | CE
            FULL_MASK_LW=1.0,
            # region loss -------------------------
            REGION_LOSS_TYPE="CE",  # CE
            REGION_LOSS_MASK_GT="visib",  # trunc | visib | obj
            REGION_LW=0.01,
            # REGION_LW=1.0, # should be 0.1 at least for now to make it comparable
            # pm loss --------------
            PM_LOSS_SYM=True,  # NOTE: sym loss
            PM_R_ONLY=True,  # only do R loss in PM
            PM_LW=1.0,
            # centroid loss -------
            CENTROID_LOSS_TYPE="L1",
            CENTROID_LW=1.0,
            # z loss -----------
            Z_LOSS_TYPE="L1",
            Z_LW=1.0,

            # just for selfocc
            # Q0 loss ---------------------
            Q0_LOSS_TYPE="L1",
            Q0_LOSS_MASK_GT="visib",  # computed from Q0
            Q0_LW=0.0,
            Q0_DEF_LW=0.0, # 10?
            # cross-task loss -------------------
            CT_LW=0.0,
            CT_P_LW=0.0,
            # occlusion mask loss weight
            OCC_LW=0.0,
            PM_NORM_BY_EXTENT=True,
            # Q direction
            QD_LW=0.0,
            #
            HANDLE_SYM=False,
        ),
    ),
)

VAL = dict(
    DATASET_NAME="denstereo",
    SPLIT="test~left",
    SPLIT_TYPE="pbr",
    SCRIPT_PATH="lib/pysixd/scripts/eval_pose_results_more.py",
    TARGETS_FILENAME="test_bboxes/test_targets_pbr_left.json",
    ERROR_TYPES="vsd,mspd,mssd",
    USE_BOP=True,  # whether to use bop toolkit
)

TEST = dict(EVAL_PERIOD=0, VIS=False, TEST_BBOX_TYPE="est")  # gt | est
