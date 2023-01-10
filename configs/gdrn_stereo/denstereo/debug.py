_base_ = "./all_early.py"
OUTPUT_DIR = "output/gdrn_stereo/denstereo/debug"
DATASETS = dict(TRAIN=("denstereo_debug_train_pbr",), TEST=("denstereo_debug_test_pbr",))

DATALOADER = dict(NUM_WORKERS=1,)

SOLVER = dict(
    TOTAL_EPOCHS=10,
    IMS_PER_BATCH=16,
    )
