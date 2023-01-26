from pathlib import Path
import json
import os.path as osp

import numpy as np
from tqdm import tqdm

def get_scenes(dataset_root, split):
    train_scenes = []
    with open(osp.join(dataset_root, "split", "biolab_train_scenes.txt"), "r") as f:
        train_scenes = f.read().splitlines()
    with open(osp.join(dataset_root, "split", "mechanics_train_scenes.txt"), "r") as f:
        train_scenes.extend(f.read().splitlines())

    test_scenes = []
    with open(osp.join(dataset_root, "split", "biolab_test_scenes.txt"), "r") as f:
        test_scenes = f.read().splitlines()
    with open(osp.join(dataset_root, "split", "mechanics_test_scenes.txt"), "r") as f:
        test_scenes.extend(f.read().splitlines())

    val_scenes = []
    with open(osp.join(dataset_root, "split", "biolab_val_scenes.txt"), "r") as f:
        val_scenes = f.read().splitlines()
    with open(osp.join(dataset_root, "split", "mechanics_val_scenes.txt"), "r") as f:
        val_scenes.extend(f.read().splitlines())

    trainval_scenes = []
    with open(osp.join(dataset_root, "split", "biolab_trainval_scenes.txt"), "r") as f:
        trainval_scenes = f.read().splitlines()
    with open(osp.join(dataset_root, "split", "mechanics_trainval_scenes.txt"), "r") as f:
        trainval_scenes.extend(f.read().splitlines())

    all_scenes = trainval_scenes + test_scenes
    
    if split == 'train':
        return train_scenes
    elif split == 'test':
        return test_scenes
    elif split == 'val':
        return val_scenes
    elif split == 'trainval':
        return trainval_scenes
    elif split == 'all':
        return all_scenes


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


def load_bboxes(mask_path):
    obj_mask = np.load(mask_path, allow_pickle=True)['masks'].item()
    bboxes = {}
    for obj in obj_mask['left']:
        x_min_l = obj_mask['left'][obj]['x_min']
        x_max_l = obj_mask['left'][obj]['x_max']
        y_min_l = obj_mask['left'][obj]['y_min']
        y_max_l = obj_mask['left'][obj]['y_max']
        if x_min_l is not None:
            bboxes[obj] = [
                int(x_min_l),
                int(y_min_l),
                int(x_max_l),
                int(y_max_l),
            ]
    return bboxes


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="gen denstereo train_pbr xyz")
    # parser.add_argument("--bop_path", type=str, default="/opt/spool/jemrich/")
    parser.add_argument("--dataset_path", type=str, default="datasets/BOP_DATASETS/stereobj_1m")
    parser.add_argument("--scene", type=str, default="all", help="scene id")
    parser.add_argument("--split", type=str, default="val", help="train, test, val, trainval, all")
    args = parser.parse_args()

    # base_dir = args.bop_path
    # dataset_root = Path(base_dir) / args.dataset
    dataset_root = Path(args.dataset_path)

    all_scenes = get_scenes(dataset_root, 'all')
    id2scene = {i: scene for i, scene in enumerate(sorted(all_scenes))}
    scene2id = {scene: id for id, scene in id2scene.items()}
    print(id2scene)

    scenes = get_scenes(dataset_root, args.split)
    scenes = [id2scene[167]]
    out_dict = {}
    for scene in scenes:
        scene_root = dataset_root / scene

        for label_path in tqdm(scene_root.glob('*_rt_label.json'), leave=False, desc=f"Image in Scene {scene}"):
            str_im_id = label_path.stem[:6]
            
            mask_path = scene_root / (str_im_id + '_mask_label.npz')
            scene_im_id = "{}/{}".format(scene2id[scene], int(str_im_id)) 
            print(scene_im_id)

            if not label_path.exists():
                print(f"Label file does not exist: {label_path}")
                continue
            with open(label_path, 'r') as rt_f:
                rt_data = json.load(rt_f)

            out_labels = []
            bboxes = load_bboxes(mask_path)
            for obj, bbox in bboxes.items():
                detection = {
                    "obj_id": int(obj2id[rt_data['class'][obj]]),
                    "bbox_est": bbox,
                }
                out_labels.append(detection)
            out_dict[scene_im_id] = out_labels
    
    out_path = dataset_root / "test_bboxes"
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "debug_left.json", "w") as f:
        json.dump(out_dict, f, indent=1)