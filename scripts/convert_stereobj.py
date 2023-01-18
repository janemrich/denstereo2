from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np
import json


split = Path('/opt/spool/jemrich/stereobj_1m/split')
train_left = Path('/opt/spool/jemrich/stereobj_1m/train~left')
train_right = Path('/opt/spool/jemrich/stereobj_1m/train~right')
test_right = Path('/opt/spool/jemrich/stereobj_1m/test~right')
test_left = Path('/opt/spool/jemrich/stereobj_1m/test~left')
val_left = Path('/opt/spool/jemrich/stereobj_1m/val~left')
val_right = Path('/opt/spool/jemrich/stereobj_1m/val~right')
images_annotations = Path('/opt/spool/jemrich/stereobj_1m/images_annotations')

bop_splits = {
    'train': {
        'left': train_left,
        'right': train_right,
    },
    'test': {
        'left': test_left,
        'right': test_right,
    },
    'val': {
        'left': val_left,
        'right': val_right,
    },
}

# filter for .txt files
split_txts = [f for f in split.iterdir() if f.suffix == '.txt']
#filter for .json files
split_jsons = [f for f in split.iterdir() if f.suffix == '.json']


# filter txts for train, val, test
train_txts = [f for f in split_txts if 'train' in f.name and not 'val' in f.name]
val_txts = [f for f in split_txts if 'trainval' in f.name]
test_txts = [f for f in split_txts if 'test' in f.name]

train_scenes = []
for f in train_txts:
    with open(f, 'r') as f:
        train_scenes.extend(f.read().splitlines())
val_scenes = []
for f in val_txts:
    with open(f, 'r') as f:
        val_scenes.extend(f.read().splitlines())
test_scenes = []
for f in test_txts:
    with open(f, 'r') as f:
        test_scenes.extend(f.read().splitlines())

scenes = {
    'train': train_scenes,
    'val': val_scenes,
    'test': test_scenes,
}

def create_folders():
    for split in tqdm(scenes.keys()):
        for scene in tqdm(scenes[split]):
            for side in ['left', 'right']:
                for folder in ['rgb', 'mask_visib', 'depth']:
                    Path(bop_splits[split][side] / scene / folder).mkdir(parents=True, exist_ok=True)

def move_images():
    # move files from images_annotations to train~left, val~left, test~left
    for scene in train_scenes:
        for f in images_annotations.glob(scene + '/*'):
            f.rename(train_left / scene / f.name)
    for scene in val_scenes:
        for f in images_annotations.glob(scene + '/*'):
            f.rename(val_left / scene / f.name)
    for scene in test_scenes:
        for f in images_annotations.glob(scene + '/*'):
            f.rename(test_left / scene / f.name)


def transform_images():
    total = 0
    for split in bop_splits.keys():
        for scene in scenes[split]:
            total += len(list(images_annotations.glob(scene + '/*.jpg')))

    with tqdm(total=total) as pbar:
        for split in tqdm(bop_splits.keys(), desc='Splits'):
            for scene in tqdm(scenes[split], leave=False, desc='Scenes'):
                for f in tqdm(images_annotations.glob(scene + '/*.jpg'), leave=False, desc='Images'):
                    im = Image.open(f)
                    w, h = im.size
                    # (left, upper, right, lower)
                    im_l = im.crop((0, 0, w//2, h))
                    im_r = im.crop((w//2, 0, w, h))

                    path_l = bop_splits[split]['left'] / scene / 'rgb' / f.name
                    path_r = bop_splits[split]['right'] / scene / 'rgb' / f.name
                    im_l.save(path_l)
                    im_r.save(path_r)

def transform_masks():
    file_ending = '_mask_label.npz'
    total = 0
    for split in bop_splits.keys():
        for scene in scenes[split]:
            total += len(list(images_annotations.glob(scene + '/*' + file_ending)))

    with tqdm(total=total) as pbar:
        for split in tqdm(bop_splits.keys(), desc='Splits'):
            for scene in tqdm(scenes[split], leave=False, desc='Scenes'):
                for f in tqdm(images_annotations.glob(scene + '/*.jpg'), leave=False, desc='Masks'):

                    ##### rt_label
                    label_path = images_annotations / scene / (f.stem + '_rt_label.json')
                    print('label_path', label_path)
                    with open(label_path, 'r') as rt_f:
                        rt_data = json.load(rt_f)
                        print('rt_data', rt_data)

                    mask_path = images_annotations / scene / (f.stem + '_mask_label.npz')
                    obj_mask = np.load(mask_path)['masks'].item()
                    for obj in rt_data['class']:

                        print('obj', obj)
                        mask_l = obj_mask['left'][obj]
                        mask_r = obj_mask['right'][obj]
                        print('mask_l', mask_l)

                        # save as npz with in mask_visib
                        # np.savez_compressed(bop_splits[split]['left'] / scene / 'mask_visib' / (f.stem + file_ending), mask_l)
                        # np.savez_compressed(bop_splits[split]['right'] / scene / 'mask_visib' / (f.stem + file_ending), mask_r)


# create_folders()
# transform_images()
transform_masks()