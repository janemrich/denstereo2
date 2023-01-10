from pathlib import Path
import json
from turtle import right
import numpy as np
import tqdm
import argparse

IM_WIDTH, IM_HEIGHT = 640, 480

def get_noisy_bbox(bbox):
    x, y, width, height = bbox
    x_noise = np.clip(np.random.normal(loc=0, scale=3), a_min=-10, a_max=10) 
    y_noise = np.clip(np.random.normal(loc=0, scale=3), a_min=-10, a_max=10)
    width_noise = np.random.uniform(low=0.9, high=1.1)
    height_noise = np.random.uniform(low=0.9, high=1.1)

    x += x_noise
    y += y_noise
    width *= width_noise
    height *= height_noise

    # make sure box position is inside image
    x = max(0, min(IM_WIDTH-1, x))
    y = max(0, min(IM_HEIGHT-1, y))
    # y,x is the top left corner
    if (IM_WIDTH-1) < (x+width):
        width = (IM_WIDTH-1) - x
    if (IM_HEIGHT-1) < (y+height):
        height = (IM_HEIGHT-1) - y

    if width < 5 or height < 5:
        return get_noisy_bbox(bbox)

    return [x, y, width, height]

def generate_bboxes(bop_path, dataset, scene_from, scene_to, name):
    dataset_path = bop_path / dataset
    left_path = dataset_path / 'train_pbr_left'
    right_path = dataset_path / 'train_pbr_right'
    test_bboxes = {}
    for scene_id in tqdm.tqdm(range(scene_from, scene_to)):
        scene = Path(f'{scene_id:06d}')

        with open(left_path / scene / 'scene_gt_info.json', 'r') as gt_info_file_left:
            with open(right_path / scene / 'scene_gt_info.json', 'r') as gt_info_file_right:
                with open(left_path / scene / 'scene_gt.json', 'r') as gt_file:
                    gt = json.load(gt_file)
                    gt_info_l = json.load(gt_info_file_left)
                    gt_info_r = json.load(gt_info_file_right)

                    for image in gt_info_l.keys():
                        detections = []
                        for anno_i, (anno_l, anno_r) in enumerate(zip(gt_info_l[image], gt_info_r[image])):
                            scene_im_id = f"{scene_id}/{int(image)}"

                            bbox_l = get_noisy_bbox(anno_l['bbox_visib'])
                            bbox_r = get_noisy_bbox(anno_r['bbox_visib'])

                            bbox = [
                                min(bbox_l[0], bbox_r[0]),
                                min(bbox_l[1], bbox_r[1]),
                                max(bbox_l[2], bbox_r[2]),
                                max(bbox_l[3], bbox_r[3]),
                            ]
                            detections.append(
                                {
                                    'obj_id': gt[image][anno_i]['obj_id'],
                                    'bbox_est': bbox,
                                    'bbox_est_left': bbox_l,
                                    'bbox_est_right': bbox_r,
                                }
                            )

                        test_bboxes[scene_im_id] = detections


    with open(bop_path / dataset / 'test_bboxes' / (name + '.json'), 'w') as outfile:
        json.dump(test_bboxes, outfile, indent=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate noisy test bounding boxes')
    parser.add_argument('--bop_path', type=str, default='/home/jemrich/datasets/BOP_DATASETS/')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--scene_from', type=int, default=0)
    parser.add_argument('--scene_to', type=int, default=50)
    parser.add_argument('--name', type=str, default='bbox_stereo', help='bbox file name')
    args = parser.parse_args()

    bop_path = Path(args.bop_path)
    generate_bboxes(
        bop_path,
        args.dataset,
        args.scene_from,
        args.scene_to,
        args.name,
    )

    



