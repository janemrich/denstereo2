from pathlib import Path
import json
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

    return [x, y, width, height]

def generate_bboxes(bop_path, dataset, split, scene_from, scene_to):
    dataset_path = bop_path / dataset / split
    test_bboxes = {}
    for scene_id in tqdm.tqdm(range(scene_from, scene_to)):
        scene = Path(f'{scene_id:06d}')

        with open(dataset_path / scene / 'scene_gt_info.json', 'r') as gt_info_file:
            with open(dataset_path / scene / 'scene_gt.json', 'r') as gt_file:
                gt = json.load(gt_file)
                gt_info = json.load(gt_info_file)

                for image in gt_info.keys():
                    detections = []
                    for anno_i, anno in enumerate(gt_info[image]):
                        scene_im_id = f"{scene_id}/{int(image)}"

                        detections.append(
                            {
                                'obj_id': gt[image][anno_i]['obj_id'],
                                'bbox_est': get_noisy_bbox(anno['bbox_visib'])
                            }
                        )

                    test_bboxes[scene_im_id] = detections


    with open(dataset_path / 'bbox.json', 'w') as outfile:
        json.dump(test_bboxes, outfile, indent=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate noisy test bounding boxes')
    parser.add_argument('--bop_path', type=str, default='/home/jemrich/datasets/BOP_DATASETS/')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--split', type=str)
    parser.add_argument('--scene_from', type=int, default=0)
    parser.add_argument('--scene_to', type=int, default=50)
    args = parser.parse_args()

    bop_path = Path(args.bop_path)
    generate_bboxes(bop_path, args.dataset, args.split, args.scene_from, args.scene_to)

    



