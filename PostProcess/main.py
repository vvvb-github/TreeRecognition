import os
import argparse
from utils import load_image, log_info, log_highlight, log_success, log_error
import mmcv
import cv2
import numpy as np
import json
from libs import skeleton, detect_keypoints, recognize


def main(img_path, output, save, load):
    save_info = dict()
    out_file = open(output, 'w')

    if load != None:
        log_info('Load keypoint result from path {}.'.format(load))
        load_path = os.path.join(load, 'keypoints.json')
        keypoints = json.load(open(load_path, 'r'))

    for img_name in mmcv.scandir(img_path):
        # load keypoints and image
        if load != None:
            root = keypoints[img_name]['root']
            branches = keypoints[img_name]['branch']
            leafs = keypoints[img_name]['leaf']
            path = os.path.join(load, img_name)
            img = load_image(path, 'GRAY')
        else:
            path = os.path.join(img_path, img_name)
            img = load_image(path, 'GRAY')

            img = skeleton(img, binary=False)
            root, branches, leafs = detect_keypoints(img)

            if save != None:
                kps = dict()
                kps['root'] = root
                kps['branch'] = branches
                kps['leaf'] = leafs

                save_info[img_name] = kps
                path = os.path.join(save, img_name)
                cv2.imwrite(path, img)

        # recognize
        res = recognize(img, root, branches, leafs)
        log_highlight(
            'Tree structure for {} recognized as {}.'.format(img_name, res))
        out_file.write('[{}]: {}\n'.format(img_name, res))

    out_file.close()

    if save != None:
        log_info('Save keypoint result under path {}.'.format(save))
        path = os.path.join(save, 'keypoints.json')
        res = open(path, 'w')
        json.dump(save_info, res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('img_path', help='image path')
    parser.add_argument('output', help='output file')
    parser.add_argument('--save', default=None,
                        help='save keypoint result as file')
    parser.add_argument('--load', default=None, help='load file for keypoint')
    args = parser.parse_args()

    main(args.img_path, args.output, args.save, args.load)
