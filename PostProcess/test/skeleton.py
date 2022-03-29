import os
import queue
import argparse
from utils import load_image, log_success
import mmcv
import cv2
import numpy as np
from libs import skeleton


def main(img_path, output_path):
    mmcv.mkdir_or_exist(output_path)

    for img_name in mmcv.scandir(img_path):
        path = os.path.join(img_path, img_name)
        img = load_image(path, 'GRAY')
        log_success('Load image {} success!'.format(path))

        img = skeleton(img)
        log_success('Skeleton image {} success!'.format(img_name))

        path = os.path.join(output_path, img_name)
        cv2.imwrite(path, img)
        log_success('Save image {} success.\n'.format(path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('img_path', help='image path')
    parser.add_argument('output_path', help='output path')
    args = parser.parse_args()

    main(args.img_path, args.output_path)
