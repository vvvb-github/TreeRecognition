import os
import argparse
from utils import load_image, binary_image, log_info, log_success
import mmcv
import cv2
import numpy as np
from libs import detect_keypoints


color = {
    'root': np.array([255, 0, 0], dtype=np.uint8),
    'branch': np.array([0, 255, 0], dtype=np.uint8),
    'leaf': np.array([0, 0, 255], dtype=np.uint8)
}


def draw_point(img, position, color, r=1):
    (x, y) = position
    lx, rx = x-r, x+r
    ly, ry = y-r, y+r

    for i in range(lx, rx+1):
        for j in range(ly, ry+1):
            img[i, j] = color


def main(img_path, output_path, n, radius):
    mmcv.mkdir_or_exist(output_path)

    for img_name in mmcv.scandir(img_path):
        path = os.path.join(img_path, img_name)
        img = load_image(path, 'GRAY')
        img = binary_image(img, 255, 1)
        log_success('Load image {} success!'.format(path))

        root, branches, leafs = detect_keypoints(img, n)
        log_info('Detect root {}.'.format(root))
        log_info('Detect {} branch points.'.format(len(branches)))
        log_info('Detect {} leaf points.'.format(len(leafs)))

        img = binary_image(img, 1, 255)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        draw_point(img, root, color['root'], radius)
        for p in branches:
            draw_point(img, p, color['branch'], radius)
        for p in leafs:
            draw_point(img, p, color['leaf'], radius)

        path = os.path.join(output_path, img_name)
        cv2.imwrite(path, img)
        log_success('Save image {} success.\n'.format(path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('img_path', help='image path')
    parser.add_argument('output_path', help='output path')
    parser.add_argument('--branch_n',type=int,default=9)
    parser.add_argument('--radius', type=int, default=1)
    args = parser.parse_args()

    main(args.img_path, args.output_path, args.branch_n, args.radius)
