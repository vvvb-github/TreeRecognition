import numpy as np
import random
import math
import cv2
from .log_utils import log_success, log_warn, log_error, log_info


def load_img(path, format='RGB'):
    try:
        img = cv2.imread(path)
        b, g, r = cv2.split(img)
        if format == 'RGB':
            img = cv2.merge([r, g, b])
        # TODO other format
    except Exception as e:
        log_error('Load image {} with error!'.format(path))
        raise

    return img


def random_horizontal_flip(imgs):
    if random.random() < 0.5:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1).copy()
    return imgs


def random_vertical_flip(imgs):
    if random.random() < 0.5:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=0).copy()
    return imgs


def scale_aligned(img, h_scale, w_scale):
    h, w = img.shape[0:2]
    h = int(h * h_scale + 0.5)
    w = int(w * w_scale + 0.5)
    if h % 32 != 0:
        h = h + (32 - h % 32)
    if w % 32 != 0:
        w = w + (32 - w % 32)
    img = cv2.resize(img, dsize=(w, h))

    return img


def random_scale_with_gt(img, gt, short_size, scales, aspects):
    h, w = img.shape[0:2]
    scale = np.random.uniform(scales[0], scales[1])
    scale = (scale * short_size) / min(h, w)
    if aspects is not None:
        aspect = np.random.uniform(aspects[0], aspects[1])
        h_scale = scale * math.sqrt(aspect)
        w_scale = scale / math.sqrt(aspect)
        img = scale_aligned(img, h_scale, w_scale)
        gt = scale_aligned(gt, h_scale, w_scale)
    else:
        img = scale_aligned(img, scale, scale)
        gt = scale_aligned(gt, scale, scale)

    return img, gt


def random_crop_padding(imgs, target_size):
    h, w = imgs[0].shape[0:2]
    t_w, t_h = target_size
    p_w, p_h = target_size
    if w == t_w and h == t_h:
        return imgs

    t_h = t_h if t_h < h else h
    t_w = t_w if t_w < w else w

    if random.random() > 3.0 / 8.0 and np.max(imgs[1]) > 0:
        # make sure to crop the text region
        tl = np.min(np.where(imgs[1] > 0), axis=1) - (t_h, t_w)
        tl[tl < 0] = 0
        br = np.max(np.where(imgs[1] > 0), axis=1) - (t_h, t_w)
        br[br < 0] = 0
        br[0] = min(br[0], h - t_h)
        br[1] = min(br[1], w - t_w)

        i = random.randint(tl[0], br[0]) if tl[0] < br[0] else 0
        j = random.randint(tl[1], br[1]) if tl[1] < br[1] else 0
    else:
        i = random.randint(0, h - t_h) if h - t_h > 0 else 0
        j = random.randint(0, w - t_w) if w - t_w > 0 else 0

    n_imgs = []
    for idx in range(len(imgs)):
        if len(imgs[idx].shape) == 3:
            s3_length = int(imgs[idx].shape[-1])
            img = imgs[idx][i:i + t_h, j:j + t_w, :]
            img_p = cv2.copyMakeBorder(img, 0, p_h - t_h, 0, p_w - t_w, borderType=cv2.BORDER_CONSTANT,
                                       value=tuple(0 for i in range(s3_length)))
        else:
            img = imgs[idx][i:i + t_h, j:j + t_w]
            img_p = cv2.copyMakeBorder(
                img, 0, p_h - t_h, 0, p_w - t_w, borderType=cv2.BORDER_CONSTANT, value=(0,))
        n_imgs.append(img_p)
    return n_imgs


def scale_aligned_short(img, short_size=640, mode=None):
    h, w = img.shape[0:2]
    scale = short_size * 1.0 / min(h, w)
    h = int(h * scale + 0.5)
    w = int(w * scale + 0.5)
    if h % 32 != 0:
        h = h + (32 - h % 32)
    if w % 32 != 0:
        w = w + (32 - w % 32)
    if mode is not None:
        img = cv2.resize(img, dsize=(w, h), interpolation=mode)
    else:
        img = cv2.resize(img, dsize=(w, h))
    return img


def scale_aligned_long(img, long_size=640):
    h, w = img.shape[0:2]
    scale = long_size * 1.0 / max(h, w)
    h = int(h * scale + 0.5)
    w = int(w * scale + 0.5)
    if h % 32 != 0:
        h = h + (32 - h % 32)
    if w % 32 != 0:
        w = w + (32 - w % 32)
    img = cv2.resize(img, dsize=(w, h))
    return img
