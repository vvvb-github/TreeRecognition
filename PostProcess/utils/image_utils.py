import cv2
from .log_utils import log_success, log_warn, log_error, log_info


def load_image(path, format='BGR'):
    try:
        if format == 'RGB':
            img = cv2.imread(path)
            b, g, r = cv2.split(img)
            img = cv2.merge([r, g, b])
        elif format == 'BGR':
            img = cv2.imread(path)
        elif format == 'GRAY':
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # TODO other format
    except Exception as e:
        log_error('Load image {} with format {} error!'.format(path, format))
        raise

    return img


def binary_image(img, threshold, value, background=0):
    try:
        shape = img.shape
        for h in range(shape[0]):
            for w in range(shape[1]):
                if img[h, w] >= threshold:
                    img[h, w] = value
                else:
                    img[h, w] = background
    except:
        log_error('Get error while binarying image!')
        raise

    return img
