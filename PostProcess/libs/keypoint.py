from utils import convolution, distance, multi_mat
import numpy as np


def remove_near(points, dis=10):
    res = list()

    for i in range(len(points)):
        should_rm = False
        for j in range(i):
            if distance(points[i], points[j]) <= dis:
                should_rm = True
                break
        if not should_rm:
            res.append(points[i])

    return res


def detect_root(img, add=2):
    (h, w) = img.shape

    for j in range(w):
        for i in range(h):
            if img[i, j] == 1:
                for k in range(1, add+1):
                    img[i, j-k] = 1
                return (i, j-add)


def detect_leaf(img, root=None):
    (h, w) = img.shape
    kernel = np.array(
        [[1, 1, 1],
         [1, 0, 1],
         [1, 1, 1]], dtype=np.uint8
    )
    leafs = list()

    for j in range(w):
        for i in range(h):
            if (i, j) == root:
                continue
            if img[i, j] == 1 and convolution(img, (i, j), kernel) == 1:
                leafs.append((i, j))

    return leafs


def detect_branch(img):
    (h, w) = img.shape
    kernel = np.array(
        [[10, 10, 10, 10, 1, 1],
         [10, 10, 10, 10, 0, 1],
         [10, 10, 10, 10, 1, 1]], dtype=np.uint8
    )
    branches = list()

    for j in range(w):
        for i in range(h):
            if img[i, j] == 1 and 42 <= multi_mat(img, kernel, (i-1, j-4)) <= 45:
                branches.append((i, j))

    # remove points who's distance is small
    return remove_near(branches)


def detect_keypoints(img):
    root = detect_root(img)
    branches = detect_branch(img)
    leafs = detect_leaf(img, root)

    return root, branches, leafs
