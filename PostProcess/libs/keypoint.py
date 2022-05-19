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


def detect_root(img, add=20):
    (h, w) = img.shape

    for j in range(w):
        for i in range(h):
            if img[i, j] == 1:
                for k in range(1, add+1):
                    img[i, j-k] = 1
                return (i, j-add)


def detect_leaf(img, root=None):
    (h, w) = img.shape
    kernel3 = np.array(
        [[1, 1, 1],
         [1, 0, 1],
         [1, 1, 1]], dtype=np.uint8
    )
    kernel5 = np.array(
        [[1, 1, 1, 1, 1],
         [1, 0, 0, 0, 1],
         [1, 0, 0, 0, 1],
         [1, 0, 0, 0, 1],
         [1, 1, 1, 1, 1]], dtype=np.uint8
    )
    leafs = list()

    for j in range(w):
        for i in range(h):
            if (i, j) == root:
                continue
            if img[i, j] == 1 and convolution(img, (i, j), kernel3) == 1:
                leafs.append((i, j))

    return leafs


def detect_branch(img, kernel_size):
    (h, w) = img.shape
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.uint8)
    for i in range(1, kernel_size):
        kernel[0, i] = 1
        kernel[kernel_size-1, i] = 1
    for i in range(1, kernel_size-1):
        kernel[i, 0] = 3*kernel_size
        kernel[i, kernel_size-1] = 1
    branches = list()

    for j in range(w):
        for i in range(h):
            if img[i, j] == 1 and 3*kernel_size+2 <= convolution(img, (i, j), kernel) <= 6*kernel_size:
                branches.append((i, j))

    # remove points who's distance is small
    return remove_near(branches)


def detect_keypoints(img, n=9):
    root = detect_root(img)
    branches = detect_branch(img, n)
    leafs = detect_leaf(img, root)

    return root, branches, leafs
