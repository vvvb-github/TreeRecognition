from utils import binary_image, convolution
import numpy as np
import queue
from skimage.morphology import skeletonize


kernel = np.array(
    [[1, 1, 1],
     [1, 0, 1],
     [1, 1, 1]], dtype=np.uint8
)


def bfs(img, id, x, y):
    (h, w) = img.shape
    q = queue.Queue()
    q.put_nowait((x, y))
    img[x, y] = id
    mx = [-1, -1, -1, 0, 0, 1, 1, 1]
    my = [-1, 0, 1, -1, 1, -1, 0, 1]
    size = 0

    while not q.empty():
        (i, j) = q.get_nowait()
        size += 1
        for k in range(8):
            ni, nj = i+mx[k], j+my[k]
            if ni >= 0 and ni < h and nj >= 0 and nj < w:
                if img[ni, nj] == 255:
                    img[ni, nj] = id
                    q.put_nowait((ni, nj))

    return size


def remove_components(img):
    (h, w) = img.shape
    components = dict()
    id = 1

    for i in range(h):
        for j in range(w):
            if img[i, j] == 255:
                components[id] = bfs(img, id, i, j)
                id += 1

    id = 1
    biggest_size = components[id]
    for i in components.keys():
        if biggest_size < components[i]:
            id = i
            biggest_size = components[id]

    for i in range(h):
        for j in range(w):
            img[i, j] = 255 if img[i, j] == id else 0

    return img


def is_tail(img, position):
    return convolution(img, position, kernel) == 1


def is_branch(img, position):
    return convolution(img, position, kernel) >= 3


def blur(img, threshold=10):
    (h, w) = img.shape
    mx = [-1, -1, -1, 0, 0, 1, 1, 1]
    my = [-1, 0, 1, -1, 1, -1, 0, 1]

    for i in range(h):
        for j in range(w):
            if img[i, j] > 0 and is_tail(img, (i, j)):
                size = 0
                points = list()
                cur = (i, j)
                parent = (0, 0)
                while not is_branch(img, cur):
                    points.append(cur)
                    (x, y) = cur
                    for k in range(8):
                        nx, ny = x+mx[k], y+my[k]
                        if (nx, ny) != parent and img[nx, ny] > 0:
                            parent = cur
                            cur = (nx, ny)
                            break
                    if len(points) > threshold:
                        break
                if len(points) <= threshold:
                    for p in points:
                        img[p[0], p[1]] = 0

    return img


def skeleton(img, binary=True):
    # skeletonize
    img = binary_image(img, 255, 1)
    img = skeletonize(img).astype(np.uint8)
    img = binary_image(img, 1, 255)

    # remove other component on image
    remove_components(img)

    # blur
    img = binary_image(img, 255, 1)
    img = blur(img, 20)
    if binary:
        img = binary_image(img, 1, 255)

    return img
