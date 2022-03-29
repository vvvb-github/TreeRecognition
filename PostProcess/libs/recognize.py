import numpy as np


mx = [0, -1, -1, -1, 1, 1, 1, 0]
my = [-1, -1, 0, 1, -1, 0, 1, 1]
n = 1
res = ''


def dfs(img, tag, start):
    global n, res

    stack = list()
    stack.append(start)

    while len(stack) > 0:
        (x, y) = stack[-1]
        if img[x, y] == 0:
            stack.pop()
            if tag[x, y] > 0:
                res += '),'
        else:
            if img[x, y] == 2:
                res += '('
            elif img[x, y] == 3:
                res += '{},'.format(n)
                n += 1

            img[x, y] = 0
            for i in range(8):
                nx, ny = x+mx[i], y+my[i]
                if img[nx, ny] > 0:
                    stack.append((nx, ny))


def recognize(img, root, branches, leafs):
    global n, res

    n = 1
    res = ''
    tag = np.zeros(img.shape, dtype=np.uint8)

    for b in branches:
        x, y = b[0], b[1]
        img[x, y] = 2
        tag[x, y] = 1
    for l in leafs:
        x, y = l[0], l[1]
        img[x, y] = 3

    if type(root) != tuple:
        root = (root[0], root[1])

    dfs(img, tag, root)

    res = res.strip(',')
    res = res.replace(',)', ')')

    return res
