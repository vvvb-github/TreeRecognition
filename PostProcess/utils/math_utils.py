def convolution(img, position, kernel):
    (x, y) = position
    kernel_size = kernel.shape[0]//2
    res = 0

    for i in range(3):
        for j in range(3):
            res += kernel[i, j]*img[x-kernel_size+i, y-kernel_size+j]

    return res


def multi_mat(img, mat, border):
    (lx, ly) = border
    (r, c) = mat.shape
    res = 0

    for i in range(r):
        for j in range(c):
            res += mat[i, j]*img[lx+i, ly+j]

    return res


def distance(p1, p2, near=8):
    (x1, y1) = p1
    (x2, y2) = p2
    if near == 8:
        return max(abs(x1-x2), abs(y1-y2))
    elif near == 4:
        return abs(x1-x2)+abs(y1-y2)
