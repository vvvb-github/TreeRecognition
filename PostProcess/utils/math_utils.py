def convolution(img, position, kernel):
    (x, y) = position
    n = kernel.shape[0]
    half = n//2
    res = 0

    for i in range(n):
        res += kernel[0, i]*img[x-half, y-half+i] + \
            kernel[n-1, i]*img[x+half, y-half+i]
    for i in range(n-2):
        res += kernel[i+1, 0]*img[x-half+i+1, y-half] + \
            kernel[i+1, n-1]*img[x-half+i+1, y+half]

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
