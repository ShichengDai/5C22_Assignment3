import numpy as np

def makewindow(y, x , N, m):
    h, w, c = m.shape
    halfN = N // 2
    n1 = halfN
    n2 = N - halfN - 1
    window = np.full((N, N, c), np.nan)
    xmin = max(1, x - n1)
    xmax = min(w, x + n2)
    ymin = max(1, y - n1)
    ymax = min(h, y + n2)
    pxmin = halfN - (x - xmin) + 1
    pxmax = halfN + (xmax - x) + 1
    pymin = halfN - (y - ymin) + 1
    pymax = halfN + (ymax - y) + 1
    window[pymin:pymax, pxmin:pxmax, :] = m[ymin:ymax, xmin:xmax, :]
    return window