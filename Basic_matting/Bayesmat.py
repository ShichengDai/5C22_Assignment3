import numpy as np
import cv2
from scipy.ndimage import filters
from skimage.morphology import square
from numpy.linalg import inv, det, norm
from math import sqrt, pi
from block import makewindow




def Bayesmat(im, trimap, N, sigma):
    # make masks for fg bg and unkreg
    bgmask = trimap == 0 
    fgmask = trimap == 1 
    unkmask = np.logical_not(bgmask | fgmask)

    # initialize F B and Alpha
    F = np.copy(im)
    F[~np.repeat(fgmask[:, :, np.newaxis], 3, axis=2)] = 0
    B = np.copy(im)
    B[~np.repeat(bgmask[:, :, np.newaxis], 3, axis=2)] = 0
    alpha = np.zeros(trimap.shape)
    alpha[fgmask] = 1
    alpha[unkmask] = np.nan
    nUnknown = np.sum(unkmask)

    # make gaussian parameter g
    g = filters.gaussian_filter(np.zeros((N, N)), sigma)
    # normalize the parameter to make sure p will not change the image luminance
    g = g / np.max(g)
    # square structuring element for eroding the unknown region(s)
    se = np.ones((3, 3))

    # set a threshold for the minimum valid pixels in the neighbourhood
    # change the value here if the loop stucks
    Nthres = 10

    n = 1
    unkreg = unkmask
    while n < nUnknown:

        # get unknown pixels to process at this iteration
        unkreg = cv2.erode(unkreg, se, iterations=1)
        unkpixels = np.logical_and(np.logical_not(unkreg), unkmask)
        Y, X = np.nonzero(unkpixels)

        for i in range(len(Y)):

            # take current pixel
            x = X[i]
            y = Y[i]
            c = im[y, x, :].reshape((3, 1))

            # take surrounding alpha values
            a = makewindow(y, x, N, alpha)

            # take surrounding foreground pixels
            f_pixels = makewindow(y, x, N, F)
            f_weights = (a ** 2) * g
            f_pixels = f_pixels[f_weights > 0, :]
            f_weights = f_weights[f_weights > 0]

            # take surrounding background pixels
            b_pixels = makewindow(y, x, N, B)
            b_weights = ((1 - a) ** 2) * g
            b_pixels = b_pixels[b_weights > 0, :]
            b_weights = b_weights[b_weights > 0]

            if len(f_weights) < Nthres or len(b_weights) < Nthres:
             continue
            mu_f, sigma_f = clustFunc(f_pixels, f_weights)
            mu_b, sigma_b = clustFunc(b_pixels, b_weights)

            alpha_init = np.nanmean(a.ravel())

            f, b, alphaT = solve(mu_f, sigma_f, mu_b, sigma_b, p, 0.01, alpha_init, 50, 1e-6)

            F[y, x] = f.ravel()
            B[y, x] = b.ravel()
            alpha[y, x] = alphaT
            unkmask[y, x] = 0
    return alpha