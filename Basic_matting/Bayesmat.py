import numpy as np
import cv2
from block import makewindow
from gaussian_filter import gaussian_filter
from clustering import clustFunc
from solve import solve
import imageio




def Bayesmat(im, trimap, N, sigma):
    # make masks for fg bg and unkreg
    im = im/255
    h, w, c = im.shape
    bgmask = trimap == 0 
    fgmask = trimap == 255 
    unkmask = True ^ np.logical_or(fgmask, bgmask)

    # initialize F B and Alpha
    F = np.copy(im)
    F[~np.repeat(fgmask[:, :, np.newaxis], 3, axis=2)] = 0
    B = np.copy(im)
    B[~np.repeat(bgmask[:, :, np.newaxis], 3, axis=2)] = 0
    alpha = np.zeros((h, w))
    alpha[fgmask] = 1
    alpha[unkmask] = np.nan
    nUnknown = np.sum(unkmask)

    # make gaussian parameter g
    g = gaussian_filter(N, sigma)
    # normalize the parameter to make sure p will not change the image luminance
    g = g / np.max(g)
    print(g.shape)
    # square structuring element for eroding the unknown region(s)
    se = np.ones((3, 3))

    # set a threshold for the minimum valid pixels in the neighbourhood
    # change the value here if the loop stucks
    Nthres = 5

    n = 1
    unkreg = unkmask
    while n < nUnknown:

        # get unknown pixels to process at this iteration
        unkreg = cv2.erode(unkreg.astype(np.uint8), se, iterations=1)
        unkpixels = np.logical_and(np.logical_not(unkreg), unkmask)
        Y, X = np.nonzero(unkpixels)

        for i in range(len(Y)):
            print(n, nUnknown)

            # take current pixel
            x = X[i]
            y = Y[i]
            c = im[y, x]

            # take surrounding alpha values
            a = makewindow(y, x, N, alpha[:, :, np.newaxis])[:, :, 0]

            # take surrounding foreground pixels
            f_pixels = makewindow(y, x, N, F)
            f_weights = ((a ** 2) * g).ravel()
            f_pixels = np.reshape(f_pixels, (N*N, 3))
            f_pixels = f_pixels[np.nan_to_num(f_weights) > 0, :]
            f_weights = f_weights[np.nan_to_num(f_weights) > 0]

            # take surrounding background pixels
            b_pixels = makewindow(y, x, N, B)
            b_weights = (((1 - a) ** 2) * g).ravel()
            b_pixels = np.reshape(b_pixels, (N*N, 3))
            b_pixels = b_pixels[np.nan_to_num(b_weights) > 0, :]
            b_weights = b_weights[np.nan_to_num(b_weights) > 0]

            if len(f_weights) < Nthres or len(b_weights) < Nthres:
              continue
            mu_f, sigma_f = clustFunc(f_pixels, f_weights)
            mu_b, sigma_b = clustFunc(b_pixels, b_weights)

            alpha_init = np.nanmean(a.ravel())

            f, b, alphaT = solve(mu_f, sigma_f, mu_b, sigma_b, c, 0.01, alpha_init, 50, 1e-6)

            F[y, x] = f.ravel()
            B[y, x] = b.ravel()
            alpha[y, x] = alphaT
            unkmask[y, x] = 0
            n += 1

    return alpha


def main():
    img = imageio.imread("Image_Source\Raw_Image\gandalf-input.png")[:, :, :3]
    trimap = imageio.imread("Image_Source\Trimap\gandalf-trimap.png", as_gray=True)
    alpha = Bayesmat(img, trimap, 25, 8)
    plt.title("Alpha matte")
    plt.imshow(alpha, cmap='gray')
    plt.show()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    main()