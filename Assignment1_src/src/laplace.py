import matplotlib
import numpy as np
import skimage
from matplotlib import pyplot as plt
from skimage.io import imshow
from skimage.measure import compare_ssim

import cv2
from output import savefig

matplotlib.rcParams['figure.figsize'] = [20, 20]


def laplace(img):
    lenna = skimage.img_as_float(img)
    ksize = 5  # for the internal Sobel operative.
    sigmas = [1, 2, 4, 8]

    fig, axs = plt.subplots(len(sigmas) + 1, 2, tight_layout=True)

    # original
    lenna_laplace = cv2.Laplacian(lenna, ksize=ksize, ddepth=-1)

    axs[0, 0].imshow(lenna, cmap='gray')
    axs[0, 0].set_title("Original")

    axs[0, 1].imshow(lenna_laplace, cmap='gray')
    axs[0, 1].set_title("Laplace")

    for i, sigma in enumerate(sigmas):
        lenna_blurred = cv2.GaussianBlur(lenna, (0, 0),
                                         sigmaX=sigma, sigmaY=sigma)
        lenna_laplace = cv2.Laplacian(lenna_blurred, ksize=ksize, ddepth=-1)

        axs[i + 1, 0].imshow(lenna_blurred, cmap='gray')
        axs[i + 1, 0].set_title("Blur - sigma: %s" % (sigma))

        axs[i + 1, 1].imshow(lenna_laplace, cmap='gray')
        axs[i + 1, 1].set_title("Laplace")

    savefig(fig, "laplace.png")
