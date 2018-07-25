import matplotlib
import numpy as np
import skimage
from matplotlib import pyplot as plt
from skimage.io import imshow
from skimage.measure import compare_ssim

import cv2

from output import savefig

matplotlib.rcParams['figure.figsize'] = [20, 20]


def get_sobel(img, kernel=None):
    sobelx = None
    sobely = None
    if kernel is not None:
        sobelx = cv2.Sobel(img, ddepth=-1, dx=1, dy=0, ksize=kernel)
        sobely = cv2.Sobel(img, ddepth=-1, dx=0, dy=1, ksize=kernel)
    else:
        sobelx = cv2.Sobel(img, ddepth=-1, dx=1, dy=0)
        sobely = cv2.Sobel(img, ddepth=-1, dx=0, dy=1)
    return cv2.sqrt(sobelx**2 + sobely**2)


def sobel(img):
    # Change datatype to float, to match `Sobel` method.
    lenna = skimage.img_as_float(img)

    # first image
    sigmas = [1, 2, 4, 8]

    fig, axs = plt.subplots(len(sigmas), 3, tight_layout=True)

    prev_img_gradient = None

    for i, sigma in enumerate(sigmas):
        lenna_blurred = cv2.GaussianBlur(
            lenna,
            (0, 0),
            sigmaX=sigma,
            sigmaY=sigma
        )
        lenna_gradients = get_sobel(lenna_blurred)

        axs[i, 0].imshow(lenna, cmap='gray')
        axs[i, 0].set_title("Original")

        axs[i, 1].imshow(lenna_blurred, cmap='gray')
        axs[i, 1].set_title("Blur - sigma: %s" % (sigma))

        axs[i, 2].imshow(lenna_gradients, cmap='gray')
        axs[i, 2].set_title("Gradient - default kernel")

    savefig(fig, "sobel.png")
