import matplotlib
import numpy as np
import skimage
from matplotlib import pyplot as plt
from skimage.io import imshow
from skimage.measure import compare_ssim

import cv2
from output import savefig

matplotlib.rcParams['figure.figsize'] = [20, 20]


def gaussian_filter(img):
    sigmas = [1, 2, 4, 8]

    fig, axs = plt.subplots(len(sigmas), 3, tight_layout=True)

    for i, sigma in enumerate([1, 2, 4, 8]):
        # Notice that in calling the `GaussianBlur` method I provide a kernel
        # size of `(0,0)`. In this situation the kernel size will be chosen
        # based on the provided $\sigma$ value.
        img_gaussian_blur = cv2.GaussianBlur(
            img, (0, 0), sigmaX=sigma, sigmaY=sigma)

        axs[i, 0].imshow(img, cmap='gray')
        axs[i, 0].set_title("Original")
        axs[i, 1].imshow(img_gaussian_blur, cmap='gray')
        axs[i, 1].set_title("Blur - sigma: %s" % sigma)

        score, diff = compare_ssim(img, img_gaussian_blur, full=True)
        axs[i, 2].imshow(diff, cmap='gray')
        axs[i, 2].set_title("Similarity to original: %s%%" % (score * 100))

    filename = "gaussian.png"
    savefig(fig, filename)
