import matplotlib
import numpy as np
import skimage
from matplotlib import pyplot as plt
from skimage.io import imshow
from skimage.measure import compare_ssim

import cv2
from output import savefig

matplotlib.rcParams['figure.figsize'] = [20, 20]


def canny(lenna):
    # first
    fig, axs = plt.subplots(nrows=1, ncols=2, tight_layout=True)
    fig.set_tight_layout(False)

    minVal = 100
    maxVal = 200
    lenna_edges_1 = cv2.Canny(lenna, minVal, maxVal)
    axs[0].imshow(lenna, cmap='gray')
    axs[0].set_title("Original")
    axs[1].imshow(lenna_edges_1, cmap='gray')
    axs[1].set_title("Canny. minVal = %s. maxVal = %s" % (minVal, maxVal))

    savefig(fig, "canny1.png")

    # second
    fig, axs = plt.subplots(nrows=1, ncols=2, tight_layout=True)
    fig.set_tight_layout(False)

    minVal = 70
    maxVal = 200
    lenna_edges_1 = cv2.Canny(lenna, minVal, maxVal)
    axs[0].imshow(lenna, cmap='gray')
    axs[0].set_title("Original")
    axs[1].imshow(lenna_edges_1, cmap='gray')
    axs[1].set_title("Canny. minVal = %s. maxVal = %s" % (minVal, maxVal))

    savefig(fig, "canny2.png")

    # third
    fig, axs = plt.subplots(nrows=1, ncols=2, tight_layout=True)
    fig.set_tight_layout(False)

    minVal = 70
    maxVal = 300
    lenna_edges_1 = cv2.Canny(lenna, minVal, maxVal)
    axs[0].imshow(lenna, cmap='gray')
    axs[0].set_title("Original")
    axs[1].imshow(lenna_edges_1, cmap='gray')
    axs[1].set_title("Canny. minVal = %s. maxVal = %s" % (minVal, maxVal))

    savefig(fig, "canny3.png")

    # fourth
    fig, axs = plt.subplots(nrows=1, ncols=2, tight_layout=True)
    fig.set_tight_layout(False)

    minVal = 100
    maxVal = 200
    lenna_edges_1 = cv2.Canny(lenna, minVal, maxVal, apertureSize=5)
    axs[0].imshow(lenna, cmap='gray')
    axs[0].set_title("Original")
    axs[1].imshow(lenna_edges_1, cmap='gray')
    axs[1].set_title(
        "Canny. minVal = %s. \
        maxVal = %s. \
        apperture = 5" % (minVal, maxVal)
    )

    savefig(fig, "canny4.png")

    # fifth
    fig, axs = plt.subplots(nrows=1, ncols=2, tight_layout=True)
    fig.set_tight_layout(False)

    minVal = 900
    maxVal = 1500
    lenna_edges_1 = cv2.Canny(lenna, minVal, maxVal, apertureSize=5)
    axs[0].imshow(lenna, cmap='gray')
    axs[0].set_title("Original")
    axs[1].imshow(lenna_edges_1, cmap='gray')
    axs[1].set_title(
        "Canny. minVal = %s. \
    maxVal = %s. \
    apperture = 5" % (minVal, maxVal)
    )

    savefig(fig, "canny5.png")
