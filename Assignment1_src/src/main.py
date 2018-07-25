import cv2
from gaussian_filter import gaussian_filter
from sobel import sobel
from laplace import laplace
from canny import canny

# Using `0` reads the image as grayscale.
img = cv2.imread("lenna.jpg", 0)

# we run each section
# check output for which files are produced
gaussian_filter(img)
sobel(img)
laplace(img)
canny(img)
