import marshal
import math
import os

import matplotlib
import numpy as np
import skimage
from matplotlib import pyplot as plt
from skimage.io import imshow

from output import savefig

import cv2

img1 = skimage.io.imread('Img001.png')
img2 = skimage.io.imread('Img002.png')
img9 = skimage.io.imread('Img009.png')


def get_patch_of_img(img, x, y, patch_size):
    half_patch = math.floor(patch_size/2)
    return img[y - half_patch:y + half_patch + 1, x - half_patch:x + half_patch + 1]


def generate_left_to_right(kp_left, kp_right, img_left, img_right, N):
    # left to right
    kp_left_idx_of_matches_in_kp_right = np.repeat(None, (len(kp_left)))
    dissimilarities = np.repeat(None, (len(kp_left)))
    
    count_second_lowest_ratio_removed = 0

    for kp_left_index, keyp1 in enumerate(kp_left):
        lowest = None
        second_lowest = None
        x1 = keyp1[0]
        y1 = keyp1[1]
        patch1 = get_patch_of_img(img_left, x1, y1, N)
        patch_flat1 = patch1.reshape(-1)
        for keyp2_nr, keyp2 in enumerate(kp_right):
            x2 = keyp2[0]
            y2 = keyp2[1]
            patch2 = get_patch_of_img(img_right, x2, y2, N)
            patch_flat2 = patch2.reshape(-1)
            # we drop patches that won't be of the same size due 
            # to it being near the perimeter
            if (patch_flat1.shape == patch_flat2.shape):
                dissimilarity = sum(np.power(patch_flat1 - patch_flat2, 2))
                if lowest is None or dissimilarity < lowest:
                    second_lowest = lowest
                    lowest = dissimilarity
                    kp_left_idx_of_matches_in_kp_right[kp_left_index] = keyp2_nr
            # The ratio of dissimilarity between the best and the second best match is
            # too close to 1 (say above 0.7).
        if lowest is not None and second_lowest is not None:
            if lowest/second_lowest > 0.7:
                kp_left_idx_of_matches_in_kp_right[kp_left_index] = None
                count_second_lowest_ratio_removed += 1
        
        if kp_left_idx_of_matches_in_kp_right[kp_left_index] is not None:
            dissimilarities[kp_left_index] = lowest
                    
    return kp_left_idx_of_matches_in_kp_right, count_second_lowest_ratio_removed, dissimilarities



def generate_right_to_left(kp_left, kp_right, img_left, img_right, N):
    # right to left
    kp_right_idx_of_matches_in_kp_left = np.repeat(None, (len(kp_right)))
    
    count_second_lowest_ratio_removed = 0

    for kp_right_index, keyp2 in enumerate(kp_right):
        lowest = None
        second_lowest = None
        x2 = keyp2[0]
        y2 = keyp2[1]
        patch2 = get_patch_of_img(img_right, x2, y2, N)
        patch_flat2 = patch2.reshape(-1,)
        for keyp1_nr, keyp1 in enumerate(kp_left):
            x1 = keyp1[0]
            y1 = keyp1[1]
            patch1 = get_patch_of_img(img_left, x1, y1, N)
            patch_flat1 = patch1.reshape(-1,)
            # we drop patches that won't be of the same size due 
            # to it being near the perimeter
            if (patch_flat1.shape == patch_flat2.shape):
                dissimilarity = sum(np.power(patch_flat1 - patch_flat2, 2))
                if lowest is None or dissimilarity < lowest:
                    second_lowest = lowest
                    lowest = dissimilarity
                    kp_right_idx_of_matches_in_kp_left[kp_right_index] = keyp1_nr
                #  The ratio of dissimilarity between the best and the second best match is
        # too close to 1 (say above 0.7).
        if lowest is not None and second_lowest is not None:
            if lowest/second_lowest > 0.7:
                kp_right_idx_of_matches_in_kp_left[kp_right_index] = None
                count_second_lowest_ratio_removed += 1
    
    return kp_right_idx_of_matches_in_kp_left, count_second_lowest_ratio_removed



def get_index_of_match_in_right_img_from_index_in_kp_left(kp1_idx_of_matches_in_kp2, nth_kp):
    return kp1_idx_of_matches_in_kp2[nth_kp]



def get_index_of_match_in_left_img_from_index_in_kp_right(kp2_idx_of_matches_in_kp1, nth_kp):
    return kp2_idx_of_matches_in_kp1[nth_kp]



def verify_left_right_right_left_matching(kp_left, kp_right, kp_left_idx_of_matches_in_kp_right, kp_right_idx_of_matches_in_kp_left, dissimilarities):
    count = 0
    for i in range(len(kp_left)):
        left_kp = kp_left[i]
        right_img_match_index = get_index_of_match_in_right_img_from_index_in_kp_left(kp_left_idx_of_matches_in_kp_right, i)
        if right_img_match_index is not None:
            # this should be = i
            j = get_index_of_match_in_left_img_from_index_in_kp_right(kp_right_idx_of_matches_in_kp_left, right_img_match_index)
            if i != j:
                count += 1
                kp_left_idx_of_matches_in_kp_right[i] = None
                dissimilarities[i] = None
                kp_right_idx_of_matches_in_kp_left[right_img_match_index] = None
                
    return (kp_left_idx_of_matches_in_kp_right, kp_right_idx_of_matches_in_kp_left, count, dissimilarities)



def count_Nones(a_list):
    count = 0

    for item in a_list:
        if item == None:
            count += 1

    return count



def generate_totalimg(img_left, img_right, kp_left, kp_right, kp_left_idx_of_matches_in_kp_right, kp_right_idx_of_matches_in_kp_left, filename):
    extra_size = 50
    x_is_X = False
    
    # the amount of Y, the amount of X = (600, 800)
    # the amount of rows, the amount of columns
    count = 0
    totalimgY = img_left.shape[0]
    totalimgX = img_left.shape[1] + extra_size + 1 + img_right.shape[1]
    totalimg = np.zeros((totalimgY, totalimgX), dtype=np.float32)

    totalimg[0:img_left.shape[0], 0:img_left.shape[1]] = img_left
    totalimg[0:img_right.shape[0], img_left.shape[1] + extra_size:img_left.shape[1] + extra_size + img_right.shape[1]] = img_right

    fig, ax = plt.subplots(frameon=False)
    ax.imshow(totalimg, cmap="gray")
    ax.set_axis_off()
    
    for left_match_index, right_match_index in enumerate(kp_left_idx_of_matches_in_kp_right):
        if right_match_index is not None:
            left_match_pt = kp_left[left_match_index]
            right_match_pt = kp_right[right_match_index]
            y1 = left_match_pt[0]
            x1 = left_match_pt[1] # is the horizontal value
            y2 = right_match_pt[0]
            x2 = right_match_pt[1]
#             print(x1, y1)
#             print(x2, y2)
#             print()
            # x1, x2 - y1, y2 , y is the Y axis (vertical), x is the X axis (horizontal)
#             cv2.line(totalimg, (int(x1), int(y1)), (int(x2) + extra_size + img_left.shape[1], int(y2)), (200, 50, 50), 1)
            ax.plot([x1, x2  + extra_size + img_left.shape[1]], [y1, y2], linewidth=0.5)
            
    savefig(fig, filename)

    return



def save_dissimilarities(dissimilarities, filename):
    filename += "-dissimilarities.npy"
    print("Saving dissimilarities to %s" %filename)
    np.save(open(filename, "wb"), dissimilarities)
    return



def main():
    print("Running exercise 2...")
    kp1 = marshal.load(open("img1kp.bin", "rb"))
    kp2 = marshal.load(open("img2kp.bin", "rb"))
    kp9 = marshal.load(open("img9kp.bin", "rb"))
    # make all floats ints
    ms = [kp1, kp2, kp9]
    for index_of_m, m in enumerate(ms):
        for index_of_entry, entry in enumerate(m):
            for index_of_nr, nr in enumerate(entry):
                entry[index_of_nr] = int(nr)
            m[index_of_entry] = entry[:2]

    # keep track which combination for file name
    comb1 = True
    patch_sizes = [5, 7, 9, 13]
#     patch_sizes = [13]
    combinations = [(img1, img2), (img1, img9)]
#     combinations = [(img1, img2)]

    for combination in combinations:
        for patch_size in patch_sizes:

            if comb1:
                print("Comparing img001 w/ img002")
            else:
                print("Comparing img001 w/ img009")
            print("Patch size is %s" %patch_size)

            img_left = combination[0]
            img_right = combination[1]

            kp_left = kp1
            kp_right = None
            if comb1:
                kp_right = kp2
            else:
                kp_right = kp9

            kp_left_idx_of_matches_in_right, count_second_lowest_ratio_removed_left, dissimilarities = generate_left_to_right(kp_left, kp_right, img_left, img_right, patch_size)
            print("%s original keypoints found in left image. %s removed when applying the second lowest/lowest ratio filter" %(len(kp_left_idx_of_matches_in_right), count_second_lowest_ratio_removed_left))

            kp_right_idx_of_matches_in_left, count_second_lowest_ratio_removed_right = generate_right_to_left(kp_left, kp_right, img_left, img_right, patch_size)
            print("%s original keypoints found in right image. %s removed when applying the second lowest/lowest ratio filter" %(len(kp_right_idx_of_matches_in_left), count_second_lowest_ratio_removed_right))

            kp_left_idx_of_matches_in_right, kp_right_idx_of_matches_in_left, nr_removed, dissimilarities = verify_left_right_right_left_matching(kp_left, kp_right, kp_left_idx_of_matches_in_right, kp_right_idx_of_matches_in_left, dissimilarities)
            print("%s were removed when verifying left-right corresponds to right-left" %(nr_removed))

            filename = ""
            if comb1:
                filename = "Result_001-to-002"
            else:
                filename = "Result_001-to-009"
            filename += "-patch-size-%s" %patch_size
            filename += ".png"
            print("Saving image: %s" %(filename))

            generate_totalimg(img_left, img_right, kp_left, kp_right, kp_left_idx_of_matches_in_right, kp_right_idx_of_matches_in_left, filename)

            save_dissimilarities(dissimilarities, filename)

        # the other combination
        comb1 = False


def dissimilarity_analysis():
    comb1 = True
    patch_sizes = [5, 7, 9, 13]
    #     patch_sizes = [13]
    combinations = [(img1, img2), (img1, img9)]
    
    for combination in combinations:
        for patch_size in patch_sizes:
            filename = ""
            if comb1:
                filename = "Result_001-to-002"
            else:
                filename = "Result_001-to-009"
            filename += "-patch-size-%s" %patch_size
            filename += ".png"
            filename += "-dissimilarities.npy"
#             print("Analyzing dissimilarities for %s" %filename)

            diss = np.load(open(filename, "rb"))
            diss_vals = diss[diss != np.array(None)]
            std_dev = np.std(diss_vals)
            mean = diss_vals.mean()
            print("For %s:\n - std. dev. = %s\n - mean = %s" %(filename, std_dev, mean))
                
        comb1 = False
