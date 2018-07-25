"""main"""
import math
import os

import matplotlib.pyplot as plt
import numpy as np

import cv2
import testing
from output import savefig


def get_patch_of_img(img, x_val, y_val, patch_size):
    """# Patch size function to get a local window of pixel values."""
    half_patch = math.floor(patch_size / 2)
    patch = img[y_val - half_patch:y_val + half_patch +
                1, x_val - half_patch:x_val + half_patch + 1]
    return patch


# Function below calculates position of matches
#
# Note: We only need one function for left-to-right and right-to-left matching, this helps cut down on code. We only need to simply swtich the image inputs. For example, for left-to-right matching image1 will be img_left while image2 will be img_right. Then for right-to-left matching we will use the same function again where image2 will be img_left and image1 wil be img_right.
def find_matches(img_left, img_right, patch_size, search_window_size=12, prev_pyramid_matches=None):
    # below we create a new image to fill that will be the same size
    matches = np.zeros((img_left.shape[0], img_left.shape[1]))

    # TODO unused atm. Maybe in the future?
    # greatest_distance = 0

    # Start with left input image
    # x and y should ignore window size (N) at the margin
    half_padding = math.floor(patch_size / 2)
    # y is vertical position
    for y1 in range(half_padding, img_left.shape[0] - half_padding):
        # x is horizontal position
        for x1 in range(half_padding, img_left.shape[1] - half_padding):
            patch_left = get_patch_of_img(img_left, x1, y1, patch_size)
            patch_flat_left = patch_left.reshape(-1)
            # The above finds patches around pixels at coordinates x1,y1 within the left image.

            # This is empty at first, we fill it in in the loop below.
            best_match = None
            # TODO this unused atm. Could be used in the future
            # second_best_match = None
            best_match_index = 0

            # Initialize search window to the full image width
            range_start = half_padding
            range_end = img_left.shape[1] - half_padding

            # Check for a match in previous pyramid
            if prev_pyramid_matches is not None:
                prev_pyr_vert_pos = math.floor(y1 / 2)
                prev_pyr_hor_pos = math.floor(x1 / 2)
                approximate_match_coordinates = 2 * \
                    int(prev_pyramid_matches[prev_pyr_vert_pos]
                        [prev_pyr_hor_pos])
                range_start = approximate_match_coordinates - search_window_size
                if range_start < half_padding:
                    range_start = half_padding
                range_end = approximate_match_coordinates + search_window_size
                if range_end > img_left.shape[1] - half_padding:
                    range_end = img_left.shape[1] - half_padding

            # If we do not have a previous pyramid use a search window equal to one third of the image width
            else:
                # Double search window size because we are not searching near an approximate match
                range_start = x1 - int(search_window_size * 2)
                range_end = x1 + int(search_window_size * 2)
                if (range_start < half_padding):
                    range_start = half_padding
                if (range_end > img_left.shape[1] - half_padding):
                    range_end = img_left.shape[1] - half_padding

            # Start with right image
            # only need x coord because we know matching pixel has same y coord as left image.
            for x2 in range(range_start, range_end):
                # patch for each pixel in row for right image
                patch_right = get_patch_of_img(img_right, x2, y1, patch_size)
                patch_flat_right = patch_right.reshape(-1)

                # Search for similarity between images
                if (patch_flat_left.shape == patch_flat_right.shape and len(patch_flat_left) != 0):
                    # similarity = sum(np.power(patch_flat_left - patch_flat_right, 2)) #Sum of squared differences
                    # Normalized Cross Correlation
                    similarity = sum(cv2.matchTemplate(
                        patch_left, patch_right, cv2.TM_CCOEFF_NORMED))
                    # similarity = match_template(patch_left, patch_right)[0][0] #Normalized Cross Correlation skimage (SLOW)
                    if best_match is None or similarity > best_match:  # Must be < for SSD and > for NCC
                        # TODO unused atm. Maybe in the future.
                        # second_best_match = best_match
                        best_match = similarity
                        best_match_index = x2

            matches[y1][x1] = best_match_index
    return(matches)


# Function for two way matching
# Reduce number of false matches and accept candidate matches that agree. See paragraph 2 on page 2 of assignment.
# We do NOT need this for computing a disparity map but I think it is useful for doing so. It is recommended in the assignment.
# This function helps with areas of occlusion
def check_for_two_way_matches(l_to_r, r_to_l):
    two_way_matches = np.array(l_to_r, copy=True)

    for y1 in range(l_to_r.shape[0]):
        for x1 in range(l_to_r.shape[1]):
            x2 = int(l_to_r[y1][x1])
            if (r_to_l[y1][x2] != x1):
                two_way_matches[y1][x1] = -1
    return two_way_matches


# Function for generating the disparity map, finally!


def generate_disparity_map(matches, patch_size):
    padding = math.floor(patch_size / 2)
    # copy to avoid modifying original
    disparity_map = np.array(matches, copy=True)

    # calculate greatest distance so that we can normalize our distance values
    greatest_distance = 0
    for y in range(padding, matches.shape[0] - padding):
        for x in range(padding, matches.shape[1] - padding):
            # get distance in pixels to match
            distance = np.absolute(matches[y][x] - x)
            if distance > greatest_distance:
                greatest_distance = distance
    print("Greatest distance = ", greatest_distance)

    # Assign color to disparity map based on distance between matches
    for y in range(padding, disparity_map.shape[0] - padding):
        for x in range(padding, disparity_map.shape[1] - padding):
            color = 0
            # get distance in pixels to match
            distance = np.absolute(matches[y][x] - x)
            if (disparity_map[y][x] != -1):  # check bad matches
                normalized_distance = distance / greatest_distance
                color = normalized_distance * 255.0
            disparity_map[y][x] = int(color)
            if (y < padding or y >= disparity_map.shape[0] - padding):
                disparity_map[y][x] = 0
            if (x < padding or x >= disparity_map.shape[1] - padding):
                disparity_map[y][x] = 0
    return disparity_map


# Local mean function
# The idea is to replace deviating disparity values within a patch of the disparity map with that patch's local average.
def local_mean_disparity_value(original, patchsize, threshold, padding=0):
    image = np.array(original, copy=True)  # copy to avoid modifying original

    # y is vertical position
    for y1 in range(padding, image.shape[0] - padding):
        # x is horizontal position
        for x1 in range(padding, image.shape[1] - padding):

            # Take patches again, "look" inside patch
            patch_disparity_map = get_patch_of_img(image, x1, y1, patchsize)
            patch_flat = patch_disparity_map.reshape(-1)

            # Remove black pixels from average (failed matches)
            patch_flat = [x for x in patch_flat if x > 0]
            mean_of_patch = np.mean(patch_flat)  # Taking the mean

            # NAN values come from perimeter sampling when we get patches. Now we make them zero.
            if math.isnan(mean_of_patch):
                mean_of_patch = 0

            # Replace deviating disparity values with local average in patch if difference between average and disparity is greater than the threshold
            if np.absolute(mean_of_patch - image[y1][x1]) > threshold or image[y1][x1] == 0 or mean_of_patch == 0:
                image[y1][x1] = mean_of_patch

    return image


def construct_pyramid_cv2(original_image, number_images):
    next_image = original_image.copy()
    pyramid = [next_image]
    for _ in range(0, number_images):
        next_image = cv2.pyrDown(next_image)
        pyramid.append(next_image)
    return pyramid


def pyramid_matching(left_pyramid, right_pyramid, patch_size, window_size):
    stereo_matches = None
    # the `patch_size_to_use` and `patch_size_to_use_in_loop` are not used. KEEP THEM???
    # patch_size_to_use = 7
    # Go through the pyramid from smallest to largest (largest is at index 0)
    for image_index in range(len(left_pyramid) - 1, -1, -1):
        # patch_size_to_use_in_loop = int(patch_size_to_use / (image_index + 1))
        stereo_matches = find_matches(
            left_pyramid[image_index],
            right_pyramid[image_index],
            patch_size,
            window_size,
            stereo_matches
        )
    return stereo_matches


def create_figure_pyramid(pyramid):
    rows, cols = pyramid[0].shape
    composite_image = np.zeros((rows, cols + cols // 2), dtype=np.double)

    composite_image[:rows, :cols] = pyramid[0]

    i_row = 0
    for p in pyramid[1:]:
        n_rows, n_cols = p.shape[:2]
        composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p
        i_row += n_rows

    fig, ax = plt.subplots()
    ax.imshow(composite_image, cmap="gray")
    ax.set_axis_off()
    return fig


def get_datasets():
    tsukuba_disparity_map = cv2.imread(
        'tsukuba/truedisp.row3.col3.pgm')
    tsukuba_disparity_map = cv2.cvtColor(
        tsukuba_disparity_map, cv2.COLOR_BGR2RGB)

    tsukuba_1 = cv2.imread('tsukuba/scene1.row3.col1.ppm')
    tsukuba_1 = cv2.cvtColor(tsukuba_1, cv2.COLOR_BGR2GRAY)

    tsukuba_3 = cv2.imread('tsukuba/scene1.row3.col3.ppm')
    tsukuba_3 = cv2.cvtColor(tsukuba_3, cv2.COLOR_BGR2GRAY)

    venus_disparity_map = cv2.imread(
        'venus/disp2.pgm')
    venus_disparity_map = cv2.cvtColor(
        venus_disparity_map, cv2.COLOR_BGR2RGB)

    venus_2 = cv2.imread('venus/im2.ppm')
    venus_2 = cv2.cvtColor(venus_2, cv2.COLOR_BGR2GRAY)

    venus_6 = cv2.imread('venus/im6.ppm')
    venus_6 = cv2.cvtColor(venus_6, cv2.COLOR_BGR2GRAY)

    map_disparity_map = cv2.imread(
        'map/disp0.pgm')
    map_disparity_map = cv2.cvtColor(
        map_disparity_map, cv2.COLOR_BGR2RGB)

    map_0 = cv2.imread('map/im0.pgm')
    map_0 = cv2.cvtColor(map_0, cv2.COLOR_BGR2GRAY)

    map_1 = cv2.imread('map/im1.pgm')
    map_1 = cv2.cvtColor(map_1, cv2.COLOR_BGR2GRAY)
    return [
        (tsukuba_1, tsukuba_3, tsukuba_disparity_map, "tsukuba"),
        (venus_2, venus_6, venus_disparity_map, "venus"),
        (map_0, map_1, map_disparity_map, "map")
    ]


def save_true_disp(true_disp, name):
    fig, ax = plt.subplots()
    ax.imshow(true_disp, cmap="gray")
    ax.set_axis_off()
    savefig(fig, "%s_true_disp.png" %name)


def save_ltr_initial(ltr_disparity, name, patch):
    fig, ax = plt.subplots()
    ax.imshow(ltr_disparity, cmap="gray")
    ax.set_axis_off()
    savefig(fig, "%s_patch_%s_initial_disparity.png" %(name, patch))


def save_disparity_final(mean_disparity_map, name, patch):
    fig, ax = plt.subplots()
    ax.imshow(mean_disparity_map, cmap="gray")
    ax.set_axis_off()
    savefig(fig, "%s_patch_%s_final_disparity.png" %(name, patch))


PATCH_SIZES = [5, 7, 11]

def main():
    if not os.path.exists("out"):
        os.mkdir("out")
    """# # Results and Load Images
    # Load images using OpenCV
    # For comparison"""
    for image_left, image_right, true_disp, name in get_datasets():
        print("Handling %s..." %name)
        # Create Pyramids
        left_pyramid = construct_pyramid_cv2(image_left, 3)
        right_pyramid = construct_pyramid_cv2(image_right, 3)

        savefig(create_figure_pyramid(left_pyramid),
                "%s_left_pyramid.png" % name)
        savefig(create_figure_pyramid(right_pyramid),
                "%s_right_pyramid.png" % name)
        save_true_disp(true_disp, name)

        for patch_size in PATCH_SIZES:
            print("Patch size = %s" %patch_size)

            # Find matches using pyramid
            search_window_size = 10
            left_to_right_matches = pyramid_matching(
                left_pyramid, right_pyramid, patch_size, search_window_size)
            right_to_left_matches = pyramid_matching(
                right_pyramid, left_pyramid, patch_size, search_window_size)

            ltr_disparity = generate_disparity_map(left_to_right_matches, patch_size)

            # I am ONLY plotting left to right. This step does NOT remove bad matches
            save_ltr_initial(ltr_disparity, name, patch_size)

            # Using function chechk_for_two_way_matches
            two_way_matches = check_for_two_way_matches(
                left_to_right_matches,
                right_to_left_matches
            )
            disparity = generate_disparity_map(two_way_matches, patch_size)

            # Using Local Mean with Two Way Matching
            half_patch = math.floor(patch_size / 2)
            mean_disparity_map = local_mean_disparity_value(disparity, patch_size, 25, half_patch)
            save_disparity_final(mean_disparity_map, name, patch_size)


main()
testing.main()
