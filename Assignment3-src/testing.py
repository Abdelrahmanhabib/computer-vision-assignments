"""testing part"""

import cv2

def get_datasets():
    """load dataset"""
    tsukuba_final = cv2.imread(
        "out/tsukuba_patch_11_final_disparity.png",
        0
    )
    tsukuba_true = cv2.imread(
        "out/tsukuba_true_disp.png",
        0
    )

    venus_final = cv2.imread(
        "out/venus_patch_11_final_disparity.png",
        0
    )
    venus_true = cv2.imread(
        "out/venus_true_disp.png",
        0
    )

    map_final = cv2.imread(
        "out/map_patch_11_final_disparity.png",
        0
    )
    map_true = cv2.imread(
        "out/map_true_disp.png",
        0
    )

    return [
        (tsukuba_final, tsukuba_true, "tsukuba"),
        (venus_final, venus_true, "venus"),
        (map_final, map_true, "map")
    ]


def main():
    print("Comparison of final results with true disparity maps:")
    for ours, true, name in get_datasets():
        diff_disp = abs(ours - true)
        print("Handling %s..." % name)
        mean_diff = diff_disp.mean()
        std_dev_diff = diff_disp.std()
        larger_equal_3 = sum(sum(diff_disp >= 3))
        perc_larger_eq_3 = larger_equal_3 / diff_disp.reshape(-1).shape[0]
        print("Mean diff = %.2f" % mean_diff)
        print("Std. dev. = %.2f" % std_dev_diff)
        print("Larger or equal to 3 = %s [%.2f%%]" % (
            larger_equal_3, perc_larger_eq_3 * 100))
