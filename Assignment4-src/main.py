import numpy as np
import matplotlib.pyplot as plt

import ps_utils
from output import savefig

Beethoven = "Beethoven.mat"
Buddha = "Buddha.mat"


def remove_previous_plot_windows():
    # making sure to close previous plot windows
    plt.cla()
    plt.clf()
    plt.close()


def save_first_image(I, name):
    plt.imshow(I[:, :, 0], cmap="gray")
    ax = plt.gca()
    ax.set_axis_off()
    fig = plt.figure(1)
    savefig(fig, "%s_first_image.png" % name)
    remove_previous_plot_windows()
    return


def save_albedo(albedo_for_display, name):
    # save albedo
    plt.imshow(albedo_for_display, cmap="gray")
    ax = plt.gca()
    ax.set_axis_off()
    fig = plt.figure(1)
    savefig(fig, "%s_albedo.png" % name)
    remove_previous_plot_windows()
    return


# for eaxh image pixel I(p) then I(p)= S*(n(p)*P(p)). Where n(p) is the normal
# of the pixel and P(p) is the albedo. S is light source. n(p)*P(p) = m(p).
# and S^-1 * I(p) = m(p) (the albedo modulated normal of pixel?), where S^-1 is the inverse of S.
# then, the albedo P(p) = ||m(p)|| i.e. the length of the albedo modulated normal of pixel p.
# and, the normal n(p) = m(p)/||m(p)||
def photometric_stereo(filename, name):
    # reads a dataset Matlab mat-file and returns I, mask and S.
    I, mask, S = ps_utils.read_data_file(filename)

    save_first_image(I, name)

    number_of_images = 3
    if filename == Buddha:
        number_of_images = 10

    I_transposed = np.transpose(I, (2, 0, 1))
    I_flat_stacked = np.vstack(x.ravel() for x in I_transposed)
    mask_flat = mask.ravel()
    #print("mask shape: " + str(mask_flat.shape))
    #print("I_Flat_stacked shape " + str(I_flat_stacked.shape))

    S_i = None
    if filename == Buddha:
        S_i = np.linalg.pinv(S)
    elif filename == Beethoven:
        S_i = np.linalg.inv(S)
    nz = np.sum(mask != 0)

    J = np.zeros([number_of_images, nz])
    #print("J shape " + str(J.shape))

    J[0] = I_flat_stacked[0][mask_flat != 0]
    J[1] = I_flat_stacked[1][mask_flat != 0]
    J[2] = I_flat_stacked[2][mask_flat != 0]

    M = S_i.dot(I_flat_stacked)

    image_shape = I[:, :, 0].shape
    albedo_for_display = np.zeros(image_shape)
    albedo_for_display2 = np.zeros(image_shape)
    albedo_for_display3 = np.zeros(image_shape)

    count = 0
    for y in range(0, mask.shape[0]):
        for x in range(0, mask.shape[1]):
            albedo_for_display[y][x] = M[0][count]
            albedo_for_display2[y][x] = M[1][count]
            albedo_for_display3[y][x] = M[2][count]
            count = count + 1

    # Extracting normal field by normalizing M (Albedo normalization):
    #print("M: " + str(M.shape))
    M_normalized = np.linalg.norm(M, axis=0)

    #print("M norm: " + str(M_normalized.shape))

    # M_normalized is the normal times albedo. To get the normals, we need to
    # divide by the albedo (M). However, for each albedo, there are three m's
    # in M_normalized. We cannot simply say [m11, m12, m13]/P1 (or say:
    # M_normalized/M) but must instead repeat each P (albedo) three times and
    # then divide each.
    # Using np.tile() to repeat P three times and give matching shape, and then
    # solving for normals:
    normals = (M / np.tile(M_normalized, (3, 1)))
    #print("normals shape " + str(normals.shape))

    n1 = normals[0]
    n2 = normals[1]
    n3 = normals[2]
    return n1, n2, n3, mask, albedo_for_display


def main():
    for dataset, name in [(Beethoven, "Beethoven"), (Buddha, "Buddha")]:
        n1, n2, n3, mask, albedo_for_display = photometric_stereo(dataset, name)

        albedo_for_display[mask < (mask.max() - mask.min()) / 2.] = np.nan

        save_albedo(albedo_for_display, name)

        n1_reshaped = n1.reshape(mask.shape)
        n2_reshaped = n2.reshape(mask.shape)
        n3_reshaped = n3.reshape(mask.shape)

        n_unbiased_integrated = ps_utils.unbiased_integrate(
            n1_reshaped,
            n2_reshaped,
            n3_reshaped,
            mask
            )
        ps_utils.display_depth(n_unbiased_integrated)


if __name__ == "__main__":
    main()
