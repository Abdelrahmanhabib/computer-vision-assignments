import marshal
from math import sqrt

import matplotlib.pyplot as plt
from skimage import data, io
from skimage.color import rgb2gray
from skimage.feature import blob_log

from output import savefig

def main():
    print("Running ex. 1...")
    #We first explore the threshold parameter to understand its use and to determine its fixed value.
    #Therefore we will focus on only one image
    image = io.imread('Img001.png')

    #Apply blob_log feature from Skimage http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.canny
    blobs_log01 = blob_log(image, max_sigma=10, num_sigma=10, threshold=.1) 
    blobs_log02 = blob_log(image, max_sigma=10, num_sigma=10, threshold=.2) #same image, different threshold value
    blobs_log03 = blob_log(image,  max_sigma=10, num_sigma=10, threshold=.3) #same image, different threshold value 

    #Output is x and y coordinates of blob and sigma value the blob was detected at. 

    # Compute radii in the 3rd column, size of the blob is related to the sigma value. 
    blobs_log01[:, 2] = blobs_log01[:, 2] * sqrt(2) #Do this three times becuase of 3 different threshold values
    blobs_log02[:, 2] = blobs_log02[:, 2] * sqrt(2)
    blobs_log03[:, 2] = blobs_log03[:, 2] * sqrt(2)

    blobs_img = [blobs_log01, blobs_log02, blobs_log03] #put manipulated images into a list so that we can run a loop
    blobs_colors = ['orange', 'orange', 'orange'] #choose color to show detected blobs
    img_titles = ['LoG_01 T:0.1', 'LoG_01 T:0.2','LoG_01 T:0.3'] #title displays threshold value 
    all_lists = zip(blobs_img, blobs_colors, img_titles) #python built in function, returns a list of tuples

    #create subplots via Matplotlib
    fig, axes = plt.subplots(1, 3, figsize=(15, 15), sharex=True, sharey=True, subplot_kw={'adjustable': 'box-forced'})
    ax = axes.ravel() #NumPuy function returns a contiguous flattened array

    for index, (blobs, color, titles) in enumerate(all_lists): #Returns an enumerate object https://docs.python.org/3/library/functions.html#enumerate
        ax[index].set_title(titles)
        ax[index].imshow(image, interpolation='nearest', cmap='gray') #creates grayscale image
        for blob in blobs:
            y, x, r = blob #remember: output is x, y coordinates and sigma value (equivalent to blob)
            c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False) #plot the actual circles onto the image!!! 
            ax[index].add_patch(c)
        ax[index].set_axis_off()

    # plt.show()
    savefig(fig, "ex1-fig1.png")

    #now we play with the max sigma and num sigma values until we discover a suitable set

    #still only using image 1
    image01a = io.imread('Img001.png')
    image01b = io.imread('Img001.png')
    image01c = io.imread('Img001.png')

    #Comparing different paramters, threshold is fixed 
    blobs_log01a = blob_log(image01a, max_sigma=20, num_sigma=10, threshold=.1) 
    blobs_log01b = blob_log(image01b, max_sigma=10, num_sigma=10, threshold=.1)
    blobs_log01c = blob_log(image01c, max_sigma=5, num_sigma=5, threshold=.1)    

    #compute radii
    blobs_log01a[:, 2] = blobs_log01a[:, 2] * sqrt(2)
    blobs_log01b[:, 2] = blobs_log01b[:, 2] * sqrt(2)
    blobs_log01c[:, 2] = blobs_log01c[:, 2] * sqrt(2)

    #lists for for loops!
    images_list = [image01a,image01b,image01c]
    blobs_list = [blobs_log01a, blobs_log01b, blobs_log01c]
    blobs_colors = ['yellow', 'yellow', 'yellow']
    img_titles = ['LoG_01 Max_SD:20 Num_SD:10', 'LoG_01 Max_SD:10 Num_SD:10', 'LoG_01 Max_SD:05 Num_SD:05']
    all_lists = zip(blobs_list, blobs_colors, img_titles)

    #subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 15), sharex=True, sharey=True,subplot_kw={'adjustable': 'box-forced'})
    ax = axes.ravel()

    #Loop through lists to apply blobs and colors 
    for index, (blobs, color, titles) in enumerate(all_lists):
        ax[index].set_title(titles)
        ax[index].imshow(images_list[index], interpolation='nearest', cmap='gray')
        for blob in blobs:
            y, x, r = blob
            c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
            ax[index].add_patch(c)
        ax[index].set_axis_off()

    # plt.show()
    savefig(fig, "ex1-fig2.png")


    #Blob detection for Images 02 and 09 - to compare with middle image above (Image 01 ms:10, ns:10, t:0.1)

    image2 = io.imread('Img002.png')
    image9 = io.imread('Img009.png')

    #set to suitable paramters experimented with above 
    blobs_log2 = blob_log(image2, max_sigma=10, num_sigma=10, threshold=.1)
    blobs_log9 = blob_log(image9, max_sigma=10, num_sigma=10, threshold=.1)

    #compute radii
    blobs_log2[:, 2] = blobs_log2[:, 2] * sqrt(2)
    blobs_log9[:, 2] = blobs_log9[:, 2] * sqrt(2)

    #lists for the for loop!
    images_list = [image2,image9]
    blobs_list = [blobs_log2, blobs_log9]
    blob_colors = ['dodgerblue', 'lime']
    img_titles = ['LoG_02 Max_SD:10 Num_SD:10', 'LoG_09 Max_SD:10 Num_SD:10']
    all_lists = zip(blobs_list, blob_colors, img_titles)

    fig, axes = plt.subplots(1, 2, figsize=(15, 15), sharex=True, sharey=True,subplot_kw={'adjustable': 'box-forced'})
    ax = axes.ravel()

    for index, (blobs, color, titles) in enumerate(all_lists):
        ax[index].set_title(titles)
        ax[index].imshow(images_list[index], interpolation='nearest', cmap='gray')
        for blob in blobs:
            y, x, r = blob
            c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
            ax[index].add_patch(c)
        ax[index].set_axis_off()
        
    # plt.show()
    savefig(fig, "ex1-fig3.png")

    marshal.dump(blobs_log01b.tolist(), open("img1kp.bin","wb"))
    marshal.dump(blobs_log2.tolist(), open("img2kp.bin","wb"))
    marshal.dump(blobs_log9.tolist(), open("img9kp.bin","wb"))
