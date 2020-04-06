import pywt
import numpy as np
import STORM_Samurai as sam
import matplotlib.pyplot as plt
import pandas as pd
from skimage.morphology import closing, square
from skimage.measure import label, regionprops, regionprops_table


# Load image
original = sam.loadtiffs("/Users/RajSeehra/University/Masters/Semester 2/test folder/00001.tif")


def wavelet(image, scale = 1):
    # This thresholds the data based on db1 wavelet.
    coeffs2 = pywt.dwt2(image[:, :, 0], 'db1')

    # This assigns the directionality thresholded arrays to variables.
    LL, (LH, HL, HH) = coeffs2

    # This line helps eliminate the cloud but...
    coeffs2 = LL * 0, (LH * 1, HL * 1, HH * 1)

    # Reconstruct the image based on our removal of the LL (low frequency) component.
    new_img = pywt.idwt2(coeffs2, 'db1')

    # Print statements for evaluation purposes.
    print("Reconstructed image parameters")
    print("Standard Deviations: ", np.std(new_img))
    print("Means: ", np.mean(new_img))
    print("Maxima: ", np.amax(new_img))

    # Thresholding based on the mean and std of the image..
    thresholded_image = pywt.threshold(new_img, np.mean(new_img) + scale * np.std(new_img), substitute=0, mode='greater')

    # For image viewing purposes.
    # plt.subplot(131)
    # plt.imshow(image[:, :, 0], cmap=plt.cm.gray)
    # plt.title("Original")
    # plt.subplot(132)
    # plt.imshow(new_img, cmap=plt.cm.gray)
    # plt.title("After")
    # plt.subplot(133)
    # plt.imshow(thresholded_image, cmap=plt.cm.gray)
    # plt.title("Thresholded")
    # plt.show()

    return thresholded_image


def centre_collection(thresholded_data):
    # label the image
    label_image = label(thresholded_data, connectivity=1)
    properties = ['area', 'centroid']

    # Calculate the area
    regions = regionprops_table(label_image, properties=properties)

    # made a panda table, contains, 'area', 'centroid-0', 'centroid-1'
    datax = pd.DataFrame(regions)
    datax = datax[datax['area'] >= 0]      # at area >0 with std 5 can pick up all the appropriate intensities.
    datax= datax[datax['area'] <= 5]
    return datax


wave = wavelet(original, 5)
centres = centre_collection(wave)

plt.subplot(121)
plt.imshow(original[:,:,0])
plt.subplot(122)
plt.imshow(wave)
plt.plot(centres['centroid-1'], centres['centroid-0'], '*')
plt.show()
