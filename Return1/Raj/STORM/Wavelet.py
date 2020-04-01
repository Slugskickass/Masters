import pywt
import numpy as np
import STORM_Samurai as sam
import matplotlib.pyplot as plt
import pandas as pd
from skimage.morphology import closing, square
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops, regionprops_table


# Load image
original = sam.loadtiffs("/Users/RajSeehra/University/Masters/Semester 2/test folder/00001.tif")

def wavelet(original):
    # Wavelet transform of image, and plot approximation and details
    titles = ['Approximation', ' Horizontal detail',
              'Vertical detail', 'Diagonal detail']

    # This thresholds the data based on db1 wavelet.
    coeffs2 = pywt.dwt2(original[:, :, 0], 'db1')

    # This assigns the directionality thresholded arrays to variables.
    LL, (LH, HL, HH) = coeffs2

    # This line helps eliminate the cloud but...
    coeffs2 = LL * 0, (LH * 1, HL * 1, HH * 1)

    # Print statements for evaluation purposes.
    print("Standard Deviations: ", np.std(LL), np.std(LH), np.std(HL), np.std(HH))
    print("Means: ", np.mean(LL), np.mean(LH), np.mean(HL), np.mean(HH))
    print("Maxima: ", np.amax(LL), np.amax(LH), np.amax(HL), np.amax(HH))

    new_img = pywt.idwt2(coeffs2, 'db1')

    # BE AWARE. multiplied up std and mean to get cleaner image, may make this user input based.
    y = pywt.threshold_firm(new_img, np.mean(new_img) + 1.5 * np.std(new_img), np.amax(new_img))

    plt.subplot(131)
    plt.imshow(original[:,:,0], cmap=plt.cm.gray)
    plt.title("Original")
    plt.subplot(132)
    plt.imshow(new_img, cmap=plt.cm.gray)
    plt.title("After")
    plt.subplot(133)
    plt.imshow(y, cmap=plt.cm.gray)
    plt.title("Thresholded")
    plt.show()
    return y


def centre_collection(thresholded_data):
    # label the image
    label_image = label(thresholded_data, connectivity=2)
    properties = ['area', 'centroid']

    # Calculate the area
    regions = regionprops_table(label_image, properties=properties)

    # made a panda table
    datax = pd.DataFrame(regions)
    areas = np.asarray(datax['area'])
    centroid_x = np.asarray(datax['centroid-0'])
    centroid_y = np.asarray(datax['centroid-1'])

    return label_image, areas, centroid_x, centroid_y

wave = wavelet(original)
centres = centre_collection(wave)

