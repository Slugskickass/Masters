#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 12:18:22 2020

@author: mattarnold
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import pywt

def threshold_switcher(data, settings):
    switcher = {
        'otsu': otsu,
        'statistical': stat_thresh,
        'wavelet': wavelet
    }

    return switcher.get((settings.get("threshold parameters:", {}).get("threshold type:")), data)\
        (data, (settings.get("threshold parameters:", {}).get("input parameter")))


## OTSU THRESHOLD
    # Input: data in form of numpy ndarray
    # Output: ndarray of otsu thresholded data
def otsu(filtered_data):
    thresh = threshold_otsu(filtered_data) # Calculate threshold using skimage otsu command
    
    data_out = filtered_data > thresh # Exclude data less than threshold (WHY DOES THE > OPERATOR WORK HERE BUT NOT BELOW)
    return data_out

## STATISTICAL THRESHOLD
    # Input: data in form of numpy ndarray, sumber of standard deviations below threshold is set
    # Output: ndarray of thresholded data
def stat_thresh(filtered_data, num_std):
    mean = np.mean(filtered_data) # Calculate mean of data
    std = np.std(filtered_data) # Calculate std of data
    thresh = mean + (std * int(num_std)) # Calculate threshold value
    for x in range(np.size(filtered_data,0)): # Iterate through px in x...
        for y in range(np.size(filtered_data,1)): # ...and y
            if filtered_data[x,y] < thresh: # If the pixel value is less than the threshold...
                filtered_data[x,y] = 0 #... set the value for the pixel to zero.
    return filtered_data

def wavelet(image, scale = 1):
    # This thresholds the data based on db1 wavelet.
    if image.ndim > 2:
        coeffs2 = pywt.dwt2(image[:, :,np.size(image,2)], 'db1')
    else:
        coeffs2 = pywt.dwt2(image[:, :], 'db1')

    # This assigns the directionality thresholded arrays to variables.
    LL, (LH, HL, HH) = coeffs2

    # This line helps eliminate the cloud but...
    coeffs2 = LL * 0, (LH * 0, HL * 1, HH * 0)

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
 
#data = np.load('filtered_img.npy')
#
#thresh = wavelet(data,1)
#
#plt.imshow(thresh)
#plt.show
#
#np.save('thresholded_img.npy',thresh)