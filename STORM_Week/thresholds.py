#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 12:18:22 2020

@author: mattarnold
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu

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
    thresh = mean - (std * int(num_std)) # Calculate threshold value
    for x in range(np.size(filtered_data,0)): # Iterate through px in x...
        for y in range(np.size(filtered_data,1)): # ...and y
            if filtered_data[x,y] < thresh: # If the pixel value is less than the threshold...
                filtered_data[x,y] = 0 #... set the value for the pixel to zero.
    return filtered_data
 
