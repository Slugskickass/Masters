#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 13:26:53 2020

@author: mattarnold
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, optimize

# Pixel by pixel gain analysis: To visualise the difference in gains between individual camera pixels, the gradient of
# slope of the noise squared is calculated for each pixel at a range of illumination levels. These differential gains
# are then visualised as a histogram.

data = np.load("mean_std_data_pxxpx.npy") # Load the data to be processed (np array with mean index 0, stdev index 1)


sqd_std = (data[:,:,1,:]) ** 2 # Build a separate array for the squared standard deviations

mean = data[:,:,0,:] # Build an array for the means

results = np.zeros((np.size(data,1),np.size(data,0))) #Build an array for the results


for J in range(np.size(data,1)): # Scan through pixels in y...
    for K in range(np.size(data,0)): #... and x.
        slope, intercept, r_value, p_value, std_err = stats.linregress(mean[J,K,:], sqd_std[J,K,:]) # calculate gain
        results[J,K] = slope # Save gain for each pixel in results array

new_results = np.reshape(results,np.size(results))

plt.hist(new_results,bins = np.linspace(0,3, 100)) # Plot
plt.xlabel("Pixel gain / electrons per count")
plt.ylabel("Frequency")
plt.title("Hamamatsu Orca Flash4: Pixel by pixel gain")
plt.show

plt.savefig("flash4_pxxpx_gain.png") # Save