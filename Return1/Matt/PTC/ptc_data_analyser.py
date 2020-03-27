#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 13:26:53 2020

@author: mattarnold
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, optimize

# Photn Transfer Curve production: this script takes raw mean and standard deviation data and produces a gain-corrected
# photon tansfer curve.

mean_std_array = np.load("mean_std_data.npy") # Load the data to be processed (np array with mean index 0, stdev index 1)

mean = mean_std_array[:,0] # Define variables 
std = mean_std_array[:,1]
sqd_std = (mean_std_array[:,1])**2 # Square the standard deviation (for gain calculation)

slope, intercept, r_value, p_value, std_err = stats.linregress(mean, sqd_std) # Plot stdev^2 against counts to find gain


print (slope)

# Electrons per count (gain) is equal to the slope of the linear fit here, so if we multiply the results by this number,
# we will see the output in electrons.

mean_std_array = mean_std_array[:,:] * slope # Correct for gain


plt.plot(mean_std_array[:,0],sqd_std[:],"o") # Plot
plt.xscale("symlog")
#plt.yscale("log")
plt.ylabel("Standard deviation electron counts")
plt.xlabel("Log(mean electron counts)")
plt.title('Hamamatsu Orca Flash4 PTC')
plt.show
plt.savefig('Hamamatsu Orca Flash4 PTC')
