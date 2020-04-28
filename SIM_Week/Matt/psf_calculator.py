#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 12:35:37 2020

@author: mattarnold
"""

import numpy as np
import im_func_lib as funci
import matplotlib.pyplot as plt
import pandas as pd
from skimage.morphology import closing, square
from skimage.measure import label, regionprops, regionprops_table
from scipy.optimize import curve_fit
import scipy as sp


def centre_collection(thresholded_data, scale=1):
    thresholded_data = closing(thresholded_data > np.mean(thresholded_data) + scale * np.std(thresholded_data))
    # label the image
    label_image = label(thresholded_data, connectivity=2)
    properties = ['area', 'centroid']

    # Calculate the area and centroids
    # regionprops_table caluclates centroids but turns them to integers. It seems to use a floor system to do so.
    # This is suitable for our purposes.
    regions = regionprops_table(label_image, properties=properties)

    # made a panda table, contains, 'area', 'centroid-0', 'centroid-1'
    datax = pd.DataFrame(regions)

    return datax

def point_crop (data, y, x, side_val):    
#    set crop square parameters
    side_length = int(side_val)
    half_side = int(side_length/2)
#    contingency for if square would exit frame area
    if y - side_length < 0: 
        y = 0 + half_side + 1
    if y + side_length > data.shape[1]:
        y = (data.shape[1])-half_side
    if x - side_length < 0:
        x = 0 + half_side + 1
    if x + side_length > data.shape[0]:
        x = (data.shape[0])-half_side
#    frame crop performed
    square_result = data[y - (half_side+1):y + half_side,x - (half_side+1):x + half_side]
    
#    return the square crop area as an ndarray
    return square_result


def crop(beads_frame, mic_array):
    
    a = beads_frame.shape[0]

    save_array = np.zeros((a,15,15))
    
    
    for x in range(0,beads_frame.shape[0]):
        # Define crop area from centre of mass data in imported locations table    
        centre_y = beads_frame.iloc[x,1]
        centre_x = beads_frame.iloc[x,2]
        
        # Call cropping function
        bead_crop = point_crop(mic_array, centre_y, centre_x, 15)
        
        save_array[x,:,:] = bead_crop
        
    return save_array


ref = funci.load_img("/Users/mattarnold/Masters/SIM_week/Data/SLM-SIM_Tetraspeck200_680nm.tif")
sum_ref = np.sum(ref,2)

beads = centre_collection(sum_ref)
beads_clean1 = beads[beads["area"] < 16]
beads_clean =  beads_clean1[beads_clean1["area"] > 2]

crop_beads = crop(beads_clean, sum_ref)

psf = np.mean(crop_beads, axis=0)

mtf = sp.fft.fft2(psf)

plt.imshow(abs(mtf))
plt.show