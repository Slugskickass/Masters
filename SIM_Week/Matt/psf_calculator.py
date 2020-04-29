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
from matplotlib.colors import LogNorm
from matplotlib import cm


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
        y = 0 + half_side
    if y + side_length > data.shape[1]:
        y = (data.shape[1])-half_side -1
    if x - side_length < 0:
        x = 0 + half_side
    if x + side_length > data.shape[0]:
        x = (data.shape[0])-half_side -1
#    frame crop performed
    square_result = data[y - half_side:y + half_side +1, x - half_side:x + half_side + 1]
    
#    return the square crop area as an ndarray
    return square_result


def crop(beads_frame, mic_array):
    
    a = beads_frame.shape[0]
        
    size = 20
    
    save_array = np.zeros((a, mic_array.shape[1], mic_array.shape[1]))
        
    
    for x in range(0,beads_frame.shape[0]):
        # Define crop area from centre of mass data in imported locations table    
        centre_y = beads_frame.iloc[x,1]
        centre_x = beads_frame.iloc[x,2]
        
        # Call cropping function
        bead_crop = point_crop(mic_array, centre_y, centre_x, size)
        
        save_array[x,mic_array.shape[1]//2 - size//2: mic_array.shape[1]//2 + size//2 +1,mic_array.shape[1]//2 - size//2: mic_array.shape[1]//2 + size//2 +1] = bead_crop
        
    return save_array

def data_extract(file_path):
    ref = funci.load_img(file_path)
    sum_ref = np.sum(ref,2)
    
    beads = centre_collection(sum_ref)
    beads_clean1 = beads[beads["area"] < 16]
    beads_clean =  beads_clean1[beads_clean1["area"] > 2]
    
    crop_beads = crop(beads_clean, sum_ref)

    psf = np.mean(crop_beads, axis=0)
    
    mtf_init = sp.fft.fft2(psf)
    
    mtf = sp.fft.fftshift(mtf_init)
    
    return psf, mtf, sum_ref


def complex_conj_test(data_file_path, mtf):
    data= funci.load_img(data_file_path)
    print("c")
    data_fft = sp.fft.fftshift(sp.fft.fft2(data[:,:,1]))
    print("d")
    mtf_convolve = sp.signal.convolve2d(mtf,data_fft)
    print("e")
#    complex_conj = np.conj(mtf_convolve)
#    print("f")
    return data_fft, data[:,:,1] #,complex_conj

print("a")

psf, mtf, sum_ref = data_extract("/Users/mattarnold/Masters/SIM_week/Data/SLM-SIM_Tetraspeck200_680nm.tif")

print("b")

data_fft , data = complex_conj_test("/Users/mattarnold/Masters/SIM_week/Data/TIRF_Tubulin_525nm.tif", mtf)


figure, axes = plt.subplots(figsize = (12,16))
plt.subplot(321)
plt.imshow(sum_ref)
plt.title("Summed reference image")

plt.subplot(323)
plt.imshow(psf)
plt.title("PSF")

plt.subplot(324)
plt.imshow((np.abs(mtf)))
plt.title("Calculated MTF")
plt.show

plt.subplot(325)
plt.imshow(data)
plt.title("Data")
plt.show

plt.subplot(326)
plt.imshow((np.abs(data_fft)),norm=LogNorm(vmin=5))
plt.title("Data FFT")
plt.show

#plt.subplot(326)
#plt.imshow((np.abs(conj)))
#plt.title("Comlex conjugate")
#plt.show