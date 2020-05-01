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

## FIND BEADS: function to find beads in the reference image
def centre_collection(bead_data, scale=1):
    
    #Threshold bead data
    bead_data = closing(bead_data > np.mean(bead_data) + scale * np.std(bead_data))
    
    #Label the image
    label_image = label(bead_data, connectivity=2)
    
    #Define desired region props parameters
    properties = ['area', 'centroid']

    #Calculate the area and centroids
    regions = regionprops_table(label_image, properties=properties)

    #Save the locations to a panda table, contains, 'area', 'centroid-0', 'centroid-1'
    datax = pd.DataFrame(regions)

    return datax

## CROP BEADS: crop out areas of the reference image where bead cetroids have been found
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

## BEAD CROP: cropping function to extract bead areas from reference image according to centroid data
def crop(beads_frame, mic_array):
    
    #Define number of bead areas to crop
    a = beads_frame.shape[0]
    
    #Define size of crop region in px (big enough to enclose psf of a correctly set up SIM system)    
    size = 11
    
    #Build arrays to save data to
    save_array = np.zeros((a, size, size))
    save_array_fill = np.empty([a, mic_array.shape[1], mic_array.shape[1]])
    
    #Iterate through bead data frame
    for x in range(0,beads_frame.shape[0]):
        
        # Define crop area from centre of mass data in imported locations table    
        centre_y = beads_frame.iloc[x,1]
        centre_x = beads_frame.iloc[x,2]
        
        # Call cropping function and save out crop area
        bead_crop = point_crop(mic_array, centre_y, centre_x, size)
        save_array[x,:,:] = bead_crop
        
    #Define background value for iflling out crop squares to full frame size and fill out array
    fill_val = np.min(save_array) + 1.5*np.std(save_array)
    save_array_fill.fill(fill_val)
    
    #Insert cropped areas into background corrected array
    save_array_fill[:,mic_array.shape[1]//2 - size//2: mic_array.shape[1]//2 + size//2 +1,mic_array.shape[1]//2 - size//2: mic_array.shape[1]//2 + size//2 +1] = save_array[:,:,:]
    
    #Return 3d array of cropped bead regions in frame sized area
    return save_array_fill

## Master function for loading of data and extraction of parameters (psf,otf)
def data_extract(bead_ref_file_path):
    
    #Load reference bead data
    ref = funci.load_img(bead_ref_file_path)
    
    #Sum the bead data in z to remove effects of SIM illumination
    sum_ref = np.sum(ref,2)
    
    #Run centre collection function to find bead locations and filter out areas too large or small for single beads
    beads = centre_collection(sum_ref)
    beads_clean1 = beads[beads["area"] < 12]
    beads_clean =  beads_clean1[beads_clean1["area"] > 2]
    
    #Use these locations to crop out bead areas
    crop_beads = crop(beads_clean, sum_ref)
    
    #Average the beads areas to find an approximation the system's response to a point source
    psf = np.mean(crop_beads, axis=0)
    
    #FFT of the point spread function gives the optical transfer function
    otf_init = sp.fft.fft2(psf)
    otf = sp.fft.fftshift(otf_init)
    
    #Return the summed reference image, calculated psf and otf
    return sum_ref, psf, otf

##COMPLEX CONJ: find the complex conjugate of the recorded data and the otf
def complex_conj_test(data_file_path, mtf):
    
    #Load SIM image data
    data= funci.load_img(data_file_path)
    
    #Find FFT of first frane (in this testing instance)
    data_fft = sp.fft.fftshift(sp.fft.fft2(data[:,:,1]))
    
    #Convolv this with the OTF
    otf_convolve = sp.signal.convolve2d(otf,data_fft)
    
    #Perform the complex conjugation operation
    complex_conj = np.conj(otf_convolve)
    
    #Return the fft of the data, the frame of the data being used and the calculated complex conjugate
    return data_fft, data[:,:,1] ,complex_conj

##RUN##
sum_ref, psf, otf = data_extract("/Users/mattarnold/Masters/SIM_week/Data/SLM-SIM_Tetraspeck200_680nm.tif")

data_fft , data, conj = complex_conj_test("/Users/mattarnold/Masters/SIM_week/Data/TIRF_Tubulin_525nm.tif", otf)

##PLOT##
figure, axes = plt.subplots(figsize = (12,16))
plt.subplot(321)
plt.imshow(sum_ref)
plt.title("Summed reference image")

plt.subplot(322)
plt.imshow(psf)
plt.title("PSF")

plt.subplot(325)
plt.imshow((np.abs(otf)),norm=LogNorm(vmin=5))
plt.title("Calculated OTF")
plt.show

plt.subplot(324)
plt.imshow(data)
plt.title("Data")
plt.show

plt.subplot(323)
plt.imshow((np.abs(data_fft)),norm=LogNorm(vmin=5))
plt.title("Data FFT")
plt.show

plt.subplot(326)
plt.imshow((np.real(conj)),norm=LogNorm(vmin=5))
plt.title("Comlex conjugate")
plt.show