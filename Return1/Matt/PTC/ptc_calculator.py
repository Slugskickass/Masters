#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 16:07:36 2020

@author: mattarnold
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import im_func_lib as funci
import sys

# Defining a function to background correct and then crop areas from images used in a PTC plot. The means and standard
# standard deviations of the background-corrected images are outputed as a 2D numpy array. TO RUN THIS SCRIPT YOU MUST
# ALREADY HAVE RUN dark_avg.py AND fake_bbox.py and ALLOWED THE OUTPUT TO SAVE TO THE DIRECTORY CONTAINING THE SCRIPTS.
# Input: (system argument 1) directory location of files to be analysed
# Output: A 2D numpy array with a number of rows equal to the number of images being processed, with means in column 0
#           stdevs in column 1.

input_data = sys.argv[1] 

files = funci.get_file_list(input_data) #Import list of file names for processing



def ptc_crop(file_list):
    dark_frame = np.load('dark_frame.npy') #Import previously calculated dark frame from directory
    
    num_files = len(file_list) # Determine number of files in folder
    data_table = np.zeros((num_files,2)) # Build an array for results
    
    crop_bbox = np.load('crop_box.npy') # Load the array containing bounding box dimensions ([[x-lo,x-hi][y-lo,y-hi]])
    y_low, y_high, x_low, x_high = crop_bbox[0,0],crop_bbox[0,1],crop_bbox[1,0],crop_bbox[1,1]
    
    for I in range(num_files): # Iterate through files 
        work_img = funci.load_img(files[I]) # Load each
        for J in range(np.size(work_img,2)): # Iterate through frames
            work_img[:,:,J] = np.subtract(work_img[:,:,J],dark_frame) # Subtract the dark frame from each
        crop_image = work_img [y_low : y_high, x_low : x_high,:] # Crop each image to the desired dimensions
        
        
        data_table[I,0] = np.mean(crop_image) # Average the image and save the mean to column 0
        data_table[I,1] = np.std(crop_image) # Find the stdev of the image and save it to column 1
    return np.save('mean_std_data',data_table) # Output the data, saved in a 2D numpy array 
    
test = ptc_crop(files) # Run