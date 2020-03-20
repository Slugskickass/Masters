#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 12:32:08 2020

@author: mattarnold
"""

import numpy as np
import im_func_simp as funci
import sys

# Define a function to output a 2D averaged dark frame for background subtraction from other data in PTC calculation.
# The lowest average intensity frame is found and pixel-by-pixel averaged across the time stack.
# Input: (system argument 1) directory location  of the files to be analysed
# Output: numpy ndarray with same x and y dimensions as the input files of 2D flattened averaged dark background

dir_loc = sys.argv[1] #Define input for function in command line

def dark_frame_avg(dir_loc):
    
    files = funci.get_file_list(dir_loc) #Call function from im_func_lib to build a list of .tif files in directory
    
    file_int_avgs = np.zeros(len(files)) #Build an empty array to store average intensities in for comparison
    for I in range (len(files)):
        work_file = funci.load_img(files[I]) #Iterate through list, loading each using a function from im_func_lib
        avg_int = np.mean(work_file) #Find the mean of the loaded file
        file_int_avgs[I] = avg_int #Add the mean to the processing array 
    
    result = np.where(file_int_avgs == np.amin(file_int_avgs)) #Find the index of the array where the lowest average is
    int_res = np.abs(result) #Convert this output into a useful format
    dark_num = int_res[0,0]
    dark_frame = funci.load_img(files[dark_num]) #Load the darkest frame found above
    avg_dark_frame = np.mean(dark_frame,2) #Average the frame in the z/time axis
    return np.save('dark_frame',avg_dark_frame) #Return the averaged frame, saved as a numpy array
    
     
test = dark_frame_avg(dir_loc) #Run function from input