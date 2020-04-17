#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 15:30:13 2020

@author: mattarnold
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL as im
import general as genr

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
    square_result = data[y - (half_side):y + half_side,x - (half_side):x + half_side]
    
#    return the square crop area as an ndarray
    return square_result

# Load data from .csv to pandas dataframe
molecules = pd.read_csv("/Users/mattarnold/Masters/STORM_Week/storm_output_data/panda_data_14_.csv")
molecules = molecules.loc[:,["area", "centroid-0", "centroid-1", "file_name"]]

# Load files and then crop molecules

# Define file variable
file = ""

# Create dataframe to save data into
save_data = pd.DataFrame(columns=["file_name","x-coord","y-coord","cutout_square"])

# Iterate through indices in imported locations dataframe
for index in molecules.index:

    # If the file name for current index does not match the current working file, load the file
    if file is not "{}".format(molecules.loc[index,"file_name"]):
        file = genr.load_img("{}".format(molecules.loc[index,"file_name"]))

    # Define crop area from centre of mass data in imported locations table    
    centre_y = int(np.floor(molecules.loc[index,"centroid-0"]))
    centre_x = int(np.floor(molecules.loc[index,"centroid-1"]))
    
    # Call cropping function
    molecule_crop = point_crop(file, centre_y, centre_x, 30)
    
    # Define variables for saving out data
    file_nm, centroid_one, centroid_zero = molecules.loc[index,"file_name"], molecules.loc[index,"centroid-1"], molecules.loc[index,"centroid-0"]
    
    # Format variables into a pandas series to allow saving to dataframe
    molecule_crop_id = pd.Series({"file_name":file_nm,"x-coord": centroid_one, "y-coord": centroid_zero ,"cutout_square": molecule_crop})
    
    # Convert series to dataframe, with index relating to current value of index variable
    save_row = pd.DataFrame([molecule_crop_id], index = ["{}".format(index)])
    
    # Concatenate the data from this iteration to the created data frame 
    save_data = pd.concat([save_row,save_data],axis=0)
        
# Save finished dataframe to .csv in working directory
save_data.to_csv("molecules_positions_crops.csv")

