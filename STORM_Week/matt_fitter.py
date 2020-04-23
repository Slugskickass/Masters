#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 12:25:45 2020

@author: mattarnold
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import general as genr
import pandas as pd
import os
import ast

# Function to convert arrays from cells in .csv converted pandas frames back from strings to numerical elements    
def array_clean(cutout_array):
   cutout_array = ','.join(cutout_array.replace('[ ', '[').split())
   return np.array(ast.literal_eval(cutout_array))

# Function to open the saved location data, add columns for processed data and clean arrays using above function
def build_mol_frame (csv_file_path=None):
    # If path is not specified, assume working directory
    if csv_file_path==None:
            molecules = pd.read_csv("{}/molecules_positions_crops.csv".format(os.getcwd()))
    # Otherwise, use specified path
    else:
        molecules = pd.read_csv("{}".format(csv_file_path))
    # Ditch old index column
    molecules = molecules.loc[:,["file_name", "x-coord", "y-coord", "cutout_square"]]
    # Add colmuns for fitting results
    molecules["popt"], molecules["centre-x"], molecules["centre-y"] = "", "", ""
    
    for index in molecules.index:
        clean_array = array_clean(molecules.loc[index,"cutout_square"])
        molecules.at[index,"cutout_square"] = clean_array
    # Return spruced-up dataframe
    return molecules

# FITTING FUNCTION
    #PARAMETERS: pandas dataframe generated in above function, size of Gaussian for fitting
    #RETURNS: pandas dataframe containing above data, plus fitting positions in x and y for centre
def df_gauss_fit (size, input_mols_df=None):
    mol_frame = build_mol_frame(input_mols_df)
    # Build "numberlines" of x and y in increments of one up to "size"
    x = np.linspace(0, size-1, size)
    y = np.linspace(0, size-1, size)
    # From these, build incremental grids in x and y for gaussian generation
    X, Y = np.meshgrid(x, y)
    # Convert incremental grids into 1d data for fitting
    xdata = np.vstack((X.ravel(), Y.ravel()))
    
    count=0
    # Work through index of dataframe, attempting a Gaussian fit with defined parameters, on the array in that row
    for index in mol_frame.index:
        try:
            mol_square = mol_frame.at[index,"cutout_square"]
            # Define guesses for Gaussian fit
            param_guess = [3,3,mol_square.shape[1],mol_square.shape[0],2,2,np.min(mol_square), np.max(mol_square) - np.min(mol_square)]
            param_array = np.asarray(param_guess)
            low_bound, up_bound = [-np.inf,-np.inf,0,0,1,1,-np.inf,-np.inf], [np.inf,np.inf,mol_square.shape[1],mol_square.shape[0],4,4,np.inf,np.inf]
            popt, pcov = curve_fit(genr._gaussian, xdata, mol_square.ravel(), p0=param_array, bounds=(low_bound,up_bound))
            # Save data for locations from fit to dataframe
            centre-x, centre-y = (mol_frame.loc[index,"x-coord"] - np.floor(size/2) + popt[3]), (mol_frame.loc[index,"y-coord"] - np.floor(size/2) + popt[4])
            mol_frame.loc[index,"popt"], mol_frame.loc[index,"centre-x"], mol_frame.loc[index,"centre-y"] = popt, centre-x, centre-y
        # If fit is unsuccessful (due to noisy frame), count it and continue to next frame
        except:
            #print("unable to fit square at", mol_frame.loc[index,["file_name","x-coord","y-coord"]])
            count+=1
            continue
    print("The number of failed fits is:", count)
    mol_frame.to_csv("localisation_data.csv")
    return mol_frame
        

localisation_data = df_gauss_fit(7)

