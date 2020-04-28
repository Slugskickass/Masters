#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 16:22:21 2020

@author: mattarnold
"""


def gaussian(x, y, x0, y0, xalpha, yalpha, offset, A):
    return offset + (A * np.exp(-((x-x0)/xalpha)**2 - ((y-y0)/yalpha)**2))

def _gaussian(M, *args):
    x, y = M
    arr = gaussian(x, y, *args)
    return arr

def matt_fitter (array, size=9):
    x = np.linspace(0, size-1, size)
    y = np.linspace(0, size-1, size)
    # From these, build incremental grids in x and y for gaussian generation
    X, Y = np.meshgrid(x, y)
    # Convert incremental grids into 1d data for fitting
    xdata = np.vstack((X.ravel(), Y.ravel()))
    
    count=0
    success=0
    save_array = np.zeros((array.shape[0], size, size))
    # Work through index of dataframe, attempting a Gaussian fit with defined parameters, on the array in that row
    for n in range(0,array.shape[0]):
        try:
            bead_square = array[n,:,:]
            bead_square_1 = array[n,:,:]
            # Define guesses for Gaussian fit
            param_guess = [3,3,bead_square.shape[1],bead_square.shape[0],2,2,np.min(bead_square), np.max(bead_square) - np.min(bead_square)]
            param_array = np.asarray(param_guess)
            low_bound, up_bound = [-np.inf,-np.inf,0,0,1,1,-np.inf,-np.inf], [np.inf,np.inf,bead_square.shape[1],bead_square.shape[0],4,4,np.inf,np.inf]
            popt, pcov = curve_fit(_gaussian, xdata, bead_square.ravel(), p0=param_array, bounds=(low_bound,up_bound))
            save_array[success,:,:] = bead_square_1
            success+=1
            print(success)
        except:
            #print("unable to fit square at", mol_frame.loc[index,["file_name","x-coord","y-coord"]])
            count+=1
            continue
    print("The number of failed fits is:", count)

    return save_array
