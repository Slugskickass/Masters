import numpy as np
from scipy import signal
import math


### DIFFERENCE OF GAUSSIANS

# Define function to calculate the amplitude of the Gaussian based on an area under the curve of 1 using ingtegral of
# 2D Gaussian formula
def gauss_height(width):  
    height = (1 / (math.sqrt(2) * np.abs(width) * math.sqrt(math.pi)))/(2*(np.sqrt(2)))
    return height

# Define a function to build the Gaussians based on the calculated parameters
def build_2d_gauss(width, height):  
    xo = yo = 25 # Centre of distribution...
    kernel = np.zeros((50, 50))  # ...within defined matrix size, for convultion in filtering
    for x in range(np.shape(kernel)[0]): # iterate through x and y
        for y in range(np.shape(kernel)[1]):
            kernel[x, y] = height * np.exp(-1 * (((x - xo) ** 2 + (y - yo) ** 2) / width ** 2))#Calculate value at x,y
    return kernel # Return kernel as an np array

def dog_params (param_a, param_b):
    if param_a > param_b: # Determine which is greater to assign "wide" and "narrow" correctly
        wide = param_a 
        narrow = param_b
    elif param_a < param_b:
        wide = param_b
        narrow = param_a
    else:
        raise ValueError('Widths for Gaussians must be different')#Raise an error if the same width has been entered twice
    return wide, narrow

def diff_of_gauss (data, narrow_width, wide_width): # Define function ot perform filtering
    height_narrow = gauss_height(narrow_width) # Calculate height from widths inputed
    height_wide = gauss_height(wide_width)
    narrow_kern = build_2d_gauss (narrow_width, height_narrow) # Build Gaussians
    wide_kern = build_2d_gauss (wide_width, height_wide)
    
    if data.ndim > 2: #if multiple frames in image then:
        gauss_img_small = np.zeros_like(data) # Build arrays for image convolved with each Gaussian
        gauss_img_big = np.zeros_like(data)
        for I in range(np.size(data,2)):
            # Frame by frame, convolve each calculated Gaussian (separately), with image data
            gauss_img_small[:,:,I] = signal.convolve2d(data[:,:,I], narrow_kern,mode='same',boundary='symm') 
            gauss_img_big[:,:,I] = signal.convolve2d(data[:,:,I], wide_kern, mode='same', boundary='symm')  
        diff_gauss_img = gauss_img_small - gauss_img_big # Perform difference operation of convolved images
    
    else: # For single frames:
        # Convolve each calculated Gaussian (separately), with image data
        gauss_img_small = signal.convolve2d(data, narrow_kern,mode='same',boundary='symm')
        gauss_img_big = signal.convolve2d(data, wide_kern, mode='same', boundary='symm') 
        diff_gauss_img = gauss_img_small - gauss_img_big # Perform difference operation of convolved images
    
    return diff_gauss_img # Return processed image as a numpy array
#
#
#
#
