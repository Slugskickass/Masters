
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import math
from PIL import Image as im
import system as sys

# Difference of Gaussians filtering programme. This programme takes a 2D or 3D image and Gaussian width inputs in px
# for wide and narrow Gaussians respectively as sys.argv[1][2][3], respectively. THE WIDTHS OF GAUSSIANS SHOULD BE NO
# MORE THAN 50 PX. The programme outputs the filtered image, saved to the direectory as a numpy array for further
# processing.


def load_img (file_name): # Define function to load image
    img_locat = im.open(file_name)
    print ('Image size (px)',img_locat.size)
    print ('Number of frames',img_locat.n_frames)
    if img_locat.n_frames > 1:
        img_array = np.zeros ((img_locat.size[1],img_locat.size[0],img_locat.n_frames), np.float32)
        for I in range(img_locat.n_frames):
            img_locat.seek(I)
            img_array [:,:,I] = np.asarray(img_locat)
        img_locat.close
    else:
        img_array = np.zeros ((img_locat.size[1],img_locat.size[0]), np.float32)
        img_locat.seek(0)
        img_array = np.asarray(img_locat)
    return img_array

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


# Define inputs for DOG function, from command line inputs
file = sys.argv[1]
wide = sys.argv[2]
narrow = sys.arv[3]

# Load image as array
data = load_img(file)

# Perform filtering operation
DOG = diff_of_gauss(data,narrow,wide)

# Save filtered image as numpy array
np.save('filtered_img', DOG)