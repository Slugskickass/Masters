#Week 5: Assignment 1: Write a programme which finds the brightest point of an image and cuts out a square of defined
# size containing that point.
#Trested using smaller example arrays of the type: data = np.array([[10,12,14],[16,40,20],[22,24,26]])


import numpy as np
from PIL import Image as im
import sys


#Define the function which will find the point in the image with the highest intensity. This will only work for 2D
# images. The function also crops the image down to a square of a soecified side length. The three arguments the function
# takes are the data (nd.array format) and desired crop size (in px)

# define the parts of the programme determined by the sytem arguments in the command line
data_file = "/Users/mattarnold/iCloud Drive (Archive)/Documents/Documents – Matt’s MacBook Pro/Uni/Imaging Msc/semester 2/Python Module/PTC/test.tif" #sys.argv[1]
side_val = 50 #sys.argv[2]
frame_used = 4 #sys.argv[3]


def max_intensity_crop (data, side_val):
#    calculate number of pixels in the image
    num_of_pixels = np.size(data)
    
#   reshape the array into a 1D to enable proper use of the enumerate command 
    data_arr_1d = data.reshape(num_of_pixels)
    
#    determine the highest intensity point
    max_val = np.max(data_arr_1d)
    
#    search the array for this point and find its position using enumerate
    for index,value in enumerate(data_arr_1d):
        if max_val == value:
            result_pt = index
            
#    convert the 1D position found by enumerating into a 2D position in the original image
    y = int(result_pt / np.size(data,1))
    x = int(result_pt % np.size(data,1))
    
#    set crop square parameters
    side_length = int(side_val)
    half_side = int(side_length/2)
#    contingency for if square would exit frame area
    if y - side_length < 0: 
        y = 0 + half_side + 1
    if y + side_length > data.shape[0]:
        y = (np.size(data,1))-half_side
    if x - side_length < 0:
        x = 0 + half_side + 1
    if x + side_length > np.size(data,0):
        x = (np.size(data,0))-half_side
#    frame crop performed
    square_result = data[y - (half_side):y + half_side,x - (half_side):x + half_side]
    print('Crop area is a square with ',side_length,'pixels on a side, from pixel ', x - (int(side_length/2)), 'in x, and pixel ',y - (int(side_length/2)), 'in y')
    
#    return the square crop area as an ndarray
    return square_result


#define a function to save an image or stack
def save_img_2d(file_name, data):
    images = im.fromarray(data[:, :])
    images.save(file_name)


#define a function to load an image
def load_img (file_name):
    img_locat = im.open(file_name)
    
    print ('Image size (px)',img_locat.size)
    print ('Number of frames',img_locat.n_frames)
    
    img_array = np.zeros ((img_locat.size[1],img_locat.size[0],img_locat.n_frames), np.uint16)
    for I in range(img_locat.n_frames):
        img_locat.seek(I)
        img_array [:,:,I] = np.asarray(img_locat)
    img_locat.close
    return img_array

#define a function to select a frame of a multi-frame image
def frame_sel (data,frame=None):
    selected_frame = data[:,:,frame]
    return selected_frame

# Load image into prgramme
load_data = load_img (data_file)

# select the frame of interest
data = frame_sel(load_data,int(frame_used))

# Calling the function         
test = max_intensity_crop(data,50)

# Saving the file out
save_img_2d ("doesthiswork.tif",test)

