#Week 5: Assignment 1: Write a programme which finds the brightest point of an image and cuts out a square of defined
# size containing that point.
#Trested using smaller example arrays of the type: data = np.array([[10,12,14],[16,40,20],[22,24,26]])


import numpy as np
import matplotlib.pyplot as plt
import im_func_lib as funcy
from PIL import Image as im
import sys
for I in range(len(sys.argv)):
    print(sys.argv[I])

#Define the function which will find the point in the image with the highest intensity. This will only work for 2D
# images. The function also crops the image down to a square of a soecified side length. The two arguments the function
# takes are the data (nd.array format) and desired crop size (in px)

def max_intensity_crop (data, side_length):
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
    
#    crop a square of desired size centered on the determined point
    square_result = data[y - (int(side_length/2)+1):y + (int(side_length/2)),x - (int(side_length/2)+1):x + (int(side_length/2))]
    print('Crop area is a square with ',side_length,'pixels on a side, from pixel ', x - (int(side_length/2)), 'in x, and pixel ',y - (int(side_length/2)), 'in y')
    
#    return the square crop area as an ndarray
    return square_result



# Load image into prgramme
data=funcy.load_img (input("File name to load: "))

# Calling the function         
test = max_intensity_crop(data,50)

# Showing the result
plt.imshow(test[:,:,0])
plt.show
