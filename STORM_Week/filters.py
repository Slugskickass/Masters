import numpy as np
from scipy import signal
import math

### FILTER SWITCHER

def filter_switcher(data, settings):
    switcher = {
        'kernel' : kernel_filter,
        # (data, (settings.get("filter parameters:", {}).get("input parameter a"))),
        'DOG' : diff_of_gauss,
        # (data, (settings.get("filter parameters:", {}).get("input parameter a")),
        #                 (settings.get("filter parameters:", {}).get("input parameter b"))),
    }

    return switcher.get((settings.get("filter parameters:", {}).get("filter type:")), data)\
        (data, (settings.get("filter parameters:", {}).get("input parameter a")), (settings.get("filter parameters:", {}).get("input parameter b")))


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
### KERNEL FILTER

# Takes in a single file and the matrix/kernel data and convolutes them returning the processed image as a numpy array.
# The kernel data is in list of form to be converted within the array to a matrix.
# e.g. matrix = [(1, 2, 1), (2, 4, 2), (1, 2, 1)]

def kernel_filter(data, matrix, empty):
    image = data
    kernel = np.asarray(matrix)

    # Error check in case the matrix has an even number of sides.
    if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        print("The matrix has an even number of rows and/or columns. Please make them odd and run again.")

    if sum(sum(kernel)) != 1:       # Quick check to ensure the kernel matrix is within parameters.
        print("Error, this matrix's summation value is not equal to 1. This can change the final image.")
        print("The program has divided the matrix by the sum total to return it to a value of 1.")
        print(("This total value is: " + str(sum(sum(kernel)))))
        kernel = kernel / sum(sum(kernel))
        print(kernel)

    # Takes the filter size and allows for a rectangular matrix.
    edge_cover_v = (kernel.shape[0] - 1) // 2
    edge_cover_h = (kernel.shape[1] - 1) // 2

    # to determine if the file has multiple frames or not.
    if image.shape[2] > 1:
        # adds an edge to allow pixels at the border to be filtered too.
        bordered_image = np.pad(image, ((edge_cover_v, edge_cover_v), (edge_cover_h, edge_cover_h), (0, 0)))
        # Our blank canvas below.
        processed_image = np.zeros((bordered_image.shape[0], bordered_image.shape[1], bordered_image.shape[2]))

        # Iterates the z, x and y positions.
        for z in range(0, bordered_image.shape[2]):
            for x in range(edge_cover_h, bordered_image.shape[1] - edge_cover_h):
                for y in range(edge_cover_v, bordered_image.shape[0] - edge_cover_v):
                    kernel_region = bordered_image[y - edge_cover_v:y + edge_cover_v + 1,
                                    x - edge_cover_h:x + edge_cover_h + 1, z]
                    k = (kernel * kernel_region).sum()
                    processed_image[y, x, z] = k
        # Cuts out the image to be akin to the original image size.
        processed_image = processed_image[edge_cover_v:processed_image.shape[0] - edge_cover_v,
                          edge_cover_h:processed_image.shape[1] - edge_cover_h, :]

    else:
        # adds an edge to allow pixels at the border to be filtered too.
        bordered_image = np.pad(image, ((edge_cover_v, edge_cover_v),(edge_cover_h, edge_cover_h)))
        # Our blank canvas below.
        processed_image = np.zeros((bordered_image.shape[0], bordered_image.shape[1]))

        # Iterates the x and y positions.
        for x in range(edge_cover_h, bordered_image.shape[1]-edge_cover_h):
            for y in range(edge_cover_v, bordered_image.shape[0]-edge_cover_v):
                kernel_region = bordered_image[y-edge_cover_v:y+edge_cover_v+1, x-edge_cover_h:x+edge_cover_h+1]
                k = (kernel * kernel_region).sum()
                processed_image[y,x] = k
        # Cuts out the image to be akin to the original image size.
        processed_image = processed_image[edge_cover_v:processed_image.shape[0]-edge_cover_v, edge_cover_h:processed_image.shape[1]-edge_cover_h]
    return processed_image


#
#
#
#