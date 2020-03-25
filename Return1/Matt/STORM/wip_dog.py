
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import math
from PIL import Image as im


def load_img (file_name):
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

def gauss_height (width): #This should calculate the amplitude of the Gaussian based on an area under the curve of 1
    height = 1 / (math.sqrt(2) * np.abs(width) * math.sqrt(math.pi)) 
    return height

def build_2d_gauss (width, height): #This builds the Gaussian based on the calculated parameters
    xo = yo = 50 
    kernel = np.zeros((100,100)) #PICKED SIZE ARBITRARILY... IS THIS THE RIGHT NUMBER?
    for x in range(width):
        for y in range(width):
            kernel[x, y] = height * np.exp(-1 * (((x-xo)**2 + (y-yo)**2)/width**2))
    return kernel

def diff_of_gauss (data, narrow_width, wide_width):
    height_narrow = gauss_height(narrow_width) #Calculate height from widths inputed
    height_wide = gauss_height(wide_width)
    narrow_kern = build_2d_gauss (narrow_width, height_narrow) #Build Gaussians
    wide_kern = build_2d_gauss (wide_width, height_wide)
    blur_matrix = narrow_kern - wide_kern #Find DOG
    if data.ndim > 2: #if multiple frames in image then:
        diff_gauss_img = np.zeros_like(data)
        for I in range(np.size(data,2)):
            diff_gauss_img[:,:,I] = signal.convolve2d(data[:,:,I],blur_matrix,mode='same',boundary='symm')
    else:
        diff_gauss_img = signal.convolve2d(data,blur_matrix,mode='same',boundary='symm') #convolve DOG with image
    return diff_gauss_img

data = load_img('/Users/mattarnold/Masters/Return1/Matt/STORM/cameraman.tif')


wide = 40
narrow = 5

DOG = diff_of_gauss(data,narrow,wide)

plt.imshow(data)
plt.imshow(DOG)
plt.show