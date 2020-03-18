#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 13:40:41 2020

@author: mattarnold
"""
import numpy as np
from PIL import Image as im
import matplotlib.pyplot as plt
from skimage.morphology import closing, square
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops, regionprops_table
import pandas as pd
from scipy import signal
from matplotlib.colors import LogNorm
import os


#define a function to load an image
def load_img (file_name, dimensions=False):
    img_locat = im.open(file_name)
    
    if dimensions == True:
        print ('Image size (px)',img_locat.size)
        print ('Number of frames',img_locat.n_frames)
    
    img_array = np.zeros ((img_locat.size[1],img_locat.size[0],img_locat.n_frames), np.float32)
    for I in range(img_locat.n_frames):
        img_locat.seek(I)
        img_array [:,:,I] = np.asarray(img_locat)
    img_locat.close
    return img_array

#define a function to save an image or stack
def save_img(file_name, data):
    images = []
    for I in range(np.shape(data)[2]):
        images.append(im.fromarray(data[:, :, I]))
        images[0].save(file_name, save_all=True, append_images=images[1:])

    
#define a function to cut out a given region of a single frame of an image
def crop_square_2d (data,centre_x,centre_y,frame_of_interest,side_length):
    square_result = data[centre_y - (int(side_length/2)+1):centre_y + (int(side_length/2)),centre_x - (int(side_length/2)+1):centre_x + (int(side_length/2)),frame_of_interest]
    print('Crop area is a square with ',side_length,'pixels on a side, from pixel ', centre_x - (int(side_length/2)), 'in x, and pixel ',centre_y - (int(side_length/2)), 'in y')
    return square_result

#define a function to crop an identical square from a stack of images
def crop_square_stack (data,centre_x,centre_y,side_length):
    square_result = data[centre_y - (int(side_length/2)):centre_y + (int(side_length/2)),centre_x - (int(side_length/2)):centre_x + (int(side_length/2)),:]
    print('Crop area is a square with ',side_length,'pixels on a side, from pixel ', centre_x - (int(side_length/2)), 'in x, and pixel ',centre_y - (int(side_length/2)), 'in y')
    return square_result

#select a frame of a multi-frame image
def frame_sel (data,frame):
    selected_frame = data[:,:,frame]
    return selected_frame

#perform an FFT on an image stack/series
def fft_stack(data):
    stack_fft = np.zeros(np.size(data))
    fft_frame_list = []
    for I in range (np.size(data,2)):
        frame = frame_sel(data,I)
        frame_fft = np.fft.fftshift (np.fft.fft2(frame))
        fft_frame_list.append(frame_fft)
        stack_fft = np.dstack (fft_frame_list)
    return stack_fft

def stack_show (data,start_frame=None,end_frame=None):
    if end_frame <= np.size(data,2):
        for I in range (start_frame,end_frame):
             plt.imshow (data[:,:,I])
             plt.show ()
    else:
        for I in range (start_frame,np.size(data,2)):
            plt.imshow (data[:,:,I])
            plt.show ()
    
def fft_stack_show (data):
     for I in range (np.size(data,2)):
         plt.imshow(np.abs(data[:,:,I]), norm=LogNorm(vmin=5))
         plt.show ()
         
def get_file_list(dir,file_type=".tif"):
    file_list = []
    for file in os.listdir(dir):
        if file.endswith(file_type):
            file_name = dir + '/' + file
            file_list.append(file_name)
    return file_list






