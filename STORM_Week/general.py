#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 12:34:00 2020

@author: mattarnold
"""

import numpy as np
from PIL import Image as im 

def load_img (file_name): # Define function to load image
    img_locat = im.open(file_name)
#    print ('Image size (px)',img_locat.size)
#    print ('Number of frames',img_locat.n_frames)
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


def savetiffs(file_name, data):
    if data.shape[2] > 1:
        images = []
        for I in range(np.shape(data)[2]):
            images.append(im.fromarray(data[:, :, I]))
            images[0].save(file_name, save_all=True, append_images=images[1:])
    else:
        images = im.fromarray(data[:, :])
        images.save(file_name)
        
def gaussian(x, y, x0, y0, xalpha, yalpha, offset, A):
    return offset + (A * np.exp(-((x-x0)/xalpha)**2 - ((y-y0)/yalpha)**2))

def _gaussian(M, *args):
    x, y = M
    arr = gaussian(x, y, *args)
    return arr

