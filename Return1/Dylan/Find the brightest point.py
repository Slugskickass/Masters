# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 04:20:34 2020

@author: 90759
"""

from PIL import Image
import numpy as np
from numpy import unravel_index

#define a dunction to load the image
def loadtiffs(file_name):
    img = Image.open(file_name)
    print('The Image is', img.size, 'Pixels.')
    print('With', img.n_frames, 'frames.')
    data=np.asarray(img)
    return data

#define a function to find the brightest point of the image
def find_max(img):
    ind_1d = np.argmax(img)
    print('1d max index: {}'.format(ind_1d))
    ind_2d = unravel_index(ind_1d, img.shape)
    print('2d max index: {}'.format(ind_2d))
    return ind_2d

#load the image
img = loadtiffs('C:/Users/90759/Desktop/Teaching_python-master/Teaching_python-master/Images/bacteria.tif')

#find the index of the brightest point and get a 3*3 area around it
inds = find_max(img)
w,h=inds
img_1=img[w-3:w+3, h-3:h+3]

#save the image
img_2=Image.fromarray(img_1)
img_2.save('cutout.tif')








#a=np.asarray(img)