# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 04:45:35 2020

@author: 90759
"""
import numpy as np
from scipy import misc, ndimage,signal
import matplotlib.pyplot as plt
import scipy

def loadtiffs(file_name):
    img = Image.open(file_name)
    print('The Image is', img.size, 'Pixels.')
    print('With', img.n_frames, 'frames.')
    data=np.asarray(img)
    return data
def g(v, std):
    x = v[0]
    y = v[1]
    return  np.exp(-((x ** 2) + (y ** 2)) / (2 * std ** 2))# * (1.0 / (2 * np.pi * std ** 2))
#x is the axis of 121 data, from -3~3, k is the calculated value of the gaussian function at integer coordinates
x = np.linspace(-3, 3, 121)
xy = np.meshgrid(x, x)
k = g(xy, 5.0)
k = k[::20, ::20]

#load the image
img = loadtiffs('C:/Users/90759/Desktop/Teaching_python-master/Teaching_python-master/week_2/data/640.tif)
 #use convolve and fftconcolve to do the convolution to filter the image
img_con = signal.convolve(img, k, mode = "valid")
img_fft = signal.fftconvolve(img, k, mode = "valid")
#use sepfir2d to filter the image
w = signal.gaussian(7, std = 5.0)
img_sep = signal.sepfir2d(img, w, w)
#show the result
plt.gray()
plt.subplot(221, title = "orginal")
plt.imshow(img)
plt.subplot(222, title = "fftconvolve")
plt.imshow(img_fft)
plt.subplot(223, title = "sepfir2d")
plt.imshow(img_sep)
plt.subplot(224, title = "convolve")
plt.imshow(img_con)
