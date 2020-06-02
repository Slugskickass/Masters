#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 14:05:21 2020

@author: mattarnold
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.fft as fft
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.ndimage import gaussian_filter
from scipy import signal
import scipy.special
from scipy.optimize import minimize

#Build PSF using preset scaling factor and size
def build_psf_otf(scale, size):
    x = np.linspace(0, size - 1, size)
    y = np.linspace(0, size - 1, size)
    x = x.astype(int)
    y = y.astype(int)
    [X, Y] = np.meshgrid(x, y)

    # Generate the PSF --FORGOTTEN WHAT THEY ARE DOING HERE! no gaussian model?
    R = np.sqrt(np.minimum(X, np.abs(X - size)) ** 2 + np.minimum(Y, np.abs(Y - size)) ** 2)
    yy = np.abs(2 * scipy.special.jv(1, scale * R + np.finfo(np.float32).resolution) / (scale * R + np.finfo(np.float32).resolution)) ** 2 # What does this do again?
    psf = fft.fftshift(yy)

    # Generate OTF
    OTF2d = fft.fft2(yy)
    OTF2dmax = np.max(np.abs(OTF2d))
    OTF2d = OTF2d / OTF2dmax #Is this just to normalise the OTF? If so, why?
    normalised_otf = np.abs(fft.fftshift(OTF2d))

    return (psf, normalised_otf)

def get_image(filename, frame):
    image = Image.open(filename)
    image.seek(frame)
    img_array = np.asarray(image)
    return(img_array)

def otf_smooth(system_otf):
    w = np.shape(system_otf)[0]
    wo = np.int(w / 2)
    OTF1 = system_otf[wo+1, :]
    OTFmax = np.max(np.abs(system_otf))
    OTFtruncate = 0.01
    i = 1
    while (np.abs(OTF1[i]) < OTFtruncate * OTFmax):
        smoothed_otf = wo + 1 - i
        i = i + 1
    return smoothed_otf

def ApproxFreqDuplex(raw_img_fft,otf_cutoff):
    #% AIM: approx. illumination frequency vector determination
    #% INPUT VARIABLES
    #%   raw_img_fft: FT of raw SIM image
    #%   otf_cutoff: OTF cut-off frequency
    #% OUTPUT VARIABLES
    #%   approx_k_vector: illumination frequency vector (approx)
    #%   freq_x,freq_y: coordinates of illumination frequency peaks

    raw_img_fft = np.abs(raw_img_fft)

    w = np.shape(raw_img_fft)[0]
    wo = w/2
    x = np.linspace(0,w-1,w)
    y = np.linspace(0,w-1,w)
    [X, Y] = np.meshgrid(x,y)

    Ro = np.sqrt( (X-wo)**2 + (Y-wo)**2 ) # What is Ro?
    Z0 = Ro > np.round(0.5 * otf_cutoff)
    Z1 = X > wo

    raw_img_fft = raw_img_fft * Z0 * Z1 # Uses multiplication by false to eliminate DC and half of FT


#    dumY = np.max( raw_img_fft,[],1 )
    dumY = np.max(raw_img_fft, axis=0) #Find max value in Y of the "half-moon" masked FFT

#    [dummy freq_y] = max(dumY);
    freq_y = np.argmax(dumY) #Find the position of the above value
#    dumX = max( raw_img_fft,[],2 );
    dumX = np.max(raw_img_fft, axis=1) #Ditto, in x
#    [dummy freq_x] = max(dumX);
    freq_x = np.argmax(dumX)


    approx_k_vector = np.zeros(2)
    approx_k_vector[0] = np.abs(freq_x-(wo+1)+1) ## remove abs? WHY IS IT -1 AND +1?
    approx_k_vector[1] = np.abs(freq_y-(wo+1)+1)
    return(approx_k_vector, freq_x, freq_y, raw_img_fft)

def PhaseKai2opt(k_vector,noisy_original_image_fft, system_otf):
    w = np.shape(noisy_original_image_fft)[0]
    wo = np.int(w / 2)

    noisy_original_image_fft = noisy_original_image_fft * (1 - 1 * system_otf**10) #Increase the contrast by denoising
    noisy_original_image_fft = noisy_original_image_fft * np.conj(system_otf) #Build term for minimisation (highlights places where i term is similar)
    otf_cutoff = otf_smooth(system_otf)
    DoubleMatSize = 0

    if (2 * otf_cutoff > wo):  #Contingency for the situtation where the size of the FFT is not large enough to fit in extra frequency info from SIM reconstruction
        DoubleMatSize = 1

    if (DoubleMatSize > 0):
        t = 2 * w

        noisy_original_image_fft_temp = np.zeros((t, t))
        noisy_original_image_fft_temp[wo : w + wo, wo : w + wo] = noisy_original_image_fft
        noisy_original_image_fft = noisy_original_image_fft_temp
    else:
        t = w

    to = np.int(t / 2)
    u = np.linspace(0, t - 1, t)
    v = np.linspace(0, t - 1, t)
    [U, V] = np.meshgrid(u, v)

    # Build term for comparison in cross-correlation (image with frequency added to it)
    noisy_image_freqadd = np.exp(-1j * 2 * np.pi * (k_vector[1] / t * (U - to)) + (k_vector[0] / t * (V - to))) * fft.ifft2(noisy_original_image_fft)
    noisy_image_freqadd_fft = fft.fft2(noisy_image_freqadd)

    mA = np.longlong(np.sum(noisy_original_image_fft * np.conj(noisy_image_freqadd_fft ))) # Sum across pixels of product of image with complex conjugate with frequency introduced.
    mA = mA / np.longlong((np.sum(noisy_image_freqadd_fft * np.conj(noisy_image_freqadd_fft )))) # Normalising cross-correlation term
    #print(type(mA))
    #print(-np.abs(mA))
    correlation_FOM = -abs(mA) # Negative absolute value allows for minimisation; FOM = figure of merit

    return(correlation_FOM)

def x_optimise(k_vector,noisy_original_image_fft, system_otf):
#    number = 200
#    start = 290
#    stop = 310
#    correlation_FOM = np.zeros(number)
#    points = np.linspace(start, stop, number)
#    for index, I in enumerate(points):
#        k_vector[0] = I
#        correlation_FOM[index] = PhaseKai2opt(k_vector, noisy_original_image_fft, system_otf)

    res = minimize(PhaseKai2opt, x0=k_vector, args=(noisy_original_image_fft, system_otf), method='Nelder-Mead', tol=0.00001)
    res_list = [(np.sqrt(res.x[0]**2 + res.x[1]**2)),(np.arctan2(-1*res.x[0], res.x[1]))]
    return(res_list)

def y_optimise(k_vector,noisy_original_image_fft, system_otf):
#    number = 200
#    start = 290
#    stop = 310
    correlation_FOM = np.zeros(number)
#    points = np.linspace(start, stop, number)
#    for index, I in enumerate(points):
#        k_vector[1] = I
    correlation_FOM = PhaseKai2opt(k_vector, noisy_original_image_fft, system_otf)

    res = minimize(PhaseKai2opt, x0=k_vector, args=(noisy_original_image_fft, system_otf), method='Nelder-Mead', tol=0.00001)
    res_list = [(np.sqrt(res.x[0]**2 + res.x[1]**2)),(np.arctan2(-1*res.x[0], res.x[1]))]
    return(res_list)

if __name__ == "__main__":
    #File to process
    filename = '/Users/mattarnold/Masters/SIM_Week/Data/SLM-SIM_Tetraspeck200_680nm.tif'
    
    #Extract images
    image_data = get_image(filename, 0)
    
    plt.imshow(image_data)
    plt.show
    
    #Define image parameters
    size = np.shape(image_data)[0]
    scale = 0.99
    half_size = np.int(size/2)
    
    #Perform image operations to provide inputs for frequency determination
    PSFo, system_otf = build_psf_otf(scale, size)
    noisy_original_image_fft = fft.fftshift(fft.fft2(image_data))
    smoothed_otf = otf_smooth(system_otf)
    
    
    # Approx illumination frequency vector
    [k_vector, ta, tb, test] = ApproxFreqDuplex(noisy_original_image_fft, smoothed_otf)
    #print(np.sqrt(k_vector[0]**2 + k_vector[1]**2))
    #print(np.arctan2(-1*k_vector[0], k_vector[1]))
    
    #Iterative minimisation for frequency in x and y
    resx = x_optimise(ta,noisy_original_image_fft, system_otf)
    resy = y_optimise(tb,noisy_original_image_fft, system_otf)
    
    print(resx)
    print(resy)