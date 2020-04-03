#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 12:38:02 2020

@author: mattarnold
"""
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.measure import regionprops, regionprops_table

data = np.load('thresholded_img.npy')
int_data = data.astype(np.uint32)
props = ['label','centroid','area','major_axis_length','solidity']

molecules = regionprops_table(int_data,properties=props)

table = pd.DataFrame(molecules)


x = np.asarray(table['centroid-0'])
y = np.asarray(table['centroid-1'])


areas = np.asarray(table['area'])
positions1 = np.where(areas >= 20)
positions2 = np.where(areas <= 40)
positions = np.intersect1d(positions1, positions2)

plt.imshow(data > 0)
plt.plot(y, x,'x')
plt.plot(y[positions], x[positions], 'bo')

plt.show

print(table)