#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 18:59:17 2020

@author: mattarnold
"""

import numpy as np
import matplotlib.pyplot as plt
import math


width = 33
height = 1 / (2 * (math.pi) * (width**2))

xo = yo = 50
 
kernel = np.zeros((100,100)) #PICKED SIZE ARBITRARILY... IS THIS THE RIGHT NUMBER?
for x in range(np.size(kernel,1)):
    for y in range(np.size(kernel,0)):
        kernel[x, y] = height * np.exp(-1 * ((((x-xo)**2) / 2 * (width**2)) + (((y-yo)**2) / 2 * (width**2))))

print(np.sum(kernel))
print(height)
plt.imshow(kernel)
plt.show