#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 12:36:27 2020

@author: mattarnold
"""
import numpy as np

bbox_crop = np.array(np.mat('250 350; 250 350'))

np.save('crop_box',bbox_crop)