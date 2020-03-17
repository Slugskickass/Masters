#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 12:32:08 2020

@author: mattarnold
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import im_func_lib as funci


def dark_frame_avg(dir_loc):
    
    files = funci.get_file_list(dir_loc)
    
    file_int_avgs = []
    for I in range (len(files)):
        work_file = funci.load_img(files[I])
        work_file.astype('float64')
        avg_int = np.mean(work_file)
        file_int_avgs.append(avg_int)
    avg_array = np.asarray(file_int_avgs)
    
    result = np.where(avg_array == np.amin(avg_array))
    int_res = np.abs(result)
    dark_num = int_res[0,0]
    dark_frame = funci.load_img(files[dark_num])
    dark_frame.astype('float64')
    avg_dark_frame = np.mean(dark_frame,2)
    return avg_dark_frame
    
    

dir_loc = "/Users/mattarnold/iCloud Drive (Archive)/Documents/Documents – Matt’s MacBook Pro/Uni/Imaging Msc/semester 2/Python Module/PTC"
     
test = dark_frame_avg(dir_loc)

#np.save('dark_frame',test)