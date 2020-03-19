import numpy as np
import PTC_Samurai as sam
import matplotlib.pyplot as plt
import sys

drkframe = sys.argv[1]
folder = sys.argv[2]

darkframes = np.load(drkframe)
# darkframes = np.load("/Users/RajSeehra/University/Masters/Semester 2/Git Upload Folder/Return1/Raj/PTC bits/darkframes.npy")

directory = sam.get_file_list(folder)
# directory = sam.get_file_list("/Users/RajSeehra/University/Masters/Semester 2/test folder")

# This file is generated using the Lowest_std_region.py on your brightest file.
crop_box = np.load("crop_box.npy")
# crop_box = np.load("/Users/RajSeehra/University/Masters/Semester 2/Git Upload Folder/Return1/Raj/PTC bits/crop_box.npy")


def ptc_data(directory, darkframes, crop_box):
    mean_values = np.zeros((100, 100, len(directory)))      # empty list which will collect the data in order: mean, std we want
    std_values = np.zeros((100, 100, len(directory)))
    x1 = int(crop_box[0,0])
    x2 = int(crop_box[0,1])
    y1 = int(crop_box[1,0])
    y2 = int(crop_box[1,1])

    for i in range (0, len(directory)):
        array = sam.loadtiffs(directory[i])
        for z in range (0, array.shape[2]):
            array[:, :, z] = np.subtract(array[:, :, z], darkframes)
        array = array[y1:y2, x1:x2, :]              # crop the stack to the stable window.
        mean_values[:, :, i] = np.mean(array, 2) # mean for the file
        std_values[:, :, i] = np.std(array, 2)  # std for the file

    return mean_values, std_values


mean_data, std_data = ptc_data(directory, darkframes, crop_box)

np.save("mean_Pixel_PTCdata.npy", mean_data)
np.save("std_Pixel_PTCdata.npy", std_data)