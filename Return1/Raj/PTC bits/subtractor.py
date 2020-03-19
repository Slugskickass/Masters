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
crop_box = np.load("/Users/RajSeehra/University/Masters/Semester 2/Git Upload Folder/Return1/Raj/PTC bits/crop_box.npy")


def ptc_data(directory, darkframes, crop_box):
    mean_std = []      # empty list which will collect the data in order: mean, std we want
    x1 = crop_box[0,0]
    x2 = crop_box[0,1]
    y1 = crop_box[1,0]
    y2 = crop_box[1,1]

    for i in range (0, len(directory)):
        array = sam.loadtiffs(directory[i])
        for z in range (0, array.shape[2]):
            array[:, :, z] = array[:, :, z] - darkframes
        array = array[x1:x2, y1:y2, :]              # crop the stack to the stable window.
        mean_std.append(np.mean(array)) # mean for the file
        mean_std.append(np.std(array))  # std for the file

    mean_std = np.reshape(mean_std, (-1, 2)).T      # 0 = mean, 1 = std.

    return mean_std


data = ptc_data(directory, darkframes, crop_box)

np.save("PTCdata.npy", data)