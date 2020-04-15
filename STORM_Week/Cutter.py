import general as genr
from PIL import Image
import localisation as loci
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime


def square_area(img, xcoords, ycoords, size=11):
    square_size = int(size)
    if square_size%2 == 0:
        print("None odd number hence cutout is flawed to one side.")
        print("Size value has been altered by adding 1 to it.")
        square_size = square_size +1
    # Assign intial coords for image
    xcoordmin = xcoords - int(square_size//2)
    xcoordmax = xcoords + int(square_size//2)+1
    ycoordmin = ycoords - int(square_size//2)
    ycoordmax = ycoords + int(square_size//2)+1

    # check no negative numbers, correct the squares position if at an edge.
    if xcoordmin < 0:
        xcoordmin = 0
        xcoordmax = xcoordmin + square_size
    if xcoordmax > img.shape[1]:
        xcoordmax = img.shape[1]
        xcoordmin = xcoordmax - square_size
    if ycoordmin < 0:
        ycoordmin = 0
        ycoordmax = ycoordmin + square_size
    if ycoordmax > img.shape[0]:
        ycoordmax = img.shape[0]
        ycoordmin = ycoordmax - square_size

    # Plotting the area.
    return img[ycoordmin:ycoordmax, xcoordmin:xcoordmax]

data = pd.read_csv(
    "/Users/RajSeehra/University/Masters/Semester 2/test folder/storm_output_data/panda_data_8_2020-04-15 16:32:28.277292_.csv")

# Generate a file list from the data.
file_list = [data["file_name"][0]]
for i in range(0, data.shape[0]-1):
    if data["file_name"][i+1] == data["file_name"][i]:
        continue
    else:
        file_list.append(data["file_name"][i+1])

# Create an empty array to add the data to.
cutouts = np.zeros([11, 11, data.shape[0]])
cutout_dataframe = pd.DataFrame(columns=['frame', 'X', 'Y', 'filename'])

for i in range(0, len(file_list)):
    img = genr.load_img(file_list[i])   # loads in the file
    print(file_list[i])
    for j in range(0, data.shape[0]):
        if data["file_name"][j] == file_list[i]:
            y = data["centroid-0"][j]
            x = data["centroid-1"][j]
            print(x, y)

            cutouts[:, :, j] = square_area(img, x, y)
            cutout_current = pd.DataFrame({'frame': [j], 'X': [x],'Y': [y], 'filename': [file_list[i]]})
            cutout_dataframe = pd.concat([cutout_dataframe, cutout_current], axis=0)

# Currently Data is offset by the rounding issue with the localisation.

img2 = genr.load_img("/Users/RajSeehra/University/Masters/Semester 2/test folder/00008.tif")
plt.subplot(121)
plt.imshow(cutouts[:,:,4])
plt.subplot(122)
plt.imshow(img2)
plt.show()