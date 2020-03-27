from PIL import Image
import numpy as np
import sys
import matplotlib.pyplot as plt

file = str(sys.argv[1])
size = sys.argv[2]

# open the file
file_name = file
img = Image.open(file_name)
# generate the array and apply the image data to it.
imgArray = np.zeros((img.size[1], img.size[0], img.n_frames), np.uint16)
imgArray[:, :, 0] = img
img.close()


def brightest_point(file):
    # find the brightest point(s). Makes a tuple.
    max_xy = np.where(file == np.amax(file))

    # zip the 2 arrays to get the exact coordinates from the above produced tuple.
    listOfCordinates = list(zip(max_xy[1], max_xy[0]))
    cord = listOfCordinates[0]      # selects the first points coords only
    xmax = cord[0]
    ymax = cord[1]

    return xmax,ymax


def square_area(img, size,xmax,ymax):
    square_size = int(size)
    # Assign intial coords for image
    xcoordmin = xmax - int(square_size/2)
    xcoordmax = xmax + int(square_size/2)+1
    ycoordmin = ymax - int(square_size/2)
    ycoordmax = ymax + int(square_size/2)+1

    # check no negative numbers, correct the squares position if at an edge.
    if xcoordmin < 0:
        xcoordmin = 0
        xcoordmax = xcoordmin + square_size
    if xcoordmax > img.size[0]:
        xcoordmax = img.size[0]
        xcoordmin = xcoordmax - square_size
    if ycoordmin < 0:
        ycoordmin = 0
        ycoordmax = ycoordmin + square_size
    if ycoordmax > img.size[1]:
        ycoordmax = img.size[1]
        ycoordmin = ycoordmax - square_size

    return xcoordmin,xcoordmax,ycoordmin,ycoordmax


def savetiffs(file_name, data):
    images = Image.fromarray(data[:, :])
    images.save(file_name)


# extract the x and y values
xmax, ymax = brightest_point(imgArray)[0], brightest_point(imgArray)[1]

# extract the coordinate positions of the new cut image
xcoordmin, xcoordmax, ycoordmin, ycoordmax = square_area(img, size, xmax,ymax)

# create our new array by cutting out using the above data
cutout_data = imgArray[ycoordmin:ycoordmax, xcoordmin:xcoordmax, 0]

# output the saved data
savetiffs("Cutout_data.tiff", cutout_data)

# print statement to confirm output
print("Your file has been generated from " + file_name + " with a size of " + str(size) + ".")


# Image Testing lines
# plt.imshow(cutout_data)
# plt.show()