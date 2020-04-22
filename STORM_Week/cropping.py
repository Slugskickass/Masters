import general as genr
import pandas as pd
import numpy as np


def crop_switcher(localised_data, settings):
    switcher = {
        'raj': raj_cropper,
        # 'matt': matt_cropper,
    }

    return switcher.get((settings.get("cropping parameters:", {}).get("Student_name:")), localised_data)(localised_data)


def square_area(img, xcoords, ycoords, size=11):
    square_size = int(size)
    if square_size % 2 == 0:
        print("None odd number hence cutout is flawed to one side.")
        print("Size value has been altered by adding 1 to it.")
        square_size = square_size + 1
    # Assign intial coords for image
    xcoordmin = xcoords - int(square_size // 2)
    xcoordmax = xcoords + int(square_size // 2) + 1
    ycoordmin = ycoords - int(square_size // 2)
    ycoordmax = ycoords + int(square_size // 2) + 1

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


def area_filter(data, lower_bound=0, upper_bound=5):
    data = data[data['area'] >= lower_bound]  # at area >0 with std 5 can pick up all appropriate intensities.
    data = data[data['area'] <= upper_bound]
    return data


def raj_cropper(data):
    ### BOUNDING ###
    # Remove the excess first column, an artifact from exporting a csv.
    # data = data.drop('Unnamed: 0', axis=1)

    # Filters the data set by the area to remove files that are above or below the thresholding limits.
    data = area_filter(data, 1, 2)
    data = data.reset_index()
    data = data.drop('index', axis=1)

    ### PROCESSING ###
    # Generate a file list from the data. As there are repeated filenames this compares the next file name to the current
    # and if they are different adds the next filename to the list.
    file_list = [data["file_name"][0]]
    for i in range(0, data.shape[0] - 1):
        if data["file_name"][i + 1] == data["file_name"][i]:
            continue
        else:
            file_list.append(data["file_name"][i + 1])

    # Create an empty array to add the data to.
    cutout_dataframe = pd.DataFrame(columns=['frame', 'X', 'Y', 'filename', 'cutout_array'])

    for i in range(0, len(file_list)):
        img = genr.load_img(file_list[i])  # loads in the file
        print(file_list[i])
        for j in range(0, data.shape[0]):
            if data["file_name"][j] == file_list[i]:
                y = data["centroid-0"][j]
                x = data["centroid-1"][j]
                print(x, y)

                cutouts = np.array(square_area(img, x, y, 7))
                cutout_current = pd.DataFrame({'frame': [j], 'X': [x], 'Y': [y], 'filename': [file_list[i]],
                                               'cutout_array': [cutouts]}, index=["{}".format(j)])
                cutout_dataframe = pd.concat([cutout_dataframe, cutout_current], axis=0)  # Final dataframe.

    # Currently Data is intergerised by the DCOL_type issue with regionprops_table. All values are rounded down.
    return cutout_dataframe

