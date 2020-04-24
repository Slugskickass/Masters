import general as genr
import pandas as pd
import numpy as np
import os


def crop_switcher(localised_data, settings):
    switcher = {
        'raj': raj_cropper,
        'matt': matt_cropper,
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


## Matt
    ## Currently, my cropping function loads a .csv file in the formatted outputted by the localisation function
    ## This can be changed if required
    
def point_crop (data, y, x, side_val):    
#    set crop square parameters
    side_length = int(side_val)
    half_side = int(side_length/2)
#    contingency for if square would exit frame area
    if y - side_length < 0: 
        y = 0 + half_side + 1
    if y + side_length > data.shape[1]:
        y = (data.shape[1])-half_side
    if x - side_length < 0:
        x = 0 + half_side + 1
    if x + side_length > data.shape[0]:
        x = (data.shape[0])-half_side
#    frame crop performed
    square_result = data[y - (half_side+1):y + half_side,x - (half_side+1):x + half_side]
    
#    return the square crop area as an ndarray
    return square_result

def load_csv(csv_pathname):
    # Load data from .csv to pandas dataframe
    molecules = pd.read_csv(csv_pathname)
    molecules = molecules.loc[:,["area", "centroid-0", "centroid-1", "file_name"]]

    # Create dataframe to save data into
    save_data = pd.DataFrame(columns=["file_name","x-coord","y-coord","cutout_square"])
    
    return molecules, save_data


def matt_cropper(csv_pathname = '{}/panda_data_molecule_locs{}'.format(os.getcwd(),'.csv')):
    # Iterate through indices in imported locations dataframe
    molecules, save_data = load_csv(csv_pathname)
    
    file = ""
    
    for index in molecules.index:

        # If the file name for current index does not match the current working file, load the file
        if file is not "{}".format(molecules.loc[index,"file_name"]):
            file = genr.load_img("{}".format(molecules.loc[index,"file_name"]))

        # Define crop area from centre of mass data in imported locations table    
        centre_y = int(np.floor(molecules.loc[index,"centroid-0"]))
        centre_x = int(np.floor(molecules.loc[index,"centroid-1"]))
        
        # Call cropping function
        molecule_crop = point_crop(file, centre_y, centre_x, 7)
        
        # Define variables for saving out data
        file_nm, centroid_one, centroid_zero = molecules.loc[index,"file_name"], molecules.loc[index,"centroid-1"], molecules.loc[index,"centroid-0"]
        
        # Format variables into a pandas series to allow saving to dataframe
        molecule_crop_id = pd.Series({"file_name":file_nm,"x-coord": centroid_one, "y-coord": centroid_zero ,"cutout_square": molecule_crop})
        
        # Convert series to dataframe, with index relating to current value of index variable
        save_row = pd.DataFrame([molecule_crop_id], index = ["{}".format(index)])
        
        # Concatenate the data from this iteration to the created data frame 
        save_data = pd.concat([save_row,save_data],axis=0)
    
    # Save finished dataframe to .csv in working directory
    save_data.sort_index(axis=0)
    save_data.to_csv("molecules_positions_crops.csv")
    return save_data


