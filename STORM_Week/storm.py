import json
import general as genr
import filters
import thresholds
import localisation as loci
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime


### IMPORT SECTION ####
#json file builder (should be easy to adapt to accept any filter and its inputs as param_a and param_b)
parameters = {
        "directory:" : "/Users/RajSeehra/University/Masters/Semester 2/test folder",
        "extension:" : ".tif",
        "filter parameters:" : {
                "filter type:" : "kernel",
                "input parameter a" : [(0, -1, 0), (-1, 5, -1), (0, -1, 0)],
                "input parameter b" : ''
                },
        "threshold parameters:" : {
                "threshold type:" : "wavelet",
                "input parameter" : 5
                },
        "localisation parameters:" : {
                        "lower bound:" : 0,
                        "upper bound:" : 5
        }
}

with open("params.json", "w") as write_file:
    json.dump(parameters, write_file)

# Import filter parameters from json file
with open("params.json", "r") as read_file:
    params = json.load(read_file)


### LOAD IN THE DATA ###

# Create a list of files with the specified file extension within the specified directory
file_list = []
for file in os.listdir(params["directory:"]):
    if file.endswith(params["extension:"]):
        file_list.append(file)

# For each file in the above list, execute the chosen filter and threshold and then save it out as a numpy array

a = 0   # Set up a counter

folder = "{}/storm_output_data".format(params["directory:"])
if not os.path.exists(folder):
    os.mkdir(folder)        # Makes a directory is there isn't one there.
    
for name in file_list:
    a+= 1
    file_name = "{}/{}".format(params["directory:"], name)
    print(file_name)
    img = genr.load_img(file_name)
    
    
    ### FILTERING ###
    # This takes the data and the filter params information and pulls out the relevant information to choose which
    # function to run. Based on the "filter type:", and uses the parameters a and b as required.
    # Matt, at the moment it does not account for your above if statement.

    filtered_data = filters.filter_switcher(img, params)
    
    
    ### THRESHOLDING ###
    # As above, with switcher adapted to thresholds
    thresholded_data = thresholds.threshold_switcher(filtered_data, params)

    # np.save('{}/thresholded_img_{}_{}'.format(folder, a, datetime.datetime.now()), thresholded_data)


    ### LOCALISATION ###
    local = loci.centre_collection(thresholded_data, int(params.get("localisation parameters:", {}).get("lower bound:"))\
                                   , int(params.get("localisation parameters:", {}).get("upper bound:")))
    local.to_csv('{}/panda_data_{}_{}'.format(folder, a, datetime.datetime.now()))




#plt.imshow(thresholded_data)
#plt.show