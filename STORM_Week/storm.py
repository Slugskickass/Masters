import json
import general as genr
import filters

### IMPORT SECTION ####
#json file builder (should be easy to adapt to accept any filter and its inputs as param_a and param_b)
parameters = {
        "file name:" : "/Users/RajSeehra/University/Masters/Semester 2/Git Upload Folder/STORM_Week/640.tif",
        "filter parameters:" : {
                "filter type:" : "DOG",
                "input parameter a" : 40,
                "input parameter b" : 40,
        }
}

with open("params.json", "w") as write_file:
    json.dump(parameters, write_file)

# Import filter parameters from json file
with open("DOG_params.json", "r") as read_file:
    params = json.load(read_file)

###UNPACKING### (Might remove)
# file = params['file name:'] # Unpack file name to variable "file"
# filter_params = params['filter parameters:'] # Unpack parameters
# filter_type = filter_params['filter type:']
# param_a = filter_params['input parameter a'] # save inputted widths as variables
# param_b = filter_params['input parameter b']
#
# # Command to determine desired filter
# if filter_type == "DOG":
#     # Determine input params
#     wide, narrow = filters.dog_params(param_a, param_b)
#
#     # Load image as array
#     data = genr.load_img(file)
#
#     # Perform filtering operation
#     DOG = filters.diff_of_gauss(data,narrow,wide)
#
# #
##

###LOAD IN THE DATA

file = genr.load_img(params["file name:"])


###FILTERING####
# This takes the data and the filter params information and pulls out the relevant information to choose which
# function to run. Based on the "filter type:", and uses the parameters a and b as required.
# Matt, at the moment it does not account for your above if statement.


filtered_data = filters.filter_switcher(file, params)



