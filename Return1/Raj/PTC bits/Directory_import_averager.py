import Samurai as sam
import numpy as np
import sys

# folder = sys.argv[1]
# directory = sam.get_file_list(folder)

directory = sam.get_file_list("/Users/RajSeehra/University/Masters/Semester 2/")
# directory = sam.get_file_list("/ugproj/raj/Flash4/")

means = []  # empty list which will collect the means data we want
# imports the data.
for i in range (0, len(directory)):
    mean = []       # empty list where means can be stored for each file.
    array = sam.loadtiffs(directory[i])
    for frame in range(np.size(array, 2)+1):      # calculates the mean for every frame.
        mean.append(np.mean(array[frame]))        # mean per frame
    means.append(np.mean(mean))                          # cleans up the mean data and appends to our means list
    mean = []

min_value_location = means.index(min(means))
print("The file with the lowest average intensity is: " + directory[min_value_location])
print("With an average intensity of: " + str(min(means)))
