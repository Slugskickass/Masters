import PTC_Samurai as sam
import numpy as np
import sys

# folder = sys.argv[1]
# directory = sam.get_file_list(folder)

directory = sam.get_file_list("/Users/RajSeehra/University/Masters/Semester 2/")
# directory = sam.get_file_list("/ugproj/raj/Flash4/")

means = []  # empty list which will collect the means data we want
# imports the data.
for i in range (0, len(directory)):
    array = sam.loadtiffs(directory[i])
    means.append(np.mean(array))  # mean for the file

min_value_location = means.index(min(means))
print("The file with the lowest average intensity is: " + directory[min_value_location])
print("With an average intensity of: " + str(min(means)))

max_value_location = means.index(max(means))
print("The file with the greatest average intensity is: " + directory[max_value_location])
print("With an average intensity of: " + str(max(means)))
