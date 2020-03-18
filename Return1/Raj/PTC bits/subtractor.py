import numpy as np
import PTC_Samurai as sam
import matplotlib.pyplot as plt

# drkframe = sys.argv[1]
# folder = sys.argv[2]

# darkframes = np.load(drkframe)
darkframes = np.load("/Users/RajSeehra/University/Masters/Semester 2/Git Upload Folder/Return1/Raj/PTC bits/darkframes.npy")

# directory = sam.get_file_list(folder)
directory = sam.get_file_list("/Users/RajSeehra/University/Masters/Semester 2/test folder")

mean_std = []      # empty list which will collect the data in order: mean, std we want

for i in range (0, len(directory)):
    array = sam.loadtiffs(directory[i])
    for z in range (0, array.shape[2]):
        array[:, :, z] = array[:, :, z] - darkframes
    mean_std.append(np.mean(array)) # mean for the file
    mean_std.append(np.std(array))  # std for the file


mean_std = np.reshape(mean_std, (-1, 2)).T      # 0 = mean, 1 = std.

