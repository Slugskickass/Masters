import numpy as np
import matplotlib.pyplot as plt
import SIM_Samurai as sam
import pywt
import scipy
import pandas as pd
from skimage.morphology import closing, square
from skimage.measure import label, regionprops, regionprops_table


# IMPORTS
file = sam.loadtiffs("/Users/RajSeehra/University/Masters/Semester 2/Git Upload Folder/SIM_Week/Data/SLM-SIM_Tetraspeck200_680nm.tif")
arrays = np.load("/Users/RajSeehra/University/Masters/Semester 2/Git Upload Folder/SIM_Week/Raj SIM/SIM arrays.npy")

# SET UP
array = arrays[:,:,1]
gaussian = [(1,2,1),(2,4,2),(1,2,1)]
array = pywt.threshold(array, np.mean(array) + 5*np.std(array), mode = "greater")
# array = sam.kernel_filter(array, gaussian)

# PARAMETERS FOR CIRCLE MASK
a, b = array.shape[0]/2, array.shape[1]/2
n = array.shape[1]
r = array.shape[1]/6
# Produce circle mask, ones grid = to original file and cut out.
y, x = np.ogrid[-a:n-a, -b:n-b]
mask = x*x + y*y <= r*r
ones= np.ones((array.shape[1], array.shape[0]))
ones[mask] = 0
# multiply element wise by original array.
multiply = array * ones

thresholded_data = closing(multiply > 0)
label_image = label(thresholded_data, connectivity=2)
properties = ['centroid']

# Calculate the area and centroids
regions = regionprops_table(label_image, properties=properties)

# made a panda table, contains, 'area', 'centroid-0', 'centroid-1'
datax = pd.DataFrame(regions)

# Position Data
laser1 = (datax['centroid-1'][0], datax['centroid-0'][0])
laser2 = (datax['centroid-1'][1], datax['centroid-0'][1])

# Relative distance from the centre in x and y. -1 included to account for python numbering.
centre_dist_coords = (array.shape[1]/2-laser1[0]-1, array.shape[0]/2-laser1[1]-1)
centre_dist_coords2 = (array.shape[1]/2-laser2[0]-1, array.shape[0]/2-laser2[1]-1)

# Linear Distances
centre_dist1 = np.sqrt(centre_dist_coords[0]**2 + centre_dist_coords[1]**2)
centre_dist2 = np.sqrt(centre_dist_coords2[0]**2 + centre_dist_coords2[1]**2)

# Angles.
angle1 = -(180/np.pi)*np.arctan(centre_dist_coords[1]/centre_dist_coords[0])
angle2 = (180/np.pi)*np.arctan(centre_dist_coords2[1]/centre_dist_coords2[0])


# Frequency = 1/ absolute distance from centre. ? how pixel distance affects the result
frequency1 = 1/(centre_dist1 * 0.000097)
frequency2 = 1/(centre_dist2 * 0.000097)

# model the space.
inten1 = file[laser1[1],laser1[0],0]
inten2 = file[laser2[1],laser2[0],0]

model = np.zeros((array.shape[1], array.shape[0]))
model[laser1[1], laser1[0]] = inten1
model[laser2[1], laser2[0]] = inten2



plt.imshow(array)
# plt.imshow(np.abs(scipy.fft.fftshift(scipy.fft.ifft2(model))))
plt.show()
