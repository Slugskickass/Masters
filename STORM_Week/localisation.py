import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops, regionprops_table

# data = np.load('/Users/RajSeehra/University/Masters/Semester 2/test folder/storm_output_data/thresholded_img_1_2020-04-06 20:15:04.919026.npy')
# int_data = data.astype(np.uint32)
# props = ['label', 'centroid', 'area']

# locations = regionprops_table(int_data, properties=props)
#
# table = pd.DataFrame(locations)


#x = np.asarray(table['centroid-0'])
#y = np.asarray(table['centroid-1'])
#
#
#areas = np.asarray(table['area'])
#positions1 = np.where(areas >= 20)
#positions2 = np.where(areas <= 40)
#positions = np.intersect1d(positions1, positions2)
#
#plt.imshow(data > 0)
#plt.plot(y, x,'x')
#plt.plot(y[positions], x[positions], 'bo')

#plt.show

#lower_bound = table[table.iloc[:,3] > 3]
#upper_bound = lower_bound[lower_bound.iloc[:,3] < 15]

# x = np.asarray(table['centroid-0'])
# y = np.asarray(table['centroid-1'])
#
# areas = np.asarray(table['area'])
#
# smlr_than = np.where(areas <= 15)
# bgr_than = np.where(areas > 1)
#
# positions = np.intersect1d(smlr_than,bgr_than)
#
#
# plt.imshow(data)
# #plt.plot(y, x,'x')
# plt.plot(y[positions], x[positions], 'rx')
# plt.show

def centre_collection(thresholded_data, lower_bound=0, upper_bound=5):
    # label the image
    label_image = label(thresholded_data, connectivity=1)
    properties = ['area', 'centroid']

    # Calculate the area
    regions = regionprops_table(label_image, properties=properties)

    # made a panda table, contains, 'area', 'centroid-0', 'centroid-1'
    datax = pd.DataFrame(regions)

    # datax = datax[datax['area'] >= lower_bound]      # at area >0 with std 5 can pick up all appropriate intensities.
    # datax= datax[datax['area'] <= upper_bound]
    return datax


# centres = centre_collection(int_data)

# Viewing Purposes
# plt.subplot(121)
# plt.imshow(int_data[:,:])
# plt.subplot(122)
# plt.imshow(int_data)
# plt.plot(centres['centroid-1'], centres['centroid-0'], '*')
# plt.show()

