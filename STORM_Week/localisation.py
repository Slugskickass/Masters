import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.measure import regionprops, regionprops_table

data = np.load('thresholded_img.npy')
int_data = data.astype(np.uint32)
props = ['label', 'centroid', 'area']

locations = regionprops_table(int_data, properties=props)

table = pd.DataFrame(locations)


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

x = np.asarray(table['centroid-0'])
y = np.asarray(table['centroid-1'])

areas = np.asarray(table['area'])

smlr_than = np.where(areas <= 15)
bgr_than = np.where(areas > 1)

positions = np.intersect1d(smlr_than,bgr_than)


plt.imshow(data)
#plt.plot(y, x,'x')
plt.plot(y[positions], x[positions], 'rx')
plt.show

# def centre_collection(thresholded_data):
#     # label the image
#     label_image = label(thresholded_data, connectivity=1)
#     properties = ['area', 'centroid']
#
#     # Calculate the area
#     regions = regionprops_table(label_image, properties=properties)
#
#     # made a panda table, contains, 'area', 'centroid-0', 'centroid-1'
#     datax = pd.DataFrame(regions)
#     datax = datax[datax['area'] >= 0]      # at area >0 with std 5 can pick up all the appropriate intensities.
#     datax= datax[datax['area'] <= 5]
#     return datax

