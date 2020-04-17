import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.morphology import closing, square
from skimage.measure import label, regionprops, regionprops_table


def centre_collection(thresholded_data, scale=1):
    thresholded_data = closing(thresholded_data > np.mean(thresholded_data) + scale * np.std(thresholded_data))
    # label the image
    label_image = label(thresholded_data, connectivity=2)
    properties = ['area', 'centroid']

    # Calculate the area and centroids
    # regionprops_table caluclates centroids but turns them to integers. It seems to use a floor system to do so.
    # This is suitable for our purposes.
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

