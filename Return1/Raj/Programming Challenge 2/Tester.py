import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

image = Image.open("/Users/RajSeehra/Downloads/pk_amb.jpg")

array = np.asarray(image)

array2 = np.sum(array,2)

for x in range(0,array2.shape[1]):
    for y in range(0,array2.shape[0]):
        if array2[y, x] >= 255:
            array2[y, x] = 0
        else:
            array2[y,x] = 1

pad_to_2k = np.pad(array2,((138,138),(128,128)))

np.save("ambiguous_image", pad_to_2k)
plt.imshow(pad_to_2k)
plt.show()
