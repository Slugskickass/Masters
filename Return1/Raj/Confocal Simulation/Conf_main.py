import numpy as np
import Confocal_Samurai as sam
import matplotlib.pyplot as plt
import microscPSF as msPSF
from scipy import signal

# Made a point in centre of 2D array
point = np.zeros((51, 51))
point[25, 25] = 1
point[15,15] = 1

# Made a 3D PSF
radPSF= sam.radial_PSF(100, 0.05)
radPSF = np.moveaxis(radPSF, 0, -1)

scan = np.zeros((radPSF.shape[1], radPSF.shape[0], radPSF.shape[2]))
scan = np.rot90(sam.kernel_filter(radPSF, point),2)

# Check for flatness in single particle.
# check = scan[:,:,100]-radPSF[:,:,100]
# print("minimum: " + str(np.min(check)), "maximum: " + str(np.max(check)))

# Made a gaussian. (Our spacial filter)
spacial_filter = sam.Gaussian_Map((51, 51), 0, 0, 0, 5, 1)

pinhole = np.zeros((scan.shape[1], scan.shape[0], scan.shape[2]))
for i in range(0, scan.shape[2]):
    pinhole[:, :, i] = sam.kernel_filter(spacial_filter, scan[:, :, i])

position = 100
plt.subplot(141)
plt.imshow(point)
plt.title("Point")
plt.subplot(142)
plt.imshow(radPSF[:,:,position])
plt.title("Radial PSF")
plt.subplot(143)
plt.imshow(scan[:,:,position])
plt.title("Scanned")
plt.subplot(144)
plt.imshow(pinhole[:,:,position])
plt.title("Pinhole")
plt.text(-150,-30,"Frame " + str(position) + " of 201")
plt.show()
