import numpy as np
import matplotlib.pyplot as plt
import pywt




# image_size = [100,100,100]
# offset = 0
# centre_x = 0
# centre_y = 0
# centre_z = 0
# width = 0.035
# amplitude = 1000


def Gaussian_Map_3D(image_size, offset = 0, centre_x = 0, centre_y = 0, centre_z = 0, width = 0.04, amplitude = 1000):
    # Image Generation
    x, y, z = np.meshgrid(np.linspace(-1,1,image_size[1]), np.linspace(-1, 1, image_size[0]), np.linspace(-1, 1, image_size[2]))
    dist = np.sqrt((x-centre_x) ** 2 + (y+centre_y) ** 2 + (z+centre_z) ** 2)
    intensity = offset + amplitude * np.exp(-(dist ** 2 / (2.0 * width**2)))
    return intensity

mapper_1 = Gaussian_Map_3D([100, 100, 100], 0, 0, 0, 0, 0.035, 1000)

mapper_2 = Gaussian_Map_3D([100, 100, 100], 0, 0.1, 0.2, 0, 0.05, 1000)

mapper = mapper_1 + mapper_2
# Thresholder to clean up really low values for visualisation purposes
mapper= pywt.threshold(mapper, 0.005, substitute=0, mode='greater')

# plot loop.
for i in range(40, 60, 1):
    plt.subplot(4,5,i-39)
    plt.imshow(mapper[:,:,i])
    plt.title("Frame " + str(i))

plt.show()