import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import Samurai as sam

positions = pd.read_csv("/Images/Positions.csv")

#1. scatter plot of the data.
#plt.scatter(positions['x [nm]'],positions['y [nm]'])

#2. data as an array.
# positions_array = np.asarray(positions).T
# plt.plot(positions_array[0],positions_array[1], '.')

#3. data as an array crossed with a gaussian
positions_array = np.asarray(positions)
Gauss = sam.Gaussian_Map([np.shape(positions_array)[1], np.shape(positions_array)[0]], 0,0.5,0.5,0.4,100).T
#Take the fourier transform of the two images
positions_array_fft = np.fft.fft2(positions_array)
Gauss_fft = np.fft.fft2(Gauss)
#Now take the two fourier transforms and muliply them by each other
conv_fft = positions_array_fft * Gauss_fft
# perform the shift.
conv = np.real(np.fft.ifft2(conv_fft)).T
plt.plot(conv[0], conv[1], '.')


plt.show()