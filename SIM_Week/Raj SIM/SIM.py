import SIM_Samurai as sam
import numpy as np
from matplotlib import cm
from scipy import signal
import scipy.fft
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


file = sam.loadtiffs("/Users/RajSeehra/University/Masters/Semester 2/Git Upload Folder/SIM_Week/Data/SLM-SIM_Tetraspeck200_680nm.tif")

fft = np.zeros((file.shape[1], file.shape[0], file.shape[2]), dtype=complex)
correls = np.zeros((file.shape[1], file.shape[0], file.shape[2]))

# establish psf and otf
psf = sam.psf_generator(1.2, 680, 1, 0.7)
otf = scipy.fft.fftshift(scipy.fft.fft2(psf))

for i in range(0, file.shape[2]):
    # Fourier transform the original image and shift the shape to the centre.
    fft[:, :, i] = scipy.fft.fft2(file[:, :, i])
    shifted = scipy.fft.fftshift(fft[:, :, i])

    # Multiply (convolve) the fourier image with the OTF + conjugate OTF
    multiply = shifted * np.conj(otf)

    # Correlate the fourier image with the complex conjugate and shift to the centre.
    correl = scipy.fft.fftshift(signal.correlate2d(fft[:, :, i], multiply, mode='same'))

    # Builds the array.
    correls[:, :, i] = np.abs(correl)

# Single Iteration
# fft[:, :, 0] = scipy.fft.fft2(file[:, :, 0])
# shifted = scipy.fft.fftshift(fft[:,:,0])
# multiply = sam.kernel_filter(shifted, otf)
# # multiply = signal.convolve2d(shifted, otf, mode='same')
# cmplx = np.conj(multiply)
# correl = scipy.fft.fftshift(signal.correlate2d(fft[:,:,0], cmplx, mode='same'))
# print(correl)
# correls[:, :, 0] = np.abs(correl)

# plt.subplot(231)
# plt.imshow(file[:,:,0])
# plt.title("Original")
# plt.subplot(232)
# plt.imshow(np.abs(shifted), cmap=cm.gray, norm=LogNorm(vmin=5))
# plt.title("Original FFT")
# plt.subplot(233)
# plt.imshow(np.abs(multiply), norm=LogNorm(vmin=5))
# plt.title("Multiplied")
# plt.subplot(234)
# plt.imshow(np.abs(otf), norm=LogNorm(vmin=5))
# plt.title("OTF")
# plt.subplot(235)
# plt.imshow(np.abs(correl))
# plt.title("correl")
# plt.subplot(236)
# plt.imshow(psf)
# plt.title("PSF")
# plt.show()

# # plot loop
# for i in range(0, file.shape[2]):
#     plt.subplot(3, 4, i+1)
#     plt.imshow(correls[:, :, i])
#     plt.title("Frame " + str(i))

# plt.imshow(correls[:, :, 0])
# plt.show()

np.save('SIM arrays', correls)
