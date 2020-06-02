import numpy as np
import matplotlib.pyplot as plt
import Camera_Samurai as sam
from scipy import signal as sig

### INPUT PARAMETERS ###
groundpixel = 5                 # Pixel size e.g. pixels = 5nm
photon_count = 10               # Photon count value per unit time (same unit used in exposure)
exposure_time = 1               # seconds of exposure
NA = 1.4                        # Numerical aperture
wavelength = 440                # Wavelength in nanometres
camera_pixel_size = 6500        # Camera pixel size in nanometres. usual sizes = 6 microns or 11 microns
magnification = 100             # Lens magnification
read_mean = 2                   # Read noise mean level
read_std = 2                    # Read noise standard deviation level
QE = 0.95                       # Quantum Efficiency
fixed_pattern_deviation = 0.001 # Fixed pattern standard deviation. usually affects 0.1% of pixels.
gain = 2                        # Camera gain. Usually 2 per incidence photon

# Make a Ground Truth
# Based on pixel size. e.g. 5nm pixel, therefore: 10 microns = 2kx2k array
ground_window = 10000 // groundpixel
ground = np.zeros((ground_window, ground_window))


### SAMPLE GENERATION ###
# Boring Sample
# ground[1000:2000, 1000] = photon_count * exposure_time

# Interesting Sample. Pleas kindly relink the "ambiguous_image.npy" to the np.load if you wish to view this.
ground = np.load("/Users/RajSeehra/University/Masters/Semester 2/Git Upload Folder/Return1/Raj/Programming Challenge 2/ambiguous_image.npy")
for x in range(0, ground.shape[1]):
    for y in range(0,ground.shape[0]):
        if ground[y, x] == 1:
            ground[y, x] = photon_count * exposure_time
#### END AWESOME SAMPLE MAKER ####


### LENS SIMULATOR ###
# Lens == A diffraction limited blur. Dependent on wavelength and NA
psf = sam.psf_generator(NA, wavelength, groundpixel, ground_window)

# Apply the lens as a convolution of the two arrays.
# dif_lim = sam.kernel_filter_2D(ground, psf)       # Slow method.
dif_lim = sig.fftconvolve(ground, psf, "same")            # Fiddling with fourier space to convolute, much faster.


### CAMERA SETUP ###
# Camera sensor, based on optical magnification and pixel size.
camerapixel_per_groundpixel = camera_pixel_size/groundpixel

# Used to determine the number of the ground pixels that exist within each bin
mag_ratio = camerapixel_per_groundpixel / magnification


### IMAGING TIME ###
# Initialise an empty array, with a size calculated by the above ratios
camera_image = np.zeros((int(dif_lim.shape[0]//mag_ratio), int(dif_lim.shape[1]//mag_ratio)))
# Iterate each position in the array and sum the pixels in the range from the diffraction limited image.
# We use the mag_ratio to step across the array and select out regions that are multiples of it out.
for y in range(0, camera_image.shape[0]):
    for x in range(0, camera_image.shape[1]):
        pixel_section = dif_lim[y*int(mag_ratio):y*int(mag_ratio)+int(mag_ratio), x*int(mag_ratio):x*int(mag_ratio)+int(mag_ratio)]
        camera_image[y, x] = np.sum(pixel_section)

# Account for Quantum efficiency.
camera_image = camera_image * QE


### ADD NOISE ###
# Add read and shot noise.
camera_Rnoise = sam.read_noise(camera_image, read_mean, read_std)
camera_Snoise = sam.shot_noise(np.sqrt(photon_count * exposure_time), camera_image)     # Fix this.

# Add up the camera, read and shot noises.
camera_RSnoise = camera_image + camera_Rnoise + camera_Snoise

# FP noise is remains the same for the camera.
np.random.seed(100)
camera_FPnoise = np.random.normal(1, fixed_pattern_deviation, (camera_image.shape[0], camera_image.shape[1]))

# Multiply by the fixed pattern noise
camera_all_noise = camera_RSnoise * camera_FPnoise


### GAIN, COUNT AND INTEGER ###
# Multiply by gain to convert from successful incidence photons and noise to electrons.
camera_gain = camera_all_noise * gain

# 100 count added as this is what camera's do.
camera_view = camera_gain + 100

# Convert to integer as a camera output can only take integers
# Conversion to: USER INT VALUE 16
camera_view = camera_view.astype(np.uint16)

### SAVE ###
sam.savetiff("Camera_image.tif", camera_view)

plt.imshow(camera_view)
plt.show()

