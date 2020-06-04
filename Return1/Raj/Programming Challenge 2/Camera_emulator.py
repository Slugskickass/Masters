import numpy as np
import matplotlib.pyplot as plt
import Camera_Samurai as sam
from scipy import signal

### INPUT PARAMETERS ###
# GROUND TRUTH
image_type = "Interesting"           # Ground truth image if you want user input. Choose: "Boring", "Interesting"
groundpixel = 5                 # Pixel size e.g. pixels = 5nm
photon_count = 10               # Photon count value per unit time (same unit used in exposure)
exposure_time = 1               # seconds of exposure
# PSF
NA = 1.4                        # Numerical aperture
wavelength = 440                # Wavelength in nanometres
# CAMERA
camera_pixel_size = 6500        # Camera pixel size in nanometres. usual sizes = 6 microns or 11 microns
magnification = 100             # Lens magnification
QE = 0.95                       # Quantum Efficiency
gain = 2                        # Camera gain. Usually 2 per incidence photon
# NOISE
read_mean = 2                   # Read noise mean level
read_std = 2                    # Read noise standard deviation level
fixed_pattern_deviation = 0.001 # Fixed pattern standard deviation. usually affects 0.1% of pixels.
# SAVE
SAVE = "Y"                      # Save parameter, input Y to save, other parameters will not save.

### DATA CHECKS ###
if NA <= 0 or NA >= 2:
    print("Numerical aperture value:", NA)
    raise ValueError("This NA is outside the normal range.")

if wavelength <=250 or wavelength >=700:
    print("Wavelength value:", wavelength)
    raise ValueError("This wavelength is outside the normal range.")

if QE <=0 or QE >1:
    print("Quantum efficiency value:", QE)
    raise ValueError("This QE is outside the normal range.")


### SAMPLE GENERATION ###
# Make a Ground Truth
# Based on pixel size. e.g. 5nm pixel, therefore: 10 microns = 2kx2k array
ground, ground_window = sam.image_selector(image_type, groundpixel, photon_count, exposure_time)
#### END AWESOME SAMPLE MAKER ####


### LENS SIMULATOR ###
# Lens == A diffraction limited blur. Dependent on wavelength and NA
psf = sam.psf_generator(NA, wavelength, groundpixel, ground_window)

# Apply the lens as a convolution of the two arrays producing a diffraction limited image.
dif_lim = sig.fftconvolve(ground, psf, "same")            # Fiddling with fourier space to convolute, much faster.


### CAMERA SETUP ###
# Camera sensor, based on optical magnification and pixel size.
camerapixel_per_groundpixel = camera_pixel_size/groundpixel

# Used to determine the number of the ground pixels that exist within each bin
mag_ratio = camerapixel_per_groundpixel / magnification
print("Overall Image Binning (ground pixels per bin):", mag_ratio, "by", mag_ratio)


### IMAGING TIME ###
# Initialise an empty array, with a size calculated by the above ratios.
# Gives us a rounded down number of pixels to bin into to prevent binning half a bin volume into a pixel.
camera_image = np.zeros((int(dif_lim.shape[0]//mag_ratio), int(dif_lim.shape[1]//mag_ratio)))

# Iterate each position in the array and average the pixels in the range from the diffraction limited image.
# We use the mag_ratio to step across the array and select out regions that are multiples of it out.
for y in range(0, camera_image.shape[0]):
    for x in range(0, camera_image.shape[1]):
        pixel_section = dif_lim[y*int(mag_ratio):y*int(mag_ratio)+int(mag_ratio),
                                x*int(mag_ratio):x*int(mag_ratio)+int(mag_ratio)]
        camera_image[y, x] = np.mean(pixel_section)     # Take the mean value of the section and bin it to the camera.

# Account for Quantum efficiency.
camera_image = camera_image * QE


### ADD NOISE ###
# Add read and shot noise.
camera_Rnoise = sam.read_noise(camera_image, read_mean, read_std)
camera_Snoise = sam.shot_noise(np.sqrt(photon_count * exposure_time), camera_image)

# Add up the camera, read and shot noises.
camera_RSnoise = camera_image + camera_Rnoise + camera_Snoise

# FP noise remains the same for the camera, hence we fix the seed.
# We can simulate one by producing a normal distribution around 1 with a deviation relative to the number of pixels that
# would commonly deviate under such parameters.
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
if SAVE == "Y":
    sam.savetiff("Camera_image.tif", camera_view)
 as sig
