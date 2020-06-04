from PIL import Image
import numpy as np
import os
import scipy


# PSF generator: calculates the 2D PSF of a system.
    # Takes the: numerical aperture of the system, the emission wavelength, the ground truth pixel size,
    #           the return size for the PSF and a correction value if required.
    # This is then adjusted and normalised to ensure the sum of the PSF is = 1 for convolution purposes.
    # This instance of the function uses the values for FWHM to determine the sigma size.
def psf_generator(NA, wavelength, pixel_size=100, frame_size=100, correction=1):
    # FWHM = np.sqrt(8*np.log(2)) * sigma
    sigma = (np.sqrt(8*np.log(2)) * wavelength / (2 * NA * correction)) / pixel_size
    t = frame_size
    xo = np.floor(t / 2)
    u = np.linspace(0, t - 1, t)
    v = np.linspace(0, t - 1, t)
    [U, V] = np.meshgrid(u, v)
    psf = np.exp(-1 * ((((xo - U) ** 2) / sigma ** 2) + (((xo - V) ** 2) / sigma ** 2)))

    normalised_psf = psf / np.sum(psf)

    return normalised_psf


# single frame save function.
def savetiff(file_name, data):
    images = Image.fromarray(data[:, :])
    images.save(file_name)

# Takes the data inputs required for a ground truth and returns either the boring or interesting image as per user
# request.
def image_selector(type, groundpixel, photon_count, exposure_time):
    ground_window = 10000 // groundpixel

    if type == "Boring":
        ground = np.zeros((ground_window, ground_window))
        ground[1000:2000, 950:1050] = photon_count * exposure_time
        return ground, ground_window

    elif type == "Interesting":
        # This is a fixed 2k x 2k image so the above values of ground_window hold no value only in that they match the
        # image currently.
        ground = np.load("ambiguous_image.npy")
        ground = ground * photon_count * exposure_time

        return ground, ground_window


def read_noise(y_data, read_mean=2, read_std=2):
    # Build read noise. Always positive so we take the absolute values.
    read_noise = np.random.normal(read_mean, read_std / np.sqrt(2), np.shape(y_data))
    read_noise= np.abs(read_noise)
    return read_noise


def shot_noise(sqrt_mean_signal, y_data):
    shot_noise = np.random.poisson(sqrt_mean_signal, (y_data.shape[0], y_data.shape[1]))
    return shot_noise