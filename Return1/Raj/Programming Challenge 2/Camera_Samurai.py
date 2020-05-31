from PIL import Image
import numpy as np
import os
import scipy


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


# single frame
def savetiff(file_name, data):
    images = Image.fromarray(data[:, :])
    images.save(file_name)


# single frame
def image_open(file):
    # open the file
    file_name = file
    img = Image.open(file_name)
    # generate the array and apply the image data to it.
    imgArray = np.zeros((img.size[1], img.size[0], img.n_frames), np.uint16)
    imgArray[:, :, 0] = img
    img.close()


# multi stack tiffs
def loadtiffs(file_name):
    img = Image.open(file_name)
    #print('The Image is', img.size, 'Pixels.')
    #print('With', img.n_frames, 'frames.')

    imgArray = np.zeros((img.size[1], img.size[0], img.n_frames), np.float32)
    for I in range(img.n_frames):
        img.seek(I)
        imgArray[:, :, I] = np.asarray(img)
    img.close()
    return (imgArray)


# as above
def savetiffs(file_name, data):
    images = []
    for I in range(np.shape(data)[2]):
        images.append(Image.fromarray(data[:, :, I]))
        images[0].save(file_name, save_all=True, append_images=images[1:])


def read_noise(y_data, read_mean, read_std):
    # Build read noise.
    read_noise = np.random.normal(read_mean, read_std / np.sqrt(2), np.shape(y_data))
    return read_noise


def shot_noise(sqrt_mean_signal, y_data):
    shot_noise = np.random.poisson(sqrt_mean_signal, (y_data.shape[0], y_data.shape[1]))
    return shot_noise