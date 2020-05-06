import numpy as np
import scipy.fft as fft
from PIL import Image
import matplotlib.pyplot as plt


def get_image(filename, frame):
    image = Image.open(filename)
    image.seek(frame)
    data = np.asarray(image)
    return data


def generate_patter(size, K_vector, ):
    t = size
    u = np.linspace(0, t-1, t)
    v = np.linspace(0, t-1, t)
    [U, V] = np.meshgrid(u, v)
    to = t/2
    i = np.complex(0, 1)
    S1aT = np.exp(-1*i * 2*np.pi*( K_vector[0]/t*(U-to)+K_vector[1]/t*(V-to)))
    return S1aT


def generate_PSF(NA, lamda, pixel_size, frame_size):
    # FWHM = 2.3548 * sigma
    sigma = (2.3548 * lamda / (2 * NA)) / pixel_size
    t = frame_size
    xo = np.floor(t/2)
    u = np.linspace(0, t-1, t)
    v = np.linspace(0, t-1, t)
    [U, V] = np.meshgrid(u, v)
    snuggle = np.exp(-1 * ((((xo - U)**2)/sigma**2) + (((xo - V)**2)/sigma**2)))
    return snuggle


def return_shiffetd_fft(Image):
    fft_im = fft.fft2(Image)
    fft_im_sh = fft.fftshift(fft_im)
    return fft_im_sh


# Done ?
def OTFedgeF(OTFo):
    w = np.shape(OTFo)[0]
    wo = w//2

    OTF1 = OTFo[wo+1, :]
    OTFmax =np.max(np.abs(OTFo))
    OTFtruncate = 0.01
    i = 0

    while np.abs(OTF1[i]) < OTFtruncate * OTFmax:
        Kotf = wo-i
        i = i + 1;
    return Kotf

# Done ?
def PhaseKai2opt(k2fa, fS1aTnoisy, OTFo):
#% The size of the image
    w = np.shape(fS1aTnoisy)[0]
    # wo = w/2

# Here the FFT of the original image is muliplied by the
# 1 -the OTF^10 (?)
# The new image is then multiplied byt the conj of the OTF to give the final image
    fS1aTnoisy = fS1aTnoisy*(1-1*OTFo**10)
    fS1aT = fS1aTnoisy*np.conj(OTFo)

    # Kotf = OTFedgeF(OTFo)

    t = w

    to = t/2
    u = np.linspace(0, t-1, t)
    v = np.linspace(0, t-1, t)
    [U, V] = np.meshgrid(u, v)

    S1aT = np.exp( -1j * 2 * np.pi * (k2fa[1]/t * (U-to)+k2fa[0]/t * (V-to))) * fft.ifft2(fS1aT)

    fS1aT0 = fft.fft2(S1aT)

    mA = np.sum(fS1aT * np.conj(fS1aT0))

    mA = mA / np.sum(fS1aT0 * np.conj(fS1aT0))

    return mA


if __name__ == '__main__':
    filename = '/Users/RajSeehra/University/Masters/Semester 2/Git Upload Folder/SIM_Week/Data/SLM-SIM_Tetraspeck200_680nm.tif'

    # Import data
    original_data = get_image(filename, 0)
    # FFT the data
    fS1aTnoisy = return_shiffetd_fft(original_data)

    # Makes PSF and then fft of PSF
    psf = generate_PSF(1.2, 680, 97, 512)
    OTFo = return_shiffetd_fft(psf)

    # number of peaks in x and y respectively
    k2fa = [0.01, 114]
    # Computes a optimisation variable for the phase. closer to 0 better.
    autocorrelation = PhaseKai2opt(k2fa, fS1aTnoisy, OTFo)
    # Generates a patter.
    patter = generate_patter(512, k2fa)

print(autocorrelation)
# plt.imshow(np.real(patter))
# plt.show()
