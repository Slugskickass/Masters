import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.fft as fft
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.ndimage import gaussian_filter
from scipy import signal


def PhaseKai2opt(k2fa, fS1aTnoisy, OTFo):
    w = np.shape(fS1aTnoisy)[0]  # The size of the image along one side
    wo = w / 2  # Not required

    fS1aTnoisy = fS1aTnoisy * (
                1 - 1 * OTFo ** 10)  # FFT of the original image is muliplied by 1-the OTF^10 (why is this? variable seems to suggest denoising?)
    fS1aT = fS1aTnoisy * np.conj(
        OTFo)  # The new image is multiplied byt the conj of the OTF to give the final image (what is the purpose of this?)
    # Is the above to not change the phase by multiplying the conjugate and the original by variable amounts?

    Kotf = OTFedgeF(OTFo)  # Find the high-frequency envelope of the OTF; THIS IS NOT USED, WHY IS IT HERE?

    # The below section to S1aT is basically making a pattern as is defined in the functions outside. With a few added extras.
    t = w  # Why redefine?
    to = t / 2  # ^^

    # Build a 2D linear meshgrid
    u = np.linspace(0, t - 1, t)
    v = np.linspace(0, t - 1, t)
    [U, V] = np.meshgrid(u, v)

    # What is j?  Can't find an equation like this in the paper and not sure what the variables are, so can't work out what this is doing
    # e ^ (-j* 2pi (k2fa[y]/width * (position_Y- halfwidth) + ktfa[x]/width * (position_X - halfwidth)) * inverse FFT of (FFT of the file multiplied by the conjugated base OTF)
    S1aT = np.exp(-1j * 2 * np.pi * (k2fa[1] / t * (U - to) + k2fa[0] / t * (V - to))) * fft.ifft2(fS1aT)

    fS1aT0 = fft.fft2(S1aT)

    mA = np.sum(fS1aT * np.conj(fS1aT0))

    mA = mA / np.sum(fS1aT0 * np.conj(fS1aT0))

    return (-1 * np.abs(mA))  # Specifically what does this mean in context and what are we aiming for it to be?
def OTFedgeF(OTFo):
    w = np.shape(OTFo)[0] #Shape of OTF in a single axis (returns side length), then find half of this
    wo = np.int(np.floor((w/2)))       # Should this be w//2 as it is needs to be an integer. Maybe a plus one if required too.

    # Makes a 1D array.
    OTF1 = OTFo[wo+1, :] # New matrix contains central row of otf (lateral cross-section through the theoretical diameter of the symmetrical otf)
    OTFmax =np.max(np.abs(OTFo))    # Peak value in OTF (presumably central, DC component?)
    OTFtruncate = 0.01   # Scaling factor
    i = 0   # Iteration counter
    while (np.abs(OTF1[i]) < OTFtruncate * OTFmax):# Needs edit to OTF1[i] to reflect 1D nature of array.
  #      print(i)
    #while the absolute value of given point in the central row of OTF is less than the scale factor multiplied by the max value, for each value of i in x:
    # what is the shape of OTF1? surely it is 1D? Yes, confirmed on testing.
        Kotf = wo+1-i #The half width of the FT +1 minus the iteration count: which point in frequency space axis (1/x) are we at?
        i = i + 1; #Continue to count (i+=1 also works)
    return Kotf # Return the position in x where the while statement is no longer true: this is where the intensity of the fourier pixel is greater than the
                    # max intensity (scaled by the scaling factor) and should return the "edge" of the main intensity pattern in the OTF. Why do we want this?
                # We want this as it reduces the space we need to look in to optimise our correlation as we know the start of the peaking.
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

def generate_patter(size, K_vector, ):
    t = size
    u = np.linspace(0, t-1, t)
    v = np.linspace(0, t-1, t)
    [U, V] = np.meshgrid(u, v)
    to = t/2
    i = np.complex(0, 1)
    S1aT = np.exp(-1*i * 2*np.pi*( K_vector[0]/t*(U-to)+K_vector[1]/t*(V-to)))
    return(S1aT)

def get_image(filename, frame):
    image = Image.open(filename)
    image.seek(frame)
    data = np.asarray(image)
    return(data)

def generate_PSF(NA, lamda, pixel_size, frame_size):
    # FWHM = 2.3548 * sigma
    sigma = (2.3548 * lamda / (2 * NA)) / pixel_size
    t = frame_size
    xo = np.floor(t/2)
    u = np.linspace(0, t-1, t)
    v = np.linspace(0, t-1, t)
    [U, V] = np.meshgrid(u, v)
    snuggle = np.exp(-1 * ((((xo - U)**2)/sigma**2) + (((xo - V)**2)/sigma**2)))
    return(snuggle)

def return_shiffetd_fft(Image):
    fft_im = fft.fft2(Image)
    fft_im_sh = fft.fftshift(fft_im)
    return(fft_im_sh)

def combine_image_OFT(image, oft):
    temp = return_shiffetd_fft(image)
    fft_blur = temp * oft
    final = fft.ifft2(fft.ifftshift(fft_blur))
    return(final)

filename = '/Users/Ashley/PycharmProjects/SIMple/Data/SLM-SIM_Tetraspeck200_680nm.tif'

image_data = get_image(filename, 0)

# Generate a PSF
psf = generate_PSF(1.2, 680, 97, 512)

# Generate an OFT
OTFo = return_shiffetd_fft(psf)

# Fourier of data
f_image_data = return_shiffetd_fft(image_data)


# Generate pattern
# size_x = 4000
# values = np.linspace(0, 200, size_x)
# summed = np.zeros((size_x))
# for indk, K in enumerate(values):
#     K_vector = [0.01, K]
#     S1aT = generate_patter(512, K_vector)
#     output = S1aT * fft.ifftshift(fft.ifft2(f_con_image))
#     output = fft.fft2(output)
#     summed[indk] = np.sum(np.conj(output) * fft.fft2(f_con_image))
# plt.plot(values, -1*summed)
# plt.show()

size_x = 400
values = np.linspace(110, 120, size_x)
out = np.zeros(size_x)
for indk, K in enumerate(values):
    k2fa = [0.01, K]
    out[indk] = PhaseKai2opt(k2fa, f_image_data, OTFo)
plt.plot(values, -1*out)
plt.show()









