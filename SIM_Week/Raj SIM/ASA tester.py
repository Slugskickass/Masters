import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.fft as fft
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.ndimage import gaussian_filter
from scipy import signal
import scipy.special
from scipy.optimize import minimize, basinhopping
from matplotlib.colors import LogNorm


def build_psf_oft(size, scale):
    x0 = size[1]
    y0 = size[0]
    x = np.linspace(0, x0 - 1, x0)
    y = np.linspace(0, y0 - 1, y0)
    x = x.astype(int)
    y = y.astype(int)
    [X, Y] = np.meshgrid(x, y)

    # Generate the PSF. Apply a bezel over a gaussian.
    R = np.sqrt(np.minimum(X, np.abs(X - x0)) ** 2 + np.minimum(Y, np.abs(Y - y0)) ** 2)
    yy = np.abs(2 * scipy.special.jv(1, scale * R + np.finfo(np.float32).resolution) / (scale * R + np.finfo(np.float32).resolution)) ** 2
    yy0 = fft.fftshift(yy)      # Mainly for viewing purposes.

    # Generate OTF/ dividing by the maximum value to normalise the OTF to a maximum of 1.
    OTF2d = fft.fft2(yy)
    OTF2dmax = np.max(np.abs(OTF2d))
    OTF2d = OTF2d / OTF2dmax
    OTF2dc = np.abs(fft.fftshift(OTF2d))

    return (yy0, OTF2dc)


def get_image(filename, frame):
    image = Image.open(filename)
    image.seek(frame)
    data = np.asarray(image)
    return(data)


def return_shifted_fft(Image):
    fft_im = fft.fft2(Image)
    fft_im_sh = fft.fftshift(fft_im)
    return fft_im_sh


def OTFedgeF(OTFo):     # Generates a guess for the edge of the OTF.
    w = np.shape(OTFo)[0]
    wo = np.int(w / 2)
    OTF1 = OTFo[wo+1, :]
    OTFmax = np.max(np.abs(OTFo))
    OTFtruncate = 0.01
    i = 1
    while (np.abs(OTF1[i]) < OTFtruncate * OTFmax):     # Runs till the value in a cell is greater than max* trunc.
        Kotf = wo + 1 - i
        i = i + 1
    return Kotf

def ApproxFreqDuplex(FiSMap,Kotf):
    #% AIM: approx. illumination frequency vector determination
    #% INPUT VARIABLES
    #%   FiSMap: FT of raw SIM image
    #%   Kotf: OTF cut-off frequency
    #% OUTPUT VARIABLES
    #%   maxK2: illumination frequency vector (approx)
    #%   Ix,Iy: coordinates of illumination frequency peaks

    FiSMap = np.abs(FiSMap)

    # CIRCLE PARAMETERS
    w = np.shape(FiSMap)[0]
    wo = w/2
    x = np.linspace(0,w-1,w)
    y = np.linspace(0,w-1,w)
    [X, Y] = np.meshgrid(x,y)
    # CIRCLE MAKER
    Ro = np.sqrt( (X-wo)**2 + (Y-wo)**2 )
    Z0 = Ro > np.round(0.5 * Kotf)      # Generates circle
    Z1 = X > wo                         # Generates half map.

    FiSMap = FiSMap * Z0 * Z1           # Masks the original FFT with the circle and half map.

    # Maximum Locations.
    dumY = np.max(FiSMap, axis=0)       # Max value of raw FFT of original image, in Y
    Iy = np.argmax(dumY)                # Returns the location in Y
    dumX = np.max(FiSMap, axis=1)       # Max value of raw FFT of original image, in Y
    Ix = np.argmax(dumX)                # Returns the location in X

    ### PLOTS ###
    # plt.plot(dumY)
    # plt.show()
    #
    # plt.plot(dumX)
    # plt.show()
    #
    # plt.imshow(FiSMap)
    # plt.plot(Iy, Ix, 'yo')
    # plt.show()

    # Coordinates relative to the centre.
    maxK2 = np.zeros(2)
    maxK2[0] = Ix-(wo+1)+1
    maxK2[1] = Iy-(wo+1)+1
    return(maxK2, Ix, Iy, FiSMap)


def PhaseKai2opt(k2fa, fS1aTnoisy, OTFo):
    w = np.shape(fS1aTnoisy)[0]
    wo = np.int(w / 2)

    # Clean up for noise reduction.
    fS1aTnoisy = fS1aTnoisy * (1 - 1 * OTFo**10)
    fS1aT = fS1aTnoisy * np.conj(OTFo)
    Kotf = OTFedgeF(OTFo)

    # If the FFT is too large relative to the size of the array then this will rehouse it to a doubled array.
    if (2 * Kotf > wo):
        t = 2 * w

        fS1aT_temp = np.zeros((t, t),dtype=complex)
        fS1aT_temp[wo : w + wo, wo : w + wo] = fS1aT
        fS1aT = fS1aT_temp
    else:
        t = w

    to = np.int(t / 2)
    u = np.linspace(0, t - 1, t)
    v = np.linspace(0, t - 1, t)
    [U, V] = np.meshgrid(u, v)

    # Generates the line map in real space and multiplies it by the edited real image.
    # This thus amplifies the waveforms present. A better match = better resonance.
    S1aT = np.exp(-1j * 2 * np.pi * (k2fa[1] / t * (U - to)) + (k2fa[0] / t * (V - to))) * fft.ifft2(fS1aT)
    fS1aT0 = fft.fft2(S1aT)

    # Correlation values.
    mA = np.sum(fS1aT * np.conj(fS1aT0))    # This is the correlation value
    mA = mA / (np.sum(fS1aT0 * np.conj(fS1aT0)))     # Normalise.
    CCop = -abs(mA)     # Correlation value normalised and negativised so we can easily minimise.

    return(CCop)


def debug_k_vector(axis, k2fa, range=20, iterations=200):
    if axis == 'x':
        axis = 0
    elif axis == 'y':
        axis = 1

    number = iterations  # Number of Iterations
    start = k2fa[axis] - range/2  # Start range value
    stop = k2fa[axis] + range/2  # Stop range value
    CCop = np.zeros(number)
    points = np.linspace(start, stop, number)  # List of numbers to test against.

    for index, I in enumerate(points):
        print(I)
        k2fa[axis] = I
        CCop[index] = PhaseKai2opt(k2fa, fS1aTnoisy, OTFo)

    plt.plot(points, CCop)
    plt.show()

    res = minimize(PhaseKai2opt, x0=k2fa, args=(fS1aTnoisy, OTFo), method='Nelder-Mead', tol=0.00001)
    return res


def edgetaper(image_data, PSF):

    return


### IMPORT FILE ###
#filename = '/Users/Ashley/PycharmProjects/SIMple/Data/SLM-SIM_Tetraspeck200_680nm.tif'
#filename = '/Users/Ashley/PycharmProjects/SIMple/Data/Zeiss_Actin_525nm_large.tif'
filename = '/Users/RajSeehra/University/Masters/Semester 2/Git Upload Folder/SIM_Week/Data/SLM-SIM_Tetraspeck200_680nm.tif'
# filename = '/Users/RajSeehra/University/Masters/Semester 2/Git Upload Folder/SIM_Week/Data/out.tiff'
#filename = '/Users/Ashley/PycharmProjects/SIMple/Data/out.tiff'
image_data = get_image(filename, 0)

### PARAMETERS ###
w = np.shape(image_data)
wo = np.int(w[0]/2)
scale = 0.99

### PSF/OTF GENERATOR ###
PSFo, OTFo = build_psf_oft(w, scale)

##
##
## WORK IN PROGRESS
### EDGE TAPERING RAW SIM IMAGE (WIP) ###
S1aTnoisy_et = image_data
# S1aTnoisy_et = edgetaper(image_data, PSFe) # Dont know this function yet. NEED TO MAKE.
# vignette in real space.
fS1aTnoisy_et = return_shifted_fft(S1aTnoisy_et)
##
##

### ESTIMATION OF LOCAL MINIMA DISTANCE ###
Kotf = OTFedgeF(OTFo)


### APPROX ILLUMINATION FREQUENCY VECTOR ###
[k2fa, ta, tb, test] = ApproxFreqDuplex(fS1aTnoisy_et, Kotf)
fS1aTnoisy = return_shifted_fft(image_data)
print(k2fa)
print('The magnitude is approximately', np.sqrt(k2fa[0]**2 + k2fa[1]**2))
print('The angle is approximately', np.arctan2(-1*k2fa[0], k2fa[1]))


### ITERATOR IN THE REGION ###
# Function test
res = minimize(PhaseKai2opt, x0=k2fa, args=(fS1aTnoisy, OTFo), method='Nelder-Mead', tol=0.00001)
print(res.x)
print('The magnitude is', np.sqrt(res.x[0]**2 + res.x[1]**2))
print('The angle is', np.arctan2(-1*res.x[0], res.x[1]))


### TESTING WITH BASINHOPPING ###
# class MyBounds(object):
#     def __init__(self, xmax=[k2fa[0]+10,k2fa[1]+10], xmin=[k2fa[0]-10,k2fa[1]-10] ):
#         self.xmax = np.array(xmax)
#         self.xmin = np.array(xmin)
#     def __call__(self, **kwargs):
#         x = kwargs["x_new"]
#         tmax = bool(np.all(x <= self.xmax))
#         tmin = bool(np.all(x >= self.xmin))
#         return tmax and tmin
# mybounds= MyBounds()
# res = basinhopping(PhaseKai2opt, x0=k2fa, niter=200, minimizer_kwargs={"args":((fS1aTnoisy), (OTFo))}, stepsize=1, accept_test=mybounds, disp=True)
# print(res.x)
# print('The magnitude is', np.sqrt(res.x[0]**2 + res.x[1]**2))
# print('The angle is', np.arctan2(-1*res.x[0], res.x[1]))


### NOT OPTIMISER BUT DEBUGGING STUFF. ###
# Optimiser. AXIS, X='x', Y='y'.
debug_x = debug_k_vector('x', k2fa, 20, 200)
debug_y = debug_k_vector('y', k2fa, 20, 200)
print(debug_x.x)
print(debug_y.x)

print('The magnitude in x is ', np.sqrt(debug_x.x[0]**2 + debug_x.x[1]**2))
print('The angle in x is ', np.arctan2(debug_x.x[1], debug_x.x[0]))

print('The magnitude in y is', np.sqrt(debug_y.x[0]**2 + debug_y.x[1]**2))
print('The angle in y is', np.arctan2(debug_y.x[1], debug_y.x[0]))