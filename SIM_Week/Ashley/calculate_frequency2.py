import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.fft as fft
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.ndimage import gaussian_filter
from scipy import signal
import scipy.special
from scipy.optimize import minimize


def build_psf_oft(size, scale):
    x = np.linspace(0, w - 1, w)
    y = np.linspace(0, w - 1, w)
    x = x.astype(int)
    y = y.astype(int)
    [X, Y] = np.meshgrid(x, y)

    # Generate the PSF
    R = np.sqrt(np.minimum(X, np.abs(X - w)) ** 2 + np.minimum(Y, np.abs(Y - w)) ** 2)
    yy = np.abs(2 * scipy.special.jv(1, scale * R + np.finfo(np.float32).resolution) / (scale * R + np.finfo(np.float32).resolution)) ** 2
    yy0 = fft.fftshift(yy)

    # Generate OTF
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

def OTFedgeF(OTFo):
    w = np.shape(OTFo)[0]
    wo = np.int(w / 2)
    OTF1 = OTFo[wo+1, :]
    OTFmax = np.max(np.abs(OTFo))
    OTFtruncate = 0.01
    i = 1
    while (np.abs(OTF1[i]) < OTFtruncate * OTFmax):
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

    w = np.shape(FiSMap)[0]
    wo = w/2
    x = np.linspace(0,w-1,w)
    y = np.linspace(0,w-1,w)
    [X, Y] = np.meshgrid(x,y)

    Ro = np.sqrt( (X-wo)**2 + (Y-wo)**2 )
    Z0 = Ro > np.round(0.5 * Kotf)
    Z1 = X > wo

    FiSMap = FiSMap * Z0 * Z1


#    dumY = np.max( FiSMap,[],1 )
    dumY = np.max(FiSMap, axis=0)

#    [dummy Iy] = max(dumY);
    Iy = np.argmax(dumY)
#    dumX = max( FiSMap,[],2 );
    dumX = np.max(FiSMap, axis=1)
#    [dummy Ix] = max(dumX);
    Ix = np.argmax(dumX)

    plt.plot(dumY)
    plt.show()

    plt.plot(dumX)
    plt.show()

    plt.imshow(FiSMap)
    plt.plot(Iy, Ix, 'yo')
    plt.show()



    maxK2 = np.zeros(2)
    maxK2[0] = np.abs(Ix-(wo+1)+1) ## remove abs?
    maxK2[1] = np.abs(Iy-(wo+1)+1)
    return(maxK2, Ix, Iy, FiSMap)

def PhaseKai2opt(k2fa,fS1aTnoisy, OTFo):
    w = np.shape(fS1aTnoisy)[0]
    wo = np.int(w / 2)

    fS1aTnoisy = fS1aTnoisy * (1 - 1 * OTFo**10)
    fS1aT = fS1aTnoisy * np.conj(OTFo)
    Kotf = OTFedgeF(OTFo)
    DoubleMatSize = 0

    if (2 * Kotf > wo):
        DoubleMatSize = 1

    if (DoubleMatSize > 0):
        t = 2 * w

        fS1aT_temp = np.zeros((t, t))
        fS1aT_temp[wo : w + wo, wo : w + wo] = fS1aT
        fS1aT = fS1aT_temp
    else:
        t = w

    to = np.int(t / 2)
    u = np.linspace(0, t - 1, t)
    v = np.linspace(0, t - 1, t)
    [U, V] = np.meshgrid(u, v)


    S1aT = np.exp(-1j * 2 * np.pi * (k2fa[1] / t * (U - to)) + (k2fa[0] / t * (V - to))) * fft.ifft2(fS1aT)
    fS1aT0 = fft.fft2(S1aT)

    mA = np.longdouble(np.sum(fS1aT * np.conj(fS1aT0))) # This is a correlation
    mA = mA / (np.longdouble(np.sum(fS1aT0 * np.conj(fS1aT0))))
    print(type(mA))
    #print(-np.abs(mA))
    CCop = -abs(mA)

    return(CCop)



scale = 0.99
#filename = '/Users/Ashley/PycharmProjects/SIMple/Data/SLM-SIM_Tetraspeck200_680nm.tif'
#filename = '/Users/Ashley/PycharmProjects/SIMple/Data/Zeiss_Actin_525nm_large.tif'
filename = '/Users/Ashley/PycharmProjects/SIMple/Data/Zeiss_Actin_525nm_crop.tif'
#filename = '/Users/Ashley/PycharmProjects/SIMple/Data/out.tiff'
image_data = get_image(filename, 0)
w = np.shape(image_data)[0]

PSFo, OTFo = build_psf_oft(w, scale)
wo = np.int(w/2)


PSFd = np.real(fft.fftshift(fft.ifft2(fft.fftshift(OTFo**10))))
PSFd = PSFd/np.max(PSFd)
PSFd = PSFd/np.sum(PSFd)
h = 30
PSFe = PSFd[wo-h+1:wo+h, wo-h+1:wo+h]
# Section of PSF


# edge tapering raw SIM image
S1aTnoisy_et = image_data
# S1aTnoisy_et = edgetaper(image_data, PSFe) # Dont know this function yet
fS1aTnoisy_et = fft.fftshift(fft.fft2(S1aTnoisy_et))
# fS1aTnoisy_et is the fft of the image


Kotf = OTFedgeF(OTFo)
# Max


# Approx illumination frequency vector
[k2fa, ta, tb, test] = ApproxFreqDuplex(fS1aTnoisy_et, Kotf)
fS1aTnoisy = fft.fftshift(fft.fft2(image_data))
print(k2fa)
print(np.sqrt(k2fa[0]**2 + k2fa[1]**2))
print(np.arctan2(-1*k2fa[0], k2fa[1]))

number = 200
start = 290
stop = 310
CCop = np.zeros(number)
points = np.linspace(start, stop, number)
for index, I in enumerate(points):
    print(I)
    k2fa[0] = I
    CCop[index] = PhaseKai2opt(k2fa, fS1aTnoisy, OTFo)
plt.plot(CCop)
plt.show()

res = minimize(PhaseKai2opt, x0=k2fa, args=(fS1aTnoisy, OTFo), method='Nelder-Mead', tol=0.00001)
print(res.x)
