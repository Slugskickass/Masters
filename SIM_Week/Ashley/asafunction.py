import numpy as np
import scipy.fft as fft
from PIL import Image
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def generate_patter(size, K_vector, ):
    t = size
    u = np.linspace(0, t-1, t)
    v = np.linspace(0, t-1, t)
    [U, V] = np.meshgrid(u, v)
    to = t/2
    i = np.complex(0, 1)
    S1aT = np.exp(-1*i * 2*np.pi*( K_vector[0]/t*(U-to)+K_vector[1]/t*(V-to)))
    return(S1aT)

## Image loaded into programme and appended, frame-wise to a 3D numpy array
def get_image(filename, frame):
    image = Image.open(filename)
    image.seek(frame)
    data = np.asarray(image)
    return data

## Mathematical approximation of the PSF of the system
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

## Shortcut function to combine operations of FFT2 and shift to realign in a meaningful manner
def return_shiffetd_fft(Image):
    fft_im = fft.fft2(Image)
    fft_im_sh = fft.fftshift(fft_im)
    return fft_im_sh

# Done ?
    #Input: OTF of system (o denoting microscope system base).
    #Return: point in fourier soace which marks the edge of the OTF
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

# Done ?
#Find input/function names confusing, not easy to know what it is expecting/outputting!
# The name refers to a 2 opt system for SIM developed by Kai et al. Paper link: https://www.osapublishing.org/DirectPDFAccess/C3B732C2-A58C-DD03-9E565EDD53E3B0C2_248609/oe-21-2-2032.pdf?da=1&id=248609&seq=0&mobile=no


def PhaseKai2opt(k2fa, fS1aTnoisy, OTFo):
    w = np.shape(fS1aTnoisy)[0] # The size of the image along one side
    wo = w/2 # Not required

    fS1aTnoisy = fS1aTnoisy*(1-1*OTFo**10) # FFT of the original image is muliplied by 1-the OTF^10 (why is this? variable seems to suggest denoising?)
    fS1aT = fS1aTnoisy*np.conj(OTFo) # The new image is multiplied byt the conj of the OTF to give the final image (what is the purpose of this?)
    # Is the above to not change the phase by multiplying the conjugate and the original by variable amounts?

    Kotf = OTFedgeF(OTFo) # Find the high-frequency envelope of the OTF; THIS IS NOT USED, WHY IS IT HERE?

    # The below section to S1aT is basically making a pattern as is defined in the functions outside. With a few added extras.
    t = w #Why redefine?
    to = t/2 # ^^
    
    #Build a 2D linear meshgrid
    u = np.linspace(0, t-1, t)
    v = np.linspace(0, t-1, t)
    [U, V] = np.meshgrid(u, v)

    # What is j?  Can't find an equation like this in the paper and not sure what the variables are, so can't work out what this is doing
    # e ^ (-j* 2pi (k2fa[y]/width * (position_Y- halfwidth) + ktfa[x]/width * (position_X - halfwidth)) * inverse FFT of (FFT of the file multiplied by the conjugated base OTF)
    S1aT = np.exp( -1j * 2 * np.pi * ( k2fa[1]/t * (U-to)+k2fa[0]/t * (V-to))) * fft.ifft2(fS1aT)

    fS1aT0 = fft.fft2(S1aT)

    mA = np.sum(fS1aT * np.conj(fS1aT0))

    mA = mA / np.sum(fS1aT0 * np.conj(fS1aT0))

    return (-1 * np.abs(mA))       # Specifically what does this mean in context and what are we aiming for it to be?

def newPhaseKai2opt(k2fa, fS1aTnoisy):
    S1aT = generate_patter(512, k2fa)
    output = S1aT * fft.ifftshift(fft.ifft2(fS1aTnoisy))
    output = fft.fft2(output)
    summed = np.sum(np.conj(output) * fft.fft2(fS1aTnoisy))
    return(-1*summed)





if __name__ == '__main__':
    filename = '/Users/Ashley/PycharmProjects/SIMple/Data/SLM-SIM_Tetraspeck200_680nm.tif'

    # Import data
    original_data = get_image(filename, 0)
    # FFT the data
    fS1aTnoisy = return_shiffetd_fft(original_data)

    # Makes PSF and then fft of PSF
    psf = generate_PSF(1.2, 680, 97, 512)
    OTFo = return_shiffetd_fft(psf)

    # number of peaks in x and y respectively
    # points =100
    # autocorrelation = np.zeros(points)
    # hold_pla = np.zeros(points)
    # places = np.linspace(110, 120, points)
    # for ind, X in enumerate(places):
    #     k2fa = [X, 0.01]
    #     hold_pla[ind] = X
    # # Computes a optimisation variable for the phase. ?closer to 0/1 better?
    #     autocorrelation[ind] = PhaseKai2opt(k2fa, fS1aTnoisy, OTFo)
    # plt.plot(hold_pla, autocorrelation)
    # plt.show()

    k2fa = [0, 110]

    res = minimize(PhaseKai2opt, x0=k2fa, args=(fS1aTnoisy, OTFo), method='Nelder-Mead', tol=0.00001)
#    res = minimize(newPhaseKai2opt, x0=k2fa, args=(fS1aTnoisy), method='Nelder-Mead', tol=0.000001)

    print(res.x)
    print('The magnitude is', np.sqrt(res.x[0]**2 + res.x[1]**2))
    print('The angle is', np.arctan2(res.x[1], res.x[0]))

    reconstructed = generate_patter(512, [res.x[0], res.x[1]])
    reconstructed_f = fft.ifftshift(fft.fft2(reconstructed))
    position_max = np.argmax(np.abs(reconstructed_f))
    X, Y = np.unravel_index(position_max, [512, 512])
    plt.imshow(np.log(np.abs(fS1aTnoisy**2)))
    plt.plot(Y, X, 'o', fillstyle='none')
    plt.show()

    plt.plot(np.log(np.abs(fS1aTnoisy[:, 258]))-10)
    plt.plot(np.log(np.abs(reconstructed_f[:, 258])))
    plt.show()

