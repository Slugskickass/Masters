import numpy as np
import scipy.fft as fft
from PIL import Image

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

### UNNECESSARY REPEAT?
#def generate_PSF(NA, lamda, pixel_size, frame_size):
#    # FWHM = 2.3548 * sigma
#    sigma = (2.3548 * lamda / (2 * NA)) / pixel_size
#    t = frame_size
#    xo = np.floor(t/2)
#    u = np.linspace(0, t-1, t)
#    v = np.linspace(0, t-1, t)
#    [U, V] = np.meshgrid(u, v)
#    snuggle = np.exp(-1 * ((((xo - U)**2)/sigma**2) + (((xo - V)**2)/sigma**2)))
#    return snuggle


# Done ?
    #Input: OTF of system (o denoting original/0?)
    #Return: point in fourier soace which marks the edge of the OTF
def OTFedgeF(OTFo):
    w = np.shape(OTFo)[0] #Shape of OTF in a single axis (returns side length), then find half of this
    wo = w/2

    OTF1 = OTFo[wo+1, :] #New matrix contains central row of otf (lateral cross-section through the theoretical diameter of the symmetrical otf)
    OTFmax =np.max(np.abs(OTFo)) #Peak value in OTF (presumably central, DC component?)
    OTFtruncate = 0.01 #Scaling factor
    i = 1 #Iteration counter
    while (np.abs(OTF1[1,i]) < OTFtruncate * OTFmax):
    #while the absolute value of given point in the central row of OTF is less than the scale factor multiplied by the max value, for each value of i in x:
    # what is the shape of OTF1? surely it is 1D?
        Kotf = wo+1-i #The half width of the FT +1 minus the iteration count: which point in frequency space axis (1/x) are we at?
        i = i + 1; #Continue to count (i+=1 also works)
    return Kotf #Return the position in x where the while statement is no longer true: this is where the intensity of the fourier pixel is greater than the 
                    #max intensity (scaled by the scaling factor) and should return the "edge" of the main intensity pattern in the OTF. Why do we want this?

# Done ?
#Find input/function names confusing, not easy to know what it is expecting/outputting!
def PhaseKai2opt(k2fa, fS1aTnoisy, OTFo):
    w = np.shape(fS1aTnoisy)[0] # The size of the image along one side
    wo = w/2 # Not required

    fS1aTnoisy = fS1aTnoisy*(1-1*OTFo**10) # FFT of the original image is muliplied by 1-the OTF^10 (why is this? variable seems to suggest denoising?)
    fS1aT = fS1aTnoisy*np.conj(OTFo) # The new image is multiplied byt the conj of the OTF to give the final image (what is the purpose of this?)

    Kotf = OTFedgeF(OTFo) # Find the high-frequency envelope of the OTF; THIS IS NOT USED, WHY IS IT HERE?

    t = w #Why redefine?
    to = t/2 # ^^
    
    #Build a 2D linear meshgrid
    u = np.linspace(0, t-1, t)
    v = np.linspace(0, t-1, t)
    [U, V] = np.meshgrid(u, v)

    # What is j?  Can't find an equation like this in the paper and not sure what the variables are, so can't work out what this is doing
    S1aT = np.exp( -1j * 2 * np.pi * ( k2fa[1]/t * (U-to)+k2fa[0]/t * (V-to))) * fft.ifft2(fS1aT)

    fS1aT0 = fft.fft2(S1aT)

    mA = np.sum(fS1aT * np.conj(fS1aT0))

    mA = mA / np.sum(fS1aT0 * np.conj(fS1aT0))

    return mA


if __name__ == '__main__':
    filename = '/Users/Ashley/PycharmProjects/SIMple/Data/SLM-SIM_Tetraspeck200_680nm.tif'

    original_data = get_image(filename, 1)
    fS1aTnoisy = return_shiffetd_fft(original_data)
    k2fa = [0.01, 114]
    psf = generate_PSF(1.2, 680, 97, 512)
    OTFo = return_shiffetd_fft(psf)

