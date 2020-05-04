import numpy as np
import scipy.fft as fft

# Done ?
def OTFedgeF(OTFo):
    w = np.shape(OTFo)[0]
    wo = w/2

    OTF1 = OTFo[wo+1, :]
    OTFmax =np.max(np.abs(OTFo))
    OTFtruncate = 0.01
    i = 1
    while (np.abs(OTF1[1,i]) < OTFtruncate * OTFmax):
        Kotf = wo+1-i
        i = i + 1;
    return(Kotf)



def PhaseKai2opt(k2fa,fS1aTnoisy,OTFo):
#% The size of the image
    w = np.shape(fS1aTnoisy)[0]
    wo = w/2

# Here the FFT of the original image is muliplied by the
# 1 -the OTF^10 (?)
# The new image is then multiplied byt the conj of the OTF to give the final image
    fS1aTnoisy = fS1aTnoisy*(1-1*OTFo**10)
    fS1aT = fS1aTnoisy*np.conj(OTFo)

    Kotf = OTFedgeF(OTFo)

    t = w

    to = t/2
    u = np.linspace(0, t-1, t)
    v = np.linspace(0, t-1, t)
    [U, V] = np.meshgrid(u, v)

    S1aT = np.exp( -1j * 2 * np.pi * ( k2fa[1]/t * (U-to)+k2fa[0]/t * (V-to))) * fft.ifft2(fS1aT)

    fS1aT0 = fft.fft2(S1aT)