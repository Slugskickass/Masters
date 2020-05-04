import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.fft as fft
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.ndimage import gaussian_filter
from scipy import signal

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
OTF = return_shiffetd_fft(psf)

# Fourier of data
f_image_data = return_shiffetd_fft(image_data)

# Fourier of data multiplied by conjuagate of OTF
f_con_image = f_image_data * np.conj(OTF)

# Generate pattern
size_x = 20
size_K = 20
holdall = []
values = np.linspace(110, 120, size_x)
valuesk = np.linspace(0, 10, size_K)
Y, X = np.meshgrid(values, valuesk)
summed = np.zeros((size_x, size_K))
for indk, K in enumerate(valuesk):
    for index, I in enumerate(values):
        K_vector = [K, I]
        S1aT = generate_patter(512, K_vector)
        output = S1aT * fft.ifftshift(fft.ifft2(f_con_image))
        output = fft.fft2(output)
        summed[index, indk] = np.sum(np.conj(output) * fft.fft2(f_con_image))



fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, summed, cmap=cm.coolwarm, linewidth=0, antialiased=False)

plt.show()
positions = np.unravel_index(np.argmax(summed), summed.shape)
IPos = (X[positions])
KPos = (Y[positions])
print(np.sqrt(IPos**2 + KPos**2))
print(np.cos(IPos/KPos))











