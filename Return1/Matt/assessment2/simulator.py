"""
@author: mattarnold
"""

import numpy as np
from PIL import Image as im
from scipy import signal
import argparse

# PSF_SIMULATOR: calculate an approximated point spread function for the system, scaling by the 
    # deisired size of the pixels in the ground truth image for use in calculating the effect of 
    # diffraction in the final image
    # Inputs: the numerical aperture of the lens, the emission wavelength,[ GT pixel size, PSF dimensions]
def psf_simulator (NA, wavelength, pixel_size=5, frame_size=450):
    sigma = (np.sqrt(8*np.log(2)) * wavelength / (2 * NA)) / pixel_size
    xo = np.floor(frame_size / 2)
    u = np.linspace(0, frame_size - 1, frame_size)
    v = np.linspace(0, frame_size - 1, frame_size)
    [U, V] = np.meshgrid(u, v)
    psf = np.exp(-1 * ((((xo - U) ** 2) / sigma ** 2) + (((xo - V) ** 2) / sigma ** 2)))
    return psf

# CAM_PARAMS: calculate the effective size of camera pixels after magnification
    # Input: the raw camera pixel size (nm), magnification (x)
    # Output: the effective pixel size (nm)
def cam_params(raw_px_size, mag):
    effective_px_size = int(raw_px_size / mag)
    return effective_px_size

# NOISE: calculate a random noise component for each pixel
    # Input: signal magnitude (bit value)
    # Output: sum of shot read and fixed pattern noise
def noise(signal_size):        
    shot_noise = abs(np.random.normal(2,2))
   
    signal_root = np.sqrt(signal_size)

    if signal_root < 1:
       signal_root=1.0
    read_noise = np.random.poisson(signal_root)
    
    
    fp_noise = abs(signal_size - (signal_size * abs(np.random.normal(1,0.1))))
    
    noise_val = shot_noise + read_noise + fp_noise
    return noise_val


# BINNING: average theoretical ground truth "pixel" values from the sample plane into the area of corresponding 
    # camera pixels in the image plane, taking into account the noise, pixel-induced spatial binning and camera gain.
    # Inputs: the ground truth image after convolution with a calculated point spread function (numpy 2d array), 
        # the effective pixel size of the camera as calculated by cam_params (nm), option to vary ground truth "pixel"
        # size (nm, default = 5nm)
    # Output: the binned image, with dimensions calculated by the integer number of bins which can be composed from
        # the ground truth image (numpy 2d array)
def binning(psf_convolved_gt, cam_px_size, quantum_efficiency, gt_px_size = 5):
    gt_side = psf_convolved_gt.shape[1]
    bin_size = cam_px_size//gt_px_size
    camera_gain = 100
    
    num_cam_px = int(gt_side // bin_size)
    cam_array = np.zeros([num_cam_px,num_cam_px])
    noise_array = np.zeros([num_cam_px,num_cam_px])
    print("The size of the simulated camera chip is {} pixels, for an input of {}nm camera pixels".format(num_cam_px, cam_px_size))
    
    for i in range(1,num_cam_px-1):
        work_array = gt_array[((i-1)*bin_size):(i*bin_size),:]
    
        for j in range(1,num_cam_px-1):
            work_bin = work_array[:,((j-1)*bin_size):(j*bin_size)]
            bin_val = np.mean(work_bin)

            noise_val = noise(bin_val)

            cam_array[i,j] = int ((bin_val*quantum_efficiency) + noise_val + camera_gain)
            noise_array[i,j] = noise_val
    mean_noise = np.mean(noise_array)
    return cam_array, noise_array, mean_noise

#SAVE_IMG: converts the output of the simulation to a an image file format and saves it to disk
    # Inputs: desired file name including file type extension e.g. "~.tif", 2D array-type data for saving
def save_img(file_name, data):
    images = im.fromarray(data[:, :])
    images.save(file_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Produce a simulation of diffraction and camera noise')
    parser.add_argument('-w', "-wavelength", "-lambda", metavar='w', type=float ,required=True, help='the fluorophore emission wavelngth')
    parser.add_argument('-na', "-NA", metavar='na', type=float ,required=True, help='the system numerical aperture')
    parser.add_argument('-px', "-pixel", metavar='px', type=float ,required=True, help='raw camera pixel size (IN nanometres)')
    parser.add_argument('-m', "-mag", metavar='m', type=float ,required=True, help='the system magnification, in times')
    parser.add_argument('-qe', "-q", metavar='qe', type=float ,default=1,  help='the camera quantum efficiency')
    parser.add_argument('-i', "-gt", "-input", metavar='i', type=str , default="xHTlB.jpg", help='ground truth image (optional)')
    parser.add_argument('-o', "-save", "-output", metavar='o', type=str , default="simulated_image.tif", help='save simulated image as (optional, include file type extension, e.g .tif; path can be included)')
    parser.add_argument('-n', '-noise', metavar='n', type=str, default=None, help='optional, saves image of noise added to camera pixels in simulation to specified file, define as for output image')
    args = parser.parse_args()
    
    if args.w > 750 or args.w < 350:
        print("Wavelength entered:",args.w)
        raise ValueError("Wavelength should be in nanometers, for visible light")
    if args.na < 0 or args.na > 2:
        print("Numerical aperture entered:",args.na)
        raise ValueError("This NA is incorrect")
    if args.px < 5000:
        raise ValueError("Ensure the pixel size entered is in nanometres, pre-magnification")
    if args.qe >1:
        raise ValueError("Quantum efficiency must 1 or less")
    
    ground_truth = im.open(args.i)
    gt_array = np.asarray(ground_truth)
    gt_array = gt_array[:,:,1]
    
    px_size = cam_params(args.px,args.m)
    psf = psf_simulator (args.na,args.w)
    convolution = signal.fftconvolve(gt_array,psf,mode="same")
    binned, noise_array, mean_noise = binning(convolution, px_size, args.qe)        
    save_img(args.o, binned)
    if args.n !=None:
        save_img("{}".format(args.n),noise_array)
        print("The average noise added to a pixel was", mean_noise, "counts")
