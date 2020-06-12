import PySimpleGUI as sg
import numpy as np
import matplotlib.pyplot as plt
import Camera_Samurai as sam
from scipy import signal as sig

sg.theme('DarkBlack')   # Adds a theme
# All the stuff inside the window. With parameters to define the user input within certain limits.
layout = [  [sg.Text('Image Type (Boring or Interesting)', size=(30, 1)),sg.Spin(("Boring", "Interesting"), initial_value="Interesting", size=(11, 1))],
            [sg.Text('Ground Truth Pixel Size (nm)', size=(30, 1)), sg.InputText("5")],
            [sg.Text('Photon Count', size=(30, 1)), sg.InputText("10")],
            [sg.Text('Exposure Time (s)', size=(30, 1)), sg.InputText("1")],
            [sg.Text('Numerical Aperture', size=(30, 1)), sg.Slider((0, 2), 1.4, 0.1, 0.5, 'h')],
            [sg.Text('Wavelength (250-700)', size=(30, 1)), sg.Slider((200,700), 440, 10, 100, 'h')],
            [sg.Text('Camera Pixel Size', size=(30, 1)), sg.InputText("6500")],
            [sg.Text('Magnification (1-100)', size=(30, 1)), sg.InputText("100")],
            [sg.Text('Quantum Efficiency (0-1)', size=(30, 1)), sg.Slider((0,1), 0.75, 0.1, 0.25, 'h')],
            [sg.Text('Gain', size=(30, 1)), sg.InputText("2")],
            [sg.Text('Read Noise Mean', size=(30, 1)), sg.InputText("2")],
            [sg.Text('Read Noise Standard Deviation', size=(30, 1)), sg.InputText("2")],
            [sg.Text('Fixed Pattern Deviation', size=(30, 1)), sg.InputText("0.001")],
            [sg.Text('Preview?', size=(30, 1)), sg.Spin(("Y", "N"), initial_value="Y", size=(5, 1))],
            [sg.Text('Save?', size=(30, 1)), sg.Spin(("Y", "N"), initial_value="Y", size=(5, 1))],
            [sg.Text('Filename? (TIFF by default)', size=(30, 1)), sg.InputText("Camera_image")],
            [sg.Text('Progress Bar', size=(30, 1)), sg.ProgressBar(100, bar_color=("blue", "white"), key="Progress Bar")],
            [sg.Button('Run'), sg.Button('Cancel')] ]

# Creates the Window based on the layout.
window = sg.Window('Camera Emulator', layout)
# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()   # Reads the values entered into a list.
    if event in (None, 'Cancel'):   # if user closes window or clicks cancel
        break

    # Define the progress bar to allow updates
    progress_bar = window.FindElement('Progress Bar')

    ### INPUT PARAMETERS from values ###
    # GROUND TRUTH
    image_type = values[0]  # Ground truth image if you want user input. Choose: "Boring", "Interesting"
    groundpixel = int(values[1])  # Pixel size e.g. pixels = 5nm
    photon_count = int(values[2])  # Photon count value per unit time (same unit used in exposure)
    exposure_time = float(values[3])  # seconds of exposure
    # PSF
    NA = float(values[4])  # Numerical aperture
    wavelength = int(values[5])  # Wavelength in nanometres
    # CAMERA
    camera_pixel_size = int(values[6])  # Camera pixel size in nanometres. usual sizes = 6 microns or 11 microns
    magnification = int(values[7])  # Lens magnification
    QE = float(values[8])  # Quantum Efficiency
    gain = float(values[9])  # Camera gain. Usually 2 per incidence photon
    # NOISE
    read_mean = float(values[10])  # Read noise mean level
    read_std = float(values[11])  # Read noise standard deviation level
    fixed_pattern_deviation = float(values[12])  # Fixed pattern standard deviation. usually affects 0.1% of pixels.
    # SAVE
    Preview = values[13]
    SAVE = values[14]  # Save parameter, input Y to save, other parameters will not save.
    filename = values[15]

    progress_bar.UpdateBar(10)


    ### SAMPLE GENERATION ###
    # Make a Ground Truth
    # Based on pixel size. e.g. 5nm pixel, therefore: 10 microns = 2kx2k array
    ground, ground_window = sam.image_selector(image_type, groundpixel, photon_count, exposure_time)
    #### END AWESOME SAMPLE MAKER ####
    print("Sample Created!")
    progress_bar.UpdateBar(20)


    ### LENS SIMULATOR ###
    # Lens == A diffraction limited blur. Dependent on wavelength and NA
    psf = sam.psf_generator(NA, wavelength, groundpixel, ground_window)
    print("PSF made")
    progress_bar.UpdateBar(30)

    # Apply the lens as a convolution of the two arrays producing a diffraction limited image.
    dif_lim = sig.fftconvolve(ground, psf, "same")  # Fiddling with fourier space to convolute, much faster.
    print("Convolution in progress")
    progress_bar.UpdateBar(40)


    ### CAMERA SETUP ###
    # Camera sensor, based on optical magnification and pixel size.
    camerapixel_per_groundpixel = camera_pixel_size / groundpixel

    # Used to determine the number of the ground pixels that exist within each bin
    mag_ratio = camerapixel_per_groundpixel / magnification
    print("Overall Image Binning (ground pixels per bin):", mag_ratio, "by", mag_ratio)
    progress_bar.UpdateBar(50)


    ### IMAGING TIME ###
    # Initialise an empty array, with a size calculated by the above ratios.
    # Gives us a rounded down number of pixels to bin into to prevent binning half a bin volume into a pixel.
    camera_image = np.zeros((int(dif_lim.shape[0] // mag_ratio), int(dif_lim.shape[1] // mag_ratio)))

    # Iterate each position in the array and average the pixels in the range from the diffraction limited image.
    # We use the mag_ratio to step across the array and select out regions that are multiples of it out.
    for y in range(0, camera_image.shape[0]):
        for x in range(0, camera_image.shape[1]):
            pixel_section = dif_lim[y * int(mag_ratio):y * int(mag_ratio) + int(mag_ratio),
                            x * int(mag_ratio):x * int(mag_ratio) + int(mag_ratio)]
            camera_image[y, x] = np.mean(pixel_section)  # Take the mean value of the section and bin it to the camera.
    print("Collecting Data")
    progress_bar.UpdateBar(60)

    # Account for Quantum efficiency.
    camera_image = camera_image * QE
    print("QE step")
    progress_bar.UpdateBar(70)


    ### ADD NOISE ###
    # Add read and shot noise.
    print("That pesky noise...")
    progress_bar.UpdateBar(80)

    camera_Rnoise = sam.read_noise(camera_image, read_mean, read_std)
    camera_Snoise = sam.shot_noise(np.sqrt(photon_count * exposure_time), camera_image)

    # Add up the camera, read and shot noises.
    camera_RSnoise = camera_image + camera_Rnoise + camera_Snoise

    # FP noise remains the same for the camera, hence we fix the seed.
    # We can simulate one by producing a normal distribution around 1 with a deviation relative to the number of pixels that
    # would commonly deviate under such parameters.
    np.random.seed(100)
    camera_FPnoise = np.random.normal(1, fixed_pattern_deviation, (camera_image.shape[0], camera_image.shape[1]))

    # Multiply by the fixed pattern noise
    camera_all_noise = camera_RSnoise * camera_FPnoise


    ### GAIN, COUNT AND INTEGER ###
    # Multiply by gain to convert from successful incidence photons and noise to electrons.
    camera_gain = camera_all_noise * gain
    print("All about them gains.")
    progress_bar.UpdateBar(90)

    # 100 count added as this is what camera's do.
    camera_view = camera_gain + 100

    # Convert to integer as a camera output can only take integers
    # Conversion to: USER INT VALUE 16
    camera_view = camera_view.astype(np.uint16)
    print("Complete")
    progress_bar.UpdateBar(100)


    ### PREVIEW ###
    if Preview == "Y":
        plt.imshow(camera_view)
        plt.plot


    ### SAVE ###
    if SAVE == "Y":
        sam.savetiff(filename+".tif", camera_view)
        print("Image saved.")

window.close()

# Look at sg.ProgressBar