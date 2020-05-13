import numpy as np
import matplotlib.pyplot as plt
import microscPSF as msPSF

# Radial PSF
mp = msPSF.m_params     # Microscope Parameters as defined in microscPSF. Dictionary format.

pixel_size = 0.05       # In ?microns
xy_size = 201           # In pixels.

pv = np.arange(-5.01, 5.01, pixel_size)     # Creates a 1D array stepping up by denoted pixel size,
                                            # Essentially stepping in Z.

psf_xy1 = msPSF.gLXYZParticleScan(mp, pixel_size, xy_size, pv)       # Matrix ordered (Z,Y,X)

psf_total = psf_xy1
plt.imshow(psf_total[30, :, :])

# from PIL import Image
# images = []
# for m in psf_xy:
#     images.append(Image.fromarray(m))
# images[0].save('psf.tiff', save_all=True, append_images=images[1:])
