from PIL import Image
import numpy as np
import os


def psf_generator(NA, wavelength, pixel_size=100, frame_size=100, correction=1):
    # FWHM = np.sqrt(8*np.log(2)) * sigma
    sigma = (np.sqrt(8*np.log(2)) * wavelength / (2 * NA * correction)) / pixel_size
    t = frame_size
    xo = np.floor(t / 2)
    u = np.linspace(0, t - 1, t)
    v = np.linspace(0, t - 1, t)
    [U, V] = np.meshgrid(u, v)
    banana = np.exp(-1 * ((((xo - U) ** 2) / sigma ** 2) + (((xo - V) ** 2) / sigma ** 2)))
    return banana



def Gaussian_Map(image_size, offset, centre_x, centre_y, width, amplitude):
    # Image Generation
    x, y = np.meshgrid(np.linspace(-image_size[1]//2, image_size[1]//2, image_size[1]),
                       np.linspace(-image_size[0]//2, image_size[0]//2, image_size[0]))
    dist = np.sqrt((x-centre_x) ** 2 + (y+centre_y) ** 2)
    intensity = offset + amplitude * np.exp(-(dist ** 2 / (2.0 * width ** 2)))
    return intensity


def get_file_list(dir):
    # dir = '/ugproj/Raj/Flash4/'
    file_list = []
    for file in os.listdir(dir):
        if file.endswith(".tif"):
            file_name = dir + '/' + file
            file_list.append(file_name)
    return file_list

# single frame
def savetiff(file_name, data):
    images = Image.fromarray(data[:, :])
    images.save(file_name)


# single frame
def image_open(file):
    # open the file
    file_name = file
    img = Image.open(file_name)
    # generate the array and apply the image data to it.
    imgArray = np.zeros((img.size[1], img.size[0], img.n_frames), np.uint16)
    imgArray[:, :, 0] = img
    img.close()


# multi stack tiffs
def loadtiffs(file_name):
    img = Image.open(file_name)
    #print('The Image is', img.size, 'Pixels.')
    #print('With', img.n_frames, 'frames.')

    imgArray = np.zeros((img.size[1], img.size[0], img.n_frames), np.float32)
    for I in range(img.n_frames):
        img.seek(I)
        imgArray[:, :, I] = np.asarray(img)
    img.close()
    return (imgArray)


# as above
def savetiffs(file_name, data):
    images = []
    for I in range(np.shape(data)[2]):
        images.append(Image.fromarray(data[:, :, I]))
        images[0].save(file_name, save_all=True, append_images=images[1:])


def kernel_filter(data, matrix):
    image = data
    kernel = np.asarray(matrix, dtype=complex)

    # Error check in case the matrix has an even number of sides.
    if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        print("The matrix has an even number of rows and/or columns. Please make them odd and run again.")

    if sum(sum(kernel)) != 1:       # Quick check to ensure the kernel matrix is within parameters.
        print("Error, this matrix's summation value is not equal to 1. This can change the final image.")
        print("The program has divided the matrix by the sum total to return it to a value of 1.")
        print(("This total value is: " + str(sum(sum(kernel)))))
        kernel = kernel / sum(sum(kernel))
        # print(kernel)

    # Takes the filter size and allows for a rectangular matrix.
    edge_cover_v = (kernel.shape[0] - 1) // 2
    edge_cover_h = (kernel.shape[1] - 1) // 2

    # to determine if the file has multiple frames or not.
    if data.ndim > 2:
        # adds an edge to allow pixels at the border to be filtered too.
        bordered_image = np.pad(image, ((edge_cover_v, edge_cover_v), (edge_cover_h, edge_cover_h), (0, 0)))
        # Our blank canvas below.
        processed_image = np.zeros((bordered_image.shape[1], bordered_image.shape[0], bordered_image.shape[2]), dtype=complex)

        # Iterates the z, x and y positions.
        for z in range(0, bordered_image.shape[2]):
            for x in range(edge_cover_h, bordered_image.shape[1] - edge_cover_h):
                for y in range(edge_cover_v, bordered_image.shape[0] - edge_cover_v):
                    kernel_region = bordered_image[y - edge_cover_v:y + edge_cover_v + 1,
                                    x - edge_cover_h:x + edge_cover_h + 1, z]
                    k = (kernel * kernel_region).sum()
                    processed_image[y, x, z] = k
        # Cuts out the image to be akin to the original image size.
        processed_image = processed_image[edge_cover_v:processed_image.shape[0] - edge_cover_v,
                          edge_cover_h:processed_image.shape[1] - edge_cover_h, :]

    else:
        # adds an edge to allow pixels at the border to be filtered too.
        bordered_image = np.pad(image, ((edge_cover_v, edge_cover_v), (edge_cover_h, edge_cover_h)))
        # Our blank canvas below.
        processed_image = np.zeros((bordered_image.shape[1], bordered_image.shape[0]), dtype=complex)

        # Iterates the x and y positions.
        for x in range(edge_cover_h, bordered_image.shape[1]-edge_cover_h):
            for y in range(edge_cover_v, bordered_image.shape[0]-edge_cover_v):
                kernel_region = bordered_image[y-edge_cover_v:y+edge_cover_v+1, x-edge_cover_h:x+edge_cover_h+1]
                k = (kernel * kernel_region).sum()
                processed_image[y, x] = k
        # Cuts out the image to be akin to the original image size.
        processed_image = processed_image[edge_cover_v:processed_image.shape[1]-edge_cover_v, edge_cover_h:processed_image.shape[0]-edge_cover_h]
    return processed_image
