import numpy as np
import matplotlib.pyplot as plt
import pywt


def kernel_filter(data, matrix, empty):
    image = data
    kernel = np.asarray(matrix)

    # Error check in case the matrix has an even number of sides.
    if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        print("The matrix has an even number of rows and/or columns. Please make them odd and run again.")

    if sum(sum(kernel)) != 1:       # Quick check to ensure the kernel matrix is within parameters.
        print("Error, this matrix's summation value is not equal to 1. This can change the final image.")
        print("The program has divided the matrix by the sum total to return it to a value of 1.")
        print(("This total value is: " + str(sum(sum(kernel)))))
        kernel = kernel / sum(sum(kernel))
        print(kernel)

    # Takes the filter size and allows for a rectangular matrix.
    edge_cover_v = (kernel.shape[0] - 1) // 2
    edge_cover_h = (kernel.shape[1] - 1) // 2

    # to determine if the file has multiple frames or not.
    if data.ndim > 2: #if image.shape[2] > 1: ##I have edited as 2D image have no 3rd dimension so programme crashes if passed 2D data. Matt
        # adds an edge to allow pixels at the border to be filtered too.
        bordered_image = np.pad(image, ((edge_cover_v, edge_cover_v), (edge_cover_h, edge_cover_h), (0, 0)))
        # Our blank canvas below.
        processed_image = np.zeros((bordered_image.shape[0], bordered_image.shape[1], bordered_image.shape[2]))

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
        bordered_image = np.pad(image, ((edge_cover_v, edge_cover_v),(edge_cover_h, edge_cover_h)))
        # Our blank canvas below.
        processed_image = np.zeros((bordered_image.shape[0], bordered_image.shape[1]))

        # Iterates the x and y positions.
        for x in range(edge_cover_h, bordered_image.shape[1]-edge_cover_h):
            for y in range(edge_cover_v, bordered_image.shape[0]-edge_cover_v):
                kernel_region = bordered_image[y-edge_cover_v:y+edge_cover_v+1, x-edge_cover_h:x+edge_cover_h+1]
                k = (kernel * kernel_region).sum()
                processed_image[y,x] = k
        # Cuts out the image to be akin to the original image size.
        processed_image = processed_image[edge_cover_v:processed_image.shape[0]-edge_cover_v, edge_cover_h:processed_image.shape[1]-edge_cover_h]
    return processed_image


def randompoints():
    # Random seed generator
    np.random.seed()
    x = np.random.randint(0, 101, (100, 100))
    mapper = pywt.threshold(x, 100, substitute=0, mode='greater')
    return mapper


mapper = randompoints()

matrix = [(1,2,1),(2,4,2),(1,2,1)]
gauss_blur = kernel_filter(mapper, matrix, 0)

plt.subplot(121)
plt.imshow(mapper)
plt.title("Random Points")
plt.subplot(122)
plt.imshow(gauss_blur)
plt.title('Gaussian blurred')
plt.show()
