import STORM_Samurai as sam
import numpy as np
import matplotlib.pyplot as plt

array = sam.loadtiffs("/Users/RajSeehra/University/Masters/Semester 2/Teaching_python-master/Week 2/Data/640.tif")


def kernel_filter(data, matrix):
    image = data
    kernel = np.asarray(matrix)
    if sum(sum(kernel)) != 1:       # Quick check to ensure the kernel matrix is within parameters.
        print("Error, this matrix's summation value is not equal to 1. This can change the final image.")
        print("The program has divided the matrix by the sum total to return it to a value of 1.")
        print(("This total value is: "+ str(sum(sum(kernel)))))
        kernel = kernel / sum(sum(kernel))
        print (kernel)

    # Takes the filter size and allows for a rectangular matrix.
    edge_cover_v = (kernel.shape[0] - 1) // 2
    edge_cover_h = (kernel.shape[1] - 1) // 2

    # adds an edge to allow pixels at the border to be filtered too.
    # ??? should i change the zeros to replicate the edge pixels?
    bordered_image = np.zeros((data.shape[0]+2*edge_cover_v, data.shape[1]+2*edge_cover_h))
    # makes the central pixels the ones from the image, leaving surrounding as zeroes.
    bordered_image[edge_cover_v:bordered_image.shape[0]-edge_cover_v, edge_cover_h: bordered_image.shape[1]-edge_cover_h] = image
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
    return (processed_image)


matrix = np.asarray([(1, 5, 1), (5, 25, 5), (1, 5, 1)])

# Data input is a single image frame and the matrix "? multipliers"
img = kernel_filter(array[:,:,0], matrix)


plt.subplot(121)
plt.imshow(array[:,:,0])
plt.title("Original")
plt.subplot(122)
plt.imshow(img)
plt.title("After")
plt.show()