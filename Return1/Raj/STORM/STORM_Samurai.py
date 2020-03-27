from PIL import Image
import numpy as np
import os


def pixel_cutter(file_name, x_position, y_position, window_size_x, window_size_y, frame):
    img = Image.open(file_name)
    frame = frame - 1
    x = window_size_x
    y = window_size_y

    imgArray = np.zeros((img.size[1], img.size[0], img.n_frames), np.uint16)
    imgArray[:, :, frame] = img
    img.close()

    # Assign centre and ascertain coords for image, the -1 is to ensure even spreading either side of the centre point.
    xcoordmin = x_position - int(x / 2)-1
    xcoordmax = x_position + int(x / 2)
    ycoordmin = y_position - int(y / 2)-1
    ycoordmax = y_position + int(y / 2)

    # check no negative numbers
    if xcoordmin < 0:
        xcoordmin = 0
    if xcoordmax > img.size[0]:
        xcoordmax = img.size[0]
    if ycoordmin < 0:
        ycoordmin = 0
    if ycoordmax > img.size[1]:
        ycoordmax = img.size[1]

    # Plotting the area.
    return imgArray[ycoordmin:ycoordmax, xcoordmin:xcoordmax, frame]

#Example
# x = pixel_cutter("/Users/RajSeehra/University/Masters/Semester 2/Teaching_python-master/Images/bacteria.tif", 15,100, 15,15, 0)
# print(np.shape(x))
# plt.imshow(x)
# plt.show()


def Gaussian_Map(image_size, offset, centre_x, centre_y, width, amplitude):
    #Image Generation
    x, y = np.meshgrid(np.linspace(-1,1,image_size[1]), np.linspace(-1, 1, image_size[0]))
    dist = np.sqrt((x-centre_x) ** 2 + (y+centre_y) ** 2)
    intensity = offset + amplitude * np.exp(-( (dist)**2 / ( 2.0 * width**2 ) ) )
    return intensity


def get_file_list(dir):
    #dir = '/ugproj/Raj/Flash4/'
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


def filter_switcher(data, settings):
    switcher = {
        'kernel' : kernel_filter(data, settings["filter"]),
        'DOG' : difference_of_gaussians(data, settings("filter")),
    }

    return switcher.get(settings["filter"], data)      # return an error message is no match.