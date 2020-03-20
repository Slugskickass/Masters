import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


mean_data = np.load("/Users/RajSeehra/University/Masters/Semester 2/Git Upload Folder/Return1/Raj/PTC bits/mean_Pixel_PTCdata.npy")
std_data = np.load("/Users/RajSeehra/University/Masters/Semester 2/Git Upload Folder/Return1/Raj/PTC bits/std_Pixel_PTCdata.npy")

data_array = np.zeros((2, 26))


def at_pixel_stack(mean_data, std_data, x_pos, y_pos):
    for z in range(0, mean_data.shape[2]):
        if 10 < mean_data[y_pos, x_pos, z] < 1000:
            data_array[0, z] = mean_data[y_pos, x_pos, z]
            data_array[1, z] = np.square(std_data[y_pos, x_pos, z])

    return data_array

slope_intercept = []


for x_pos in range(0, mean_data.shape[1]):
    for y_pos in range(0, mean_data.shape[0]):
        data_array= at_pixel_stack(mean_data, std_data, x_pos, y_pos)
        slope, intercept, r_value, p_value, std_err = stats.linregress(data_array[0, :], data_array[1, :])
        slope_intercept.append(slope)
        slope_intercept.append(intercept)


slope_intercept = np.reshape(slope_intercept, (-1, 2)).T

plt.hist(slope_intercept[0])
plt.xlabel("Gradient")
plt.ylabel("Number of Pixels")
plt.title("Histogram of the gradient(gain) distribution \n for each pixel (cut out mean <10 or >1000)")

plt.show()

# plt.plot(slope, intercept, "*")
# plt.scatter(data_array[0, :], data_array[1:])
# plt.title("PTC at pixel: (" + str(x_pos) + ", " + str(y_pos) + ") (cut out mean <10 or >1000)")
# plt.text(50, 25, 'Slope= '+ str(slope) + ', Intercept= ' + str(intercept))
# plt.xlabel('Mean')
# plt.ylabel('Standard Deviation Squared')
# plt.show()