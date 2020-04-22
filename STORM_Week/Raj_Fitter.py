import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import ast


def fitter(data, size):
    def gaussian(x, y, x0, y0, xalpha, yalpha, A):
        return A * np.exp(-((x-x0)/xalpha)**2 - ((y-y0)/yalpha)**2)


    def _gaussian(M, *args):
        x, y = M
        arr = 100+ gaussian(x, y, *args)
        return arr

    # Generate a gaussian
    size = size
    x = np.linspace(0, size, size)
    y = np.linspace(0, size, size)
    X, Y = np.meshgrid(x, y)
    # Replace below line with data
    Z = data


    # Fitting
    xdata = np.vstack((X.ravel(), Y.ravel()))
    guess = [4.0, 3.0, 1.0, 1.0, 1]
    popt, pcov = curve_fit(_gaussian, xdata, Z.ravel(), p0=np.asarray(guess))
    # plt.imshow(Z)
    print(popt[0], popt[1])
    # plt.show()

    return Z, popt[0], popt[1]


def clean_array(array_string):
    array_string = ','.join(array_string.replace('[ ', '[').split())
    return np.array(ast.literal_eval(array_string))


data = pd.read_csv("/Users/RajSeehra/University/Masters/Semester 2/Git Upload Folder/STORM_Week/particle_position_crops.csv")

# eight = clean_array(data['cutout_array'][27])
# plt.imshow(eight)
# plt.show()

data = data.drop([7, 27], axis=0)

data = data.reset_index()
data = data.drop('index', axis=1)

fitted_centres = pd.DataFrame(columns=['popt[0]', 'popt[1]'])
for i in range(0, data.shape[0]):
    array = clean_array(data["cutout_array"][i])
    fitted_data = fitter(array, 7)
    current = pd.DataFrame({'popt[0]': fitted_data[1], 'popt[1]': fitted_data[2]}, index=["{}".format(i)])
    fitted_centres = pd.concat([fitted_centres, current], axis=0)  # Final dataframe.



# ### FITTING ###     Using code from online. UNKNOWN TERRITORY BELOW HERE!!!
# def gaussian(height, center_x, center_y, width_x, width_y):
#     """Returns a gaussian function with the given parameters"""
#     width_x = float(width_x)
#     width_y = float(width_y)
#     return lambda x, y: height*np.exp(
#                 -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)
#
# def moments(data):
#     """Returns (height, x, y, width_x, width_y)
#     the gaussian parameters of a 2D distribution by calculating its
#     moments """
#     total = data.sum()
#     X, Y = np.indices(data.shape)
#     x = (X*data).sum()/total
#     y = (Y*data).sum()/total
#     col = data[:, int(y)]
#     width_x = np.sqrt(np.abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
#     row = data[int(x), :]
#     width_y = np.sqrt(np.abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
#     height = data.max()
#     return height, x, y, width_x, width_y
#
# def fitgaussian(data):
#     """Returns (height, x, y, width_x, width_y)
#     the gaussian parameters of a 2D distribution found by a fit"""
#     params = moments(data)
#     errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) -
#                                  data)
#     p, success = optimize.leastsq(errorfunction, params)
#     return p
#
# # Define the array to be used. MOD this to iterate.
# array1 = cutout_dataframe["cutout_array"][52]
#
# # Provide a base to plot on for imaging
# plt.matshow(array1)
#
# # Run the function to produce the fit
# final_parameters = fitgaussian(array1)
# fit = gaussian(*final_parameters)
#
# plt.contour(fit(*np.indices(array1.shape)), cmap=plt.cm.copper)
# ax = plt.gca()
# (height, x, y, width_x, width_y) = final_parameters
#
# plt.text(0.95, 0.05, """
# x : %.1f
# y : %.1f
# width_x : %.1f
# width_y : %.1f""" %(x, y, width_x, width_y),
#         fontsize=16, horizontalalignment='right',
#         verticalalignment='bottom', transform=ax.transAxes)