import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import ast


def fitter_switcher(cropped_data, settings):
    switcher = {
        'raj': raj_fitter,
        # 'matt': matt_fitter,
    }

    return switcher.get((settings.get("fitting parameters: ", {}).get("Student_name:")), cropped_data)\
        (cropped_data, (settings.get("fitting parameters: ", {}).get("size:")))


def gaussian_fitter(data, array, index):
    def gaussian(x, y, x0, y0, xalpha, yalpha, offset, A):
        return offset + A * np.exp(-((x-x0)/xalpha)**2 - ((y-y0)/yalpha)**2)

    def _gaussian(M, *args):
        x, y = M
        arr = gaussian(x, y, *args)
        return arr

    # Generate a gaussian
    x_size = array.shape[1]
    y_size = array.shape[0]
    x = np.linspace(0, x_size, y_size)
    y = np.linspace(0, x_size, y_size)
    X, Y = np.meshgrid(x, y)
    Z = array

    # Fitting: produces xdata to compare to, ravels our input array as the comparison.
    xdata = np.vstack((X.ravel(), Y.ravel()))
    guess = [x_size/2, y_size/2, 2.0, 2.0, np.min(Z), np.max(Z)-np.min(Z)]  # Generates guesses.
    # bounds in order: x0, y0, xalpha, yalpha, offset, A.
    popt, pcov = curve_fit(_gaussian, xdata, Z.ravel(), p0=np.asarray(guess),
                           bounds=((0, 0, 1, 1, 0, 0), (x_size, y_size, 4, 4, np.inf, np.inf)))

    # Correct the x and y values by their original position.
    # Actual position = original cut position - cut size/2 + xo
    corrected_popt0 = int(data['X'][index]) - x_size/2 + popt[0]
    corrected_popt1 = int(data['Y'][index]) - y_size/2 + popt[1]

    # print(popt[0], popt[1], guess)
    print(corrected_popt0, corrected_popt1, guess)

    return Z, corrected_popt0, corrected_popt1, guess


def clean_array(array_string):
    array_string = ','.join(array_string.replace('[ ', '[').split())
    return np.array(ast.literal_eval(array_string))


def raj_fitter(data, empty):
    data = data
    data = data.reset_index()
    data = data.drop('index', axis=1)

    fitted_centres = pd.DataFrame(columns=['X', 'Y', 'guess'])
    for i in range(0, data.shape[0]):
        # array = clean_array(data["cutout_array"][i])
        array = data["cutout_array"][i]

        fitted_data = gaussian_fitter(data, array, i)
        current = pd.DataFrame({'X': fitted_data[1], 'Y': fitted_data[2], 'guess': [fitted_data[3]]},
                               index=["{}".format(i)])
        fitted_centres = pd.concat([fitted_centres, current], axis=0)  # Final dataframe.
    return fitted_centres

