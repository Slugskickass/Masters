import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import general as genr
import ast
import os

def fitter_switcher(cropped_data, settings):
    switcher = {
        'raj': raj_fitter,
        'matt': matt_fitter,
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


## Matt
    ## As with cropping, this is suited currently to a .csv style workflow, where files are saved out at each step.
    ## The advantage of this approach is that metadata are available for each stage of the process, so problems are
    ## more easily identified.

# Function to convert arrays from cells in .csv converted pandas frames back from strings to numerical elements    
def array_clean(cutout_array):
   cutout_array = ','.join(cutout_array.replace('[ ', '[').split())
   return np.array(ast.literal_eval(cutout_array))

# Function to open the saved location data, add columns for processed data and clean arrays using above function
def build_mol_frame (csv_file_path=None):
    # If path is not specified, assume working directory
    if csv_file_path==None:
            molecules = pd.read_csv("{}/molecules_positions_crops.csv".format(os.getcwd()))
    # Otherwise, use specified path
    else:
        molecules = pd.read_csv("{}".format(csv_file_path))
    # Ditch old index column
    molecules = molecules.loc[:,["file_name", "x-coord", "y-coord", "cutout_square"]]
    # Add colmuns for fitting results
    molecules["popt"], molecules["centre-x"], molecules["centre-y"] = "", "", ""
    
    for index in molecules.index:
        clean_array = array_clean(molecules.loc[index,"cutout_square"])
        molecules.at[index,"cutout_square"] = clean_array
    # Return spruced-up dataframe
    return molecules

# FITTING FUNCTION
    #PARAMETERS: pandas dataframe generated in above function, size of Gaussian for fitting
    #RETURNS: pandas dataframe containing above data, plus fitting positions in x and y for centre
def matt_fitter (size, input_mols_df=None):
    mol_frame = build_mol_frame(input_mols_df)
    # Build "numberlines" of x and y in increments of one up to "size"
    x = np.linspace(0, size-1, size)
    y = np.linspace(0, size-1, size)
    # From these, build incremental grids in x and y for gaussian generation
    X, Y = np.meshgrid(x, y)
    # Convert incremental grids into 1d data for fitting
    xdata = np.vstack((X.ravel(), Y.ravel()))
    
    count=0
    # Work through index of dataframe, attempting a Gaussian fit with defined parameters, on the array in that row
    for index in mol_frame.index:
        try:
            mol_square = mol_frame.at[index,"cutout_square"]
            # Define guesses for Gaussian fit
            param_guess = [3,3,mol_square.shape[1],mol_square.shape[0],2,2,np.min(mol_square), np.max(mol_square) - np.min(mol_square)]
            param_array = np.asarray(param_guess)
            low_bound, up_bound = [-np.inf,-np.inf,0,0,1,1,-np.inf,-np.inf], [np.inf,np.inf,mol_square.shape[1],mol_square.shape[0],4,4,np.inf,np.inf]
            popt, pcov = curve_fit(genr._gaussian, xdata, mol_square.ravel(), p0=param_array, bounds=(low_bound,up_bound))
            # Save data for locations from fit to dataframe
            centre_x, centre_y = (mol_frame.loc[index,"x-coord"] - np.floor(size/2) + popt[3]), (mol_frame.loc[index,"y-coord"] - np.floor(size/2) + popt[4])
            mol_frame.loc[index,"popt"], mol_frame.loc[index,"centre-x"], mol_frame.loc[index,"centre-y"] = popt, centre_x, centre_y
        # If fit is unsuccessful (due to noisy frame), count it and continue to next frame
        except:
            #print("unable to fit square at", mol_frame.loc[index,["file_name","x-coord","y-coord"]])
            count+=1
            continue
    print("The number of failed fits is:", count)
    mol_frame.to_csv("localisation_data.csv")
    return mol_frame
