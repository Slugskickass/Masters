import numpy as np
import PTC_Samurai as sam
import sys


# Once you have determined your brightest file from your batch, run the program with that as the input file.

file = sys.argv[1]
step = int(sys.argv[2])

def lowest_std_region(step, file):
    array = sam.loadtiffs(file)

    # lists for clarity and alignment of numerical positions.
    x_pos = []
    y_pos = []
    std = []

    for x in range(0, array.shape[1], step):
        if x + 100 > array.shape[1]:            # Prevents a box being less than 100x100 in x.
            break
        for y in range(0,array.shape[0], step):
            if y+100 > array.shape[0]:          # Prevents a box being less than 100x100 in y.
                break
            cutout = array[x:x+100, y:y+100, :]     # Cutout in x an y the appropriate box.
            x_pos.append(float(x))
            y_pos.append(float(y))
            std.append(np.std(cutout))

    position = std.index(min(std))
    x1, x2, y1, y2,low_std = x_pos[position], x_pos[position]+100, y_pos[position], y_pos[position]+100, std[position]
    print (low_std)
    return(x1, x2, y1, y2)


coords = lowest_std_region(step, file)
coords = np.reshape(coords, (2,2))

np.save("crop_box.npy", coords)