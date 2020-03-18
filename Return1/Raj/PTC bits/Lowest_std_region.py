import numpy as np
import PTC_Samurai as sam


def lowest_std_region(step, file):
    array = sam.loadtiffs(file)

    x_pos = []
    y_pos = []
    std = []

    for x in range(0, array.shape[1], step):
        if x + 100 > array.shape[1]:
            break
        for y in range(0,array.shape[0], step):
            if y+100 > array.shape[0]:
                break
            cutout = array[x:x+100, y:y+100, :]
            x_pos.append(float(x))
            y_pos.append(float(y))
            std.append(np.std(cutout))

    position = std.index(min(std))
    x1, x2, y1, y2,low_std = x_pos[position], x_pos[position]+100, y_pos[position], y_pos[position]+100, std[position]
    print(x1, x2, y1, y2,low_std)
    return(x1, x2, y1, y2, low_std)


lowest_std_region(50, "/Users/RajSeehra/University/Masters/Semester 2/test folder/0.04.tif")