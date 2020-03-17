import numpy as np
import PTC_Samurai as sam
import sys

darktiff = sys.argv[1]

file = sam.loadtiffs(darktiff)


def axisaverager(file):
    array = np.mean(file, 2)
    return array


array = axisaverager(file)


np.save("darkframes.npy", array)