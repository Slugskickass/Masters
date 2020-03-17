import numpy as np
import matplotlib.pyplot as plt
import PTC_Samurai as sam
import sys

darktiff = sys.argv[1]

file = sam.loadtiffs(darktiff)


def axisaverager(file):
    array = np.mean(file, 2)
    return array


array = axisaverager(file)


sam.savetiffs("darkframes.npy", array)