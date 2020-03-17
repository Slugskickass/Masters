import numpy as np
import matplotlib.pyplot as plt
import PTC_Samurai as sam

file = sam.loadtiffs("/Users/RajSeehra/University/Masters/Semester 2/Teaching_python-master/Week 2/Data/640.tif")


def axisaverager(file):
    array = np.mean(file, 2)
    return array


array = axisaverager(file)


# sam.savetiffs("darkframes.npy", array)