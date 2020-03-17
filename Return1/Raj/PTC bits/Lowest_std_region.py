import numpy as np
import PTC_Samurai as sam
import matplotlib.pyplot as plt

array = sam.loadtiffs("/Users/RajSeehra/University/Masters/Semester 2/test folder/0.04.tif")

step = 100

x_pos = []
y_pos = []
std = []

for x in range(0, array.shape[1], step):
    for y in range(0,array.shape[0], step):
        cutout = array[x:x+step, y:y+step, 0]
        x_pos.append(float(x))
        y_pos.append(float(y))
        std.append(np.std(cutout))

position = std.index(min(std))
a,b,c = x_pos[position], y_pos[position], std[position]
print(a,b,c)