#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 12:26:45 2020

@author: mattarnold
"""

import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

data_10 = genfromtxt('/Users/mattarnold/Masters/Return1/Matt/data_10', delimiter=',')
data_100 = genfromtxt('/Users/mattarnold/Masters/Return1/Matt/data_100', delimiter=',')
data_1000 = genfromtxt('/Users/mattarnold/Masters/Return1/Matt/data_1000', delimiter=',')
data_10000 = genfromtxt('/Users/mattarnold/Masters/Return1/Matt/data_10000', delimiter=',')

graph_array = np.zeros([4,3])
graph_array[:,0] = [10,100,1000,10000]

def data_mean_std (data):
    data_mean = np.mean(data)
    data_std = np.std(data)
    return data_mean, data_std



data_10_a = data_mean_std(data_10)
graph_array[0,1],graph_array[0,2] = data_10_a
data_100_a = data_mean_std(data_100)
graph_array[1,1],graph_array[1,2] = data_100_a
data_1000_a = data_mean_std(data_1000)
graph_array[2,1],graph_array[2,2] = data_1000_a
data_10000_a = data_mean_std(data_10000)
graph_array[3,1],graph_array[3,2] = data_10000_a

x_data = graph_array[:,0]
y_data = graph_array[:,1]
error = graph_array[:,2]

print(x_data,y_data)

plt.errorbar(x_data, y_data, yerr=error, uplims=True, lolims=True)
plt.xscale("log")
plt.yscale("log")
plt.axis(xlim=(0, 10000), ylim=(0, 30))
plt.ylabel("Mean time taken, s")
plt.xlabel("Number of data sorted")
plt.title('Mean time taken to sort data using Bubble Sort Pro')

plt.show
plt.savefig('mean_sort_time_graph')


#print(graph_array)