# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 05:35:23 2020

@author: 90759
"""
import numpy as np
import time
import matplotlib.pyplot as plt

#define a function to do the bubble sort
def bubble_sort(nums):
#    setting the number of bubble sort runs
    for i in range(len(nums) - 1):    
        for j in range(len(nums) - i - 1): 
            if nums[j] > nums[j + 1]:
                nums[j], nums[j + 1] = nums[j + 1], nums[j]
    return nums
#caculate the time
    
start_a = time.time()
a=np.random.rand(10)
print(bubble_sort(a))
time_a = (time.time() - start_a)

start_b= time.time()
b=np.random.rand(100)
print(bubble_sort(b))
time_b= (time.time() - start_b)

start_c=time.time()
c=np.random.rand(1000)
print(bubble_sort(c))
time_c= (time.time() - start_c)

start_d=time.time()
d=np.random.rand(10000)
print(bubble_sort(d))
time_d=(time.time() - start_d)
#it takes a long time to process the 100000 array
start_e=time.time()
e=np.random.rand(100000)
print(bubble_sort(e))
time_e=(time.time() - start_e)       

#plot the data
x = [10,100,1000,10000,100000]
y = [time_a,time_b,time_c,time_d,time_e]
plt.plot(x,y,'s-',color = 'r')
plt.xlabel("array length")
plt.ylabel("time")
plt.show()



