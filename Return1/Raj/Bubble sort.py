import numpy as np
import matplotlib.pyplot as plt
import time


def bubble_sort(array_length):
    array = np.random.randint(1000, size=array_length)    # generates out array
    for i in range(len(array)):             # iterates up to the len of the array
        count = 0                           # provides a resetting counter that tells us if we have swapped no.
        for i in range(1, len(array)):      # first iteration of the sort.
            if array[i-1] > array[i]:       # iterative step to sort
                array[i-1], array[i] = array[i], array[i-1]
                count = count + 1
        if count == 0:                      # if we haven't swapped any numbers in an iteration then we are done.
            break
    return array


def timer(array_length):        # single run through time period
    start = time.time()
    bubble_sort(array_length)
    end = time.time()
    duration = end - start
    return duration


def repetitions(reps, array_length):        # iterates the timer function.
    times = []
    for i in range(reps):
        times.append(timer(array_length))   # append the returned value to the list
    mean = np.mean(times)
    std = np.std(times)
    return array_length, mean, std


# Final arrays by array size,
array10 = repetitions(10, 10)
array100 = repetitions(10, 100)
array1000 = repetitions(10, 1000)
array10000 = repetitions(10, 10000)

# Final output array containing the information from each array combined.
output = array10 + array100 + array1000 + array10000        # a 1D array with all the variables.
final_array = np.reshape(output, (-1, 3)).T                   # transposed and reshaped to produce a better visual array.

plt.plot(final_array[0], final_array[1])
plt.xscale('log')
plt.yscale('log')
plt.show()
