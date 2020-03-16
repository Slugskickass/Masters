import numpy as np
import matplotlib.pyplot as plt

rng = np.random.random(100)

array = np.asarray(rng)
array2 = np.reshape(rng,(-1,10))

num = np.size(array2)
dat = array2.reshape(num)


print(dat)