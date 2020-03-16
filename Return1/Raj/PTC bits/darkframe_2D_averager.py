import numpy as np
import matplotlib.pyplot as plt
import PTC_Samurai as sam

file = sam.loadtiffs("/Users/RajSeehra/University/Masters/Semester 2/Teaching_python-master/Week 2/Data/640.tif")

array = np.zeros((np.shape(file)[0], np.shape(file)[1]))

for i in range(0, np.shape(file)[2]):
    array = array + file[:,:,i]

array = array/(np.shape(file)[2])

plt.imshow(array)
plt.show()


#sam.savetiffs("darkframes.tif", array)