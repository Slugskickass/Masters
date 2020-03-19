import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

mean_std = np.load("/Users/RajSeehra/University/Masters/Semester 2/Git Upload Folder/Return1/Raj/PTC bits/PTCdata.npy")
mean_std = mean_std.T

x_data = []
std = []

for i in range(0,len(mean_std)):
    if 10 < mean_std[i, 0] < 1000:
        x_data.append(mean_std[i, 0])
        std.append((mean_std[i, 1]))

y_data = np.square(std)

slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)

print(slope, intercept)
plt.plot(slope, intercept, "*")
plt.scatter(x_data, y_data)
plt.title("PTC (cut out mean <10 or >1000)")
plt.text(50, 25, 'Slope= '+ str(slope) + ', Intercept= ' + str(intercept))
plt.xlabel('Mean')
plt.ylabel('Standard Deviation Squared')
plt.show()