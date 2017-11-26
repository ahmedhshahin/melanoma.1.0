# this script loads the vector files and plot the boxplots

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

blue_back = np.load("blue_vector_background500-514.npy")
blue_fore = np.load("blue_vector_foreground500-514.npy")
green_back = np.load("green_vector_background500-514.npy")
green_fore = np.load("green_vector_foreground500-514.npy")
red_back = np.load("red_vector_background500-514.npy")
red_fore = np.load("red_vector_foreground500-514.npy")

red = [red_back, red_fore]
green = [green_back, green_fore]
blue = [blue_back, blue_fore]
f, axs = plt.subplots(1,3)
axs[0].boxplot(red)

axs[1].boxplot(green)

axs[2].boxplot(blue)
plt.show()

# plt.scatter(blue_back, red_back, marker = '^' , color = 'blue')
# plt.scatter(blue_fore, red_fore, marker = 'o' , color = 'red', alpha= 0.05)
#
# red_patch = mpatches.Patch(color= 'red', label='Foreground')
# blue_patch = mpatches.Patch(color= 'blue', label='Background')
#
# plt.legend(handles=[red_patch, blue_patch])
# # red_axis = np.arange(0,256,1)
# # green_axis = np.arange(0,256,1)
# plt.show()
#
# x = [50]
# y = [50]
#
