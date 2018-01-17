%matplotlib 
import matplotlib.pyplot as plt
import numpy as np
from skimage import data
from skimage import filters
from skimage import exposure
from scipy import misc
from skimage.color import rgb2gray, gray2rgb


i_s = []
j_s = []
i_sr = []
j_sr = []

x_img = (misc.imread("/home/ahmed/melanoma_data/ISBI2016_ISIC_Part1_Training_Data/ISIC_0011126.jpg"))
x_gray = rgb2gray(x_img)
a, b = x_gray.shape
val = filters.threshold_otsu(x_gray)

mask = x_gray > val

for i in range(1,a - 1):
    for j in range(1,b - 1):
        if mask[i,j] == True:
            i_s.append(i)
            j_s.append(j)
            break
#     for k in range(1, b - 1):
        if mask[i,b - 1 - j] == True:
            i_sr.append(i)
            j_sr.append(b - 1 - j)
            break

h = np.ones((a,b), dtype=bool)
h[i_s, j_s] = False
h[i_sr, j_sr] = False
for i in range(1,a - 1):
    for j in range(1,b - 1):
        if h[i,j] == False:
            break
        h[i, j] = False
#     for k in range(1, b - 1):
        if h[i,b - 1 - j] == False:
            break
        h[i,b - 1 - j] = False


plt.subplot(121)
plt.imshow(x_img, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.subplot(122)
plt.imshow(h, cmap='gray', interpolation='nearest')
plt.axis('off')
