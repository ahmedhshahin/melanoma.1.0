import matplotlib.pyplot as plt
import numpy as np
from skimage import data
from skimage import filters
from skimage import exposure
from scipy import misc
from skimage.color import rgb2gray, gray2rgb
import os, sys


# wrong = []
# no = 0
# p = 0
temp = "/home/ahmed/melanoma_data/ISBI2016_ISIC_Part1_Training_Data"
#img = "ISIC_0000000.jpg"
x = sys.argv[1]
img = (7 - len(x)) * '0' + x
img = "ISIC_" + img + ".jpg"

# listing = sorted(os.listdir(temp))
# for img in listing:
i_s = []
j_s = []
i_sr = []
j_sr = []

x_img = misc.imread(temp + "/" + img)
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
for i in range(0,a):
    for j in range(0,b):
        if h[i,j] == False:
            break
        h[i, j] = False
#     for k in range(1, b - 1):
        if h[i,b - 1 - j] == False:
            break
        h[i,b - 1 - j] = False


black_out = gray2rgb(h) * x_img
# misc.imshow(black_out)

window = max(a,b) // 20
if not((np.mean(black_out[5:window, 5:window]) == 0) and (np.mean(black_out[-(window):-5, -(window):-5]) == 0) and (np.mean(black_out[-(window):-5, 5:window]) == 0) and (np.mean(black_out[5:window, -(window):-5]) == 0)):
    vproj = np.mean(black_out,axis=1)
    hproj = np.mean(black_out,axis=0)

    green_detector = np.divide(hproj[:,1],np.multiply(hproj[:,2],hproj[:,0]))
    blue_detector = np.divide(hproj[:,2],np.multiply(hproj[:,1],hproj[:,0]))
    red_detector = np.divide(hproj[:,0],np.multiply(hproj[:,1],hproj[:,2]))


    print("====================================")
    if not(int(0.25 * b) < np.where(green_detector == np.nanmax(green_detector))[0][0] < int(0.75 * b)):
        print("PLASTER")
    if not(int(0.25 * b) < np.where(blue_detector == np.nanmax(blue_detector))[0][0] < int(0.75 * b)):
        print("PLASTER")
    if not(int(0.25 * b) < np.where(red_detector == np.nanmax(red_detector))[0][0] < int(0.75 * b)):
        print("PLASTER")
plt.imshow(black_out)
plt.show()
