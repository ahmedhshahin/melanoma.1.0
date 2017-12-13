import os
from scipy import misc
import numpy as np
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d

def get_seed_diag(diag):
    diag_length = len(diag)
    x1 = np.arange(0,int(diag_length)/2,1)
    x2 = np.arange(int((diag_length)/2),0,-1)
    x3 = np.concatenate((x1, x2),axis=0)
    h = x3 * diag
    grad = np.gradient(h)
    low = np.where(grad == min(grad))[0][0]
    high = np.where(grad[low:-1] == max(grad[low:-1]))[0][0] + low
    seed = int((high + low) / 2)
    return seed


path = "/home/ahmed/Melanoma/ISBI2016_ISIC_Part1_Training_Data"
img = "ISIC_0000000.jpg"
x_img = misc.imread(path + "/" + img)
[a,b] = x_img[:,:,0].shape


diag = gaussian_filter1d(np.average(np.diagonal(x_img[5:a-5, 5:b-5]),0).astype(int),11)
seed1 = get_seed_diag(diag)
opp_diag = gaussian_filter1d(np.average(np.diagonal(np.fliplr(x_img[5:a-5, 5:b-5])),0).astype(int),7)
seed2 = get_seed_diag(opp_diag)

brightness_1 = np.average(x_img[seed1,seed1,:])
brightness_2 = np.average(x_img[seed2,seed2,:])

if brightness_1 < brightness_2:
    seed = seed1
else:
    seed = seed2

print(seed1)
print(seed2)
print(seed)
plt.plot(diag)
plt.show()
