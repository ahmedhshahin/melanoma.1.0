import os
from scipy import misc
from skimage.segmentation import mark_boundaries, slic
from sklearn.metrics import jaccard_similarity_score
from sklearn.cluster import KMeans
from skimage.util import img_as_float, img_as_bool
from skimage.filters import median
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_adaptive
from skimage.exposure import equalize_adapthist
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from skimage.segmentation import quickshift

def calc_dice(x,y):
    return ((np.sum(x[y == 1]) * 2) / (np.sum(x) + np.sum(y))) * 100


# Read the images folder
path = "/home/ahmed/Melanoma/ISBI2016_ISIC_Part1_Training_Data"
img = "ISIC_0000000.jpg"
x_img = img_as_float(misc.imread(path + "/" + img))
y_img = img_as_float(misc.imread((path + "/" + img).replace("ISBI2016_ISIC_Part1_Training_Data","ISBI2016_ISIC_Part1_Training_GroundTruth").replace(".jpg","") + "_Segmentation.png"))

x_img[:,:,0] = median(x_img[:,:,0])
x_img[:,:,1] = median(x_img[:,:,1])
x_img[:,:,2] = median(x_img[:,:,2])


########################################################################

segment = slic(x_img, n_segments=2, compactness=10, max_iter=10, sigma=0, spacing=None, multichannel=True, convert2lab=True, enforce_connectivity=True, min_size_factor=0.1, max_size_factor=3, slic_zero=False)
misc.imshow(mark_boundaries(x_img,segment))

########################################################################


########################################################################

dice = calc_dice(y_img,segment)
jaccard = jaccard_similarity_score(y_img, segment) * 100

print("Dice accuracy: " ,dice)
print("Jaccard accuracy: " ,jaccard)