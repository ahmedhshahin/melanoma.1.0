%matplotlib
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import math, os
from skimage.color import rgb2gray, gray2rgb
from skimage.filters import threshold_otsu


path = "/home/ahmed/melanoma_data/ISBI2016_ISIC_Part1_Training_Data/"
listing = sorted(os.listdir(path))
def pad_with_zeros_b(img, target_size):
    i, j = target_size[:2]
    a, b = img.shape[:2]

    out = np.zeros(target_size)
    out[:, (j - b) //2 : (j + b) //2, :] = img

    output = np.zeros((j, j, 3))
    output[(j - a) // 2 : (j + a) // 2, :, :] = out
    return output

def pad_with_zeros_a(img, target_size):
    i, j = target_size[:2]
    a, b = img.shape[:2]

    out = np.zeros(target_size)
    out[(i - a) //2 : (i + a) //2, :, :] = img

    output = np.zeros((j, j, 3))
    output[(j - i) // 2 : (j + i) // 2, :, :] = out
    return output


def eliminate_black(img):
    i_s = []
    j_s = []
    i_sr = []
    j_sr = []

    x_gray = rgb2gray(img)
    a, b = x_gray.shape
    val = threshold_otsu(x_gray)

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
    return(gray2rgb(h) * img)
    
    
for img in listing:
    x_img = eliminate_black(misc.imread(path + img))
    x_img = misc.imread(path + img)
    size = x_img.shape
    a, b = size[:2]
    x_img[0:5, :, :] = 0
    x_img[-6:, :, :] = 0
    x_img[:, 0:5, :] = 0
    x_img[:, -6:, :] = 0
    
    ratio = b/a
#     print("The original aspect ratio is: ", ratio)

    if ratio < 1.35:
        new_b = math.ceil(b * (1.35/ratio))
        if (new_b % 2 == 1): new_b += 1
#         print("The intermediate aspect ratio is: ", new_b/a)
        target_size = [a, new_b, 3]
        out = pad_with_zeros_b(x_img, target_size)

    else:
        new_a = math.ceil(a * (ratio/1.35))
        if (new_a % 2 == 1): new_a += 1
#         print("The intermediate aspect ratio is: ", b/new_a)
        target_size = [new_a, b, 3]
        out = pad_with_zeros_a(x_img, target_size)
    output = misc.imresize(out, (256,256,3))
    misc.imsave("/home/ahmed/melanoma_data/new/" + img.replace("ISIC_", "").replace(".jpg","") + "padding.jpg" , output)
