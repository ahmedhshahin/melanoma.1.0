import os
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import img_as_bool

# this script creates batches from the dataset
# batch size: 224 * 224 * 3
# we save the batch and its ground truth in two separate folders "batches" , "batches_ground


path = "/home/ahmed/melanoma/ISBI2016_ISIC_Part1_Training_Data"
listing = sorted(os.listdir(path))
k = 0
j = 0
y = []
while (k < 900):
    cur_img_id = np.random.randint(0,900 - k)
    cur_img = listing[cur_img_id]
    x_img = misc.imread(path + "/" + cur_img)
    y_img = img_as_bool(misc.imread(path.replace("ISBI2016_ISIC_Part1_Training_Data","ISBI2016_ISIC_Part1_Training_GroundTruth") + "/" + cur_img.replace(".jpg","_Segmentation.png")))
    # now we have chosen an image
    # get fore and back vectors for the image

    fore_idx = np.where(y_img == True)
    back_idx = np.where(y_img == False)
    # in the fore, pick 55 random elements
    i_f = 0
    i_b = 0
    a0 = fore_idx[0]
    a1 = fore_idx[1]
    b0 = back_idx[0]
    b1 = back_idx[1]
    
    while ((i_f < 55) and (len(a0) > 0)):
        k_fore = np.random.randint(0,len(a0))
        x_f = a0[k_fore]
        y_f = a1[k_fore]
        if (x_f >= 112) and (y_f >= 112) and (x_f+112 < y_img.shape[0]) and (y_f+112 < y_img.shape[1]):
            misc.imsave('/home/ahmed/melanoma/batches/{2}_fore_{0}_batch_{1}.jpg'.format(cur_img.replace(".jpg","").replace("ISIC_",""),i_f, j),x_img[x_f-112: x_f+112, y_f-112:y_f+112,:])
            misc.imsave('/home/ahmed/melanoma/batches_ground/{2}_fore_{0}_batch_{1}_mask.jpg'.format(cur_img.replace(".jpg","").replace("ISIC_",""),i_f,j),y_img[x_f-112: x_f+112, y_f-112:y_f+112])
            u = x_img[x_f-112: x_f+112, y_f-112:y_f+112,:]
            if (u.shape[0] != 224) or (u.shape[1] != 224):
                print("ERROR")
            i_f += 1
            j += 1
            y.append(1)

        a0 = np.delete(a0,k_fore)
        a1 = np.delete(a1,k_fore)
#         print(len(a0))

    # the same thing with the back
    while ((i_b < 55) and (len(b0) > 0)):
        k_back = np.random.randint(0,len(b0))
        x_b = b0[k_back]
        y_b = b1[k_back]
        if (x_b >= 112) and (y_b >= 112) and (x_b+112 < y_img.shape[0]) and (y_b+112 < y_img.shape[1]):
            misc.imsave('/home/ahmed/melanoma/batches/{2}_back_{0}_batch_{1}.jpg'.format(cur_img.replace(".jpg","").replace("ISIC_",""),i_b,j),x_img[x_b-112: x_b+112, y_b-112:y_b+112,:])
            misc.imsave('/home/ahmed/melanoma/batches_ground/{2}_back_{0}_batch_{1}_mask.jpg'.format(cur_img.replace(".jpg","").replace("ISIC_",""),i_b,j),y_img[x_b-112: x_b+112, y_b-112:y_b+112])
            n = x_img[x_b-112: x_b+112, y_b-112:y_b+112,:]
            if (n.shape[0] != 224) or (n.shape[1] != 224):
                print("ERROR")
            i_b += 1
            j += 1
            y.append(0)
        b0 = np.delete(b0,k_back)
        b1 = np.delete(b1,k_back)
#         print(len(b0))
    print(k)
    k += 1
