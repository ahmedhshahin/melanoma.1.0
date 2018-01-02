import os
from scipy import misc
import numpy as np
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
from skimage.morphology import closing,disk
import scipy as sp


def get_seed_diag(diag):
    diag_length = len(diag)
    x1 = np.arange(0,int(diag_length)/2,1)
    x2 = np.arange(int((diag_length)/2),0,-1)
    x3 = np.concatenate((x1, x2),axis=0)
    h = x3 * diag
    grad = np.gradient(h)
    low = np.where(grad == min(grad))[0][0]
    try:
        high = np.where(grad[low:-1] == max(grad[low:-1]))[0][0] + low
    except:
        high = low
    seed = int((high + low) / 2)
    return seed

def cart2polar(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta

def polar2cart(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def index_coords(data, origin=None):
    """Creates x & y coords for the indicies in a numpy array "data".
    "origin" defaults to the center of the image. Specify origin=(0,0)
    to set the origin to the lower left corner of the image."""
    ny, nx = data.shape[:2]
    if origin is None:
        origin_x, origin_y = nx // 2, ny // 2
    else:
        origin_x, origin_y = origin
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x -= origin_x
    y -= origin_y
    return x, y

def reproject_image_into_polar(data, origin):
    ny, nx = data.shape[:2]

    # Determine that the min and max r and theta coords will be...
    x, y = index_coords(data, origin=origin)
    r, theta = cart2polar(x, y)

    # Make a regular (in polar space) grid based on the min and max r & theta
    r_i = np.linspace(r.min(), r.max(), nx)
    theta_i = np.linspace(theta.min(), theta.max(), ny)
    theta_grid, r_grid = np.meshgrid(theta_i, r_i)

    # Project the r and theta grid back into pixel coordinates
    xi, yi = polar2cart(r_grid, theta_grid)
    xi += origin[0] # We need to shift the origin back to 
    yi += origin[1] # back to the lower-left corner...
    xi, yi = xi.flatten(), yi.flatten()
    coords = np.vstack((xi, yi)) # (map_coordinates requires a 2xn array)

    # Reproject each band individually and the restack
    # (uses less memory than reprojection the 3-dimensional array in one step)
    bands = []
    for band in data.T:
        zi = sp.ndimage.map_coordinates(band, coords, order=1)
        bands.append(zi.reshape((nx, ny)))
    output = np.dstack(bands)
    return output, r_i, theta_i


path = "/home/ahmed/melanoma_data/ISBI2016_ISIC_Part1_Training_Data"
img = "ISIC_0000026.jpg"

listing = sorted(os.listdir(path))
# for img in listing:
x_img = misc.imread(path + "/" + img)
# y_img = misc.imread(path.replace("ISBI2016_ISIC_Part1_Training_Data", "ISBI2016_ISIC_Part1_Training_GroundTruth") + "/" + img.replace(".jpg","_Segmentation.png"))

[a,b] = x_img[:,:,0].shape


diag = gaussian_filter1d(np.diagonal(x_img[5:a-5,5:b-5,2]),11)
seed1 = get_seed_diag(diag)

opp_diag = gaussian_filter1d(np.diagonal(np.fliplr(x_img[5:a-5,5:b-5,2])),11)
seed2 = get_seed_diag(opp_diag)

brightness_1 = np.average(x_img[seed1,seed1,:])
brightness_2 = np.average(x_img[seed2,seed2,:])

if brightness_1 < brightness_2:
    seed = seed1
else:
    seed = seed2

# if y_img[seed,seed] == 255:
#     correct += 1
# else:
#     wrong.append(img)
# print(i)
# i = i + 1
i = 0

polar_grid, r, theta = reproject_image_into_polar(x_img, (seed,seed))

thresholds = np.zeros((7, 2))
for angle in range(-3,4):
    xu, yu = polar2cart(r, angle)
    xu += seed
    yu = seed - yu
    try:
        th_y = np.where(yu < 0)[0][0]
    except:
        th_y = len(yu) - 1
    try:
        th_x = np.where(xu < 0)[0][0]
    except:
        th_x = len(xu) - 1
    ynew = yu[0 : min(th_x, th_y)]
    xnew = xu[0 : min(th_x, th_y)]
    ynew = ynew.astype(int)
    xnew = xnew.astype(int)
    xnew = xnew[xnew < b]
    ynew = ynew[ynew < a]
    xnew = xnew[0: min(len(xnew), len(ynew))]
    ynew = ynew[0: min(len(xnew), len(ynew))]

#     print(min(xnew), min(ynew))

    to_plot = gaussian_filter1d(x_img[ynew, xnew, 2],25)
    # plt.plot(to_plot)
    # plt.show()
    m = max(to_plot) - min(to_plot)
    temp = int(0.4 * (max(to_plot) - min(to_plot)) + min(to_plot))

#     print(temp)

    y = to_plot[0:np.where(to_plot == max(to_plot))[0][0]]
    x = np.arange(len(y))
    f = sp.interpolate.interp1d(y, x)
    threshold_id = int(f(temp))

#     print(threshold_id)

    thresh_x = xnew[threshold_id]
    thresh_y = ynew[threshold_id]

#     print(thresh_x, thresh_y)

    thresh = int(cart2polar(thresh_x - seed, thresh_y - seed)[0] * 1.5)

#     print(thresh)

    if (seed - polar2cart(thresh, angle)[1] < 0):
        thresholds[i, 0] = 0
    else:
        thresholds[i, 0] = seed - polar2cart(thresh, angle)[1]

    if (seed + polar2cart(thresh, angle)[0] < 0):
        thresholds[i, 1] = 0
    else:
        thresholds[i, 1] = seed + polar2cart(thresh, angle)[0]


    # thresholds[i, 1] = seed + polar2cart(thresh, angle)[0]
    i = i + 1
    # thresholds[i, 0] = thresh_y
    # thresholds[i, 1] = thresh_x

    thresholds = thresholds.astype(int)

out = (x_img[min(thresholds[:,0]):max(thresholds[:,0]),min(thresholds[:,1]):max(thresholds[:,1])])
plt.subplot(2, 2, 1)
plt.imshow(x_img)
plt.axis('off')
plt.subplot(2, 2, 2)
plt.imshow(out)
plt.axis('off')
plt.show()
