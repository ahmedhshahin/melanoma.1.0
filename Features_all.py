from scipy import misc
import numpy as np
import os
import matplotlib.pyplot as plt
training_path = "/home/ahmed/Melanoma/Testing_temp"
# ground_path = "/home/ahmed/Melanoma/ground_temp"

listing = os.listdir(training_path)
red_vector_foreground = []
green_vector_foreground = []
blue_vector_foreground = []

red_vector_background = []
green_vector_background = []
blue_vector_background = []

forepixelsred = []
forepixelsgreen = []
forepixelsblue = []


backpixelsred = []
backpixelsgreen = []
backpixelsblue = []

i = 1
for image in listing:
    x_image = misc.imread(training_path + "/" + image)
    gnd = training_path + "/" + image.replace(".jpg","")
    gnd = gnd.replace("Testing_temp", "ground_temp")
    y_image = misc.imread(gnd + "_Segmentation.png")
    indices_fore = np.nonzero(y_image.reshape([y_image.shape[0]*y_image.shape[1],1]))
    indices_back = np.nonzero((~y_image).reshape([y_image.shape[0] * y_image.shape[1], 1]))

    x_red = x_image[:,:,0].reshape([y_image.shape[0]*y_image.shape[1],1])
    x_red = x_red[indices_fore]
    forepixelsred[len(forepixelsred):len(forepixelsred)+100-1] = np.random.choice(x_red, 100, replace=False)

    x_green = x_image[:, :, 1].reshape([y_image.shape[0] * y_image.shape[1], 1])
    x_green = x_green[indices_fore]
    forepixelsgreen[len(forepixelsgreen):len(forepixelsgreen)+100-1] = np.random.choice(x_green, 100, replace=False)

    x_blue = x_image[:, :, 2].reshape([y_image.shape[0] * y_image.shape[1], 1])
    x_blue = x_blue[indices_fore]
    forepixelsblue[len(forepixelsblue):len(forepixelsblue)+100-1] = np.random.choice(x_blue, 100, replace=False)




    x_red = x_image[:,:,0].reshape([y_image.shape[0]*y_image.shape[1],1])
    x_red = x_red[indices_back]
    backpixelsred[len(backpixelsred):len(backpixelsred)+100-1] = np.random.choice(x_red, 100, replace=False)

    x_green = x_image[:, :, 1].reshape([y_image.shape[0] * y_image.shape[1], 1])
    x_green = x_green[indices_back]
    backpixelsgreen[len(backpixelsgreen):len(backpixelsgreen)+100-1] = np.random.choice(x_green, 100, replace=False)

    x_blue = x_image[:, :, 2].reshape([y_image.shape[0] * y_image.shape[1], 1])
    x_blue = x_blue[indices_back]
    backpixelsblue[len(backpixelsblue):len(backpixelsblue)+100-1] = np.random.choice(x_blue, 100, replace=False)

    #
    # green_foreground = np.multiply(x_image[:, :, 1], y_image)
    # blue_foreground = np.multiply(x_image[:, :, 2], y_image)
    #
    #
    #
    # red_vector_foreground = np.append(red_vector_foreground, red_foreground[nnz_indices_fore])
    # green_vector_foreground = np.append(green_vector_foreground, green_foreground[nnz_indices_fore])
    # blue_vector_foreground = np.append(blue_vector_foreground, blue_foreground[nnz_indices_fore])
    #
    # # Get R,G,B vectors in the background
    # red_background = np.multiply(x_image[:, :, 0], (~y_image))
    # green_background = np.multiply(x_image[:, :, 1], (~y_image))
    # blue_background = np.multiply(x_image[:, :, 2], (~y_image))
    #
    # nnz_indices_back = np.nonzero(red_background)
    # red_vector_background = np.append(red_vector_background ,red_background[nnz_indices_back])
    # green_vector_background = np.append(green_vector_background, green_background[nnz_indices_back])
    # blue_vector_background = np.append(blue_vector_background, blue_background[nnz_indices_back])
    print(i)
    i = i + 1



# plt.scatter(green_vector_foreground, blue_vector_foreground, color = 'blue')
# plt.show()

# print(len(blue_vector_background))
print(len(green_vector_foreground))

np.save("red_vector_foreground500-514", forepixelsred)
np.save("green_vector_foreground500-514", forepixelsgreen)
np.save("blue_vector_foreground500-514", forepixelsblue)
#
np.save("red_vector_background500-514", backpixelsred)
np.save("green_vector_background500-514", backpixelsgreen)
np.save("blue_vector_background500-514", backpixelsblue)