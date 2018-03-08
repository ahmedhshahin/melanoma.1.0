# Melanoma.1.0
## Introduction:
Cancer of the skin is by far the most common of all cancers. Melanoma accounts for only about 1% of skin cancers but causes a large majority of skin cancer deaths. One person dies of skin melanoma every hour (every 54 minutes). Early detection of skin cancer melanoma is a key for the treatment success. Automatic image-based naevus detection, identification, and classification is mandatory to advance home-based skin melanoma early detection. Also, making this technology affordable using the smartphone camera will have an impact on the early detection. Once, patient has a positive result for melanoma detection test from the mobile application, he should be admitted to a more detailed and hospital-based melanoma confirmation and grading test. This is done using other imaging and biophysical methods noninvasively e.g. Confocal Laser Microscopy (CLM) and Optical coherence tomography (OCT). Both are used to assess the subdermal melanoma geometric information to plan the therapy noninvasively.

## Project Description:
  - Develop image analysis algorithm that enables the automated detection, identification of Melanoma based on some visual features of the naevus; symmetry, border, color, diameter, and dynamic changes, those features have been incorporated in the automatic algorithm using classical machine learning techniques which did not introduce good results.
  
## Objective:
We are investigating different deep neural network architectures to achieve the accurate melanoma identification since the classical machine learning techniques did not yield accurate results.

## Done Work:
* Preprocessing:
  - Resizing all images to standard input size (256 * 256), because most of the models used required unique input size.
  - Detection and Segmentation of black regions of the image.
  - Dtermining the most informative layer of the RGB layers, in order to decide whether it is better to train with the 3 layers, one of them, or the grayscale images.

* We are testing our dataset on some deep learning models to get some intuitions about the performance of the various archetictures on our data, that will help us to propose our algorithm.

## Technologies Used:
  - Python
  - Numpy
  - Scipy
  - Matplotlib
  - Sickit-learn
  - Sickit-image
  - Tensorflow
  - Keras
  
  
## Future Work:
  - Using the metadata of patients, we want to classify each melanoma image into melanoma, nevus or seborrheic keratosis.
  - We aim to combine both imaging modalities; CLM and OCT; and use both information for a detailed and comprehensive analysis. That includes developing a pretreatment automatic segmentation of the melanoma volumetric information to facilitate and standardize the treatment delivery planning. Automation and standardizing the planning step will reduce the manual induced error. Considering information coming from different imaging modalities highlights the expected added value of deep neural networks in fusing medical image based information.
