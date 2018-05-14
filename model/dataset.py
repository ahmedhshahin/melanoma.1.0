from torchvision import transforms
from torch.utils.data.dataset import Dataset
import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy import misc
from PIL import Image
from torch.utils.data import DataLoader
import os
from utils import *
import cv2

class Melanoma(Dataset):

	def __init__(self, data_path, test_path, sizes_array, img_dim, is_train=True, is_test=False):
		
		self.transforms = transforms
		self.is_test = is_test
		self.is_train = is_train

		img_folder = np.array(sorted(glob.glob(data_path + '/*.jpg')))
		label_folder = np.array(sorted(glob.glob(data_path + '/*.png')))
		test_folder = np.array(sorted(glob.glob(test_path + '/*.jpg')))
		sizes = np.load(sizes_array)

		n_total = len(img_folder)

		np.random.seed(231)
		val_idx = np.random.choice(n_total, int(0.2 * n_total), replace=False)
		train_idx = np.array([idx for idx in range(n_total) if not idx in val_idx ])

		train_img_names = img_folder[train_idx]
		train_label_names = label_folder[train_idx]
		val_img_names = img_folder[val_idx]
		val_label_names = label_folder[val_idx]
		self.val_sizes = sizes[val_idx]

		if is_test:
			self.img_names = test_folder
			self.imgs = np.zeros((len(test_folder), img_dim, img_dim, 3), dtype=np.uint8)
			for i in range(len(test_folder)):
				self.imgs[i] = misc.imread(test_folder[i])
			self.imgs = np.transpose(self.imgs, (0,3,1,2))

		elif is_train:
			self.imgs = np.zeros((len(train_img_names), img_dim, img_dim, 3), dtype=np.uint8)
			self.labels = np.zeros((len(train_img_names), img_dim, img_dim), dtype=np.uint8)
			for i in range(len(train_img_names)):
				self.imgs[i] = misc.imread(train_img_names[i])
				self.labels[i] = misc.imread(train_label_names[i])
			self.imgs = np.transpose(self.imgs, (0,3,1,2))
			self.labels[self.labels < 128] = 0
			self.labels[self.labels >= 128] = 1

		else:
			self.imgs = np.zeros((len(val_img_names), img_dim, img_dim, 3), dtype=np.uint8)
			self.labels = np.zeros((len(val_img_names), img_dim, img_dim), dtype=np.uint8)
			for i in range(len(val_img_names)):
				self.imgs[i] = misc.imread(val_img_names[i])
				self.labels[i] = misc.imread(val_label_names[i])
			self.imgs = np.transpose(self.imgs, (0,3,1,2))
			self.labels[self.labels < 128] = 0
			self.labels[self.labels >= 128] = 1

		self.N = len(self.imgs)
# mean is [0.51892472 0.4431646  0.40640972]
# mean is [0.37666158 0.33505249 0.32253156]


	def rotateImage(self, image, angle):
        image_center = (image.shape[0] / 2, image.shape[1] / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[0:2])
        return result
        
    def flip(self, x, y):
        op = random.randint(-1,1)
        x = cv2.flip(x, op)
        y = cv2.flip(y, op)
        return (x, y)
    
    def rotate(self, x, y):
        ang = random.randint(-10, 10)
        x = self.rotateImage(x, ang)
        y = self.rotateImage(y, ang)
        return (x, y)

	def __getitem__(self, index):

		if self.is_test:
			img = self.imgs[index]
			img = img.astype(np.float)
			img /= 255
			img[0,:,:] -= 0.51892472
			img[0,:,:] /= 0.37666158
			img[1,:,:] -= 0.4434646
			img[1,:,:] /= 0.33505249
			img[2,:,:] -= 0.40640972
			img[2,:,:] /= 0.32253156
			return img

		elif self.is_train:
			img = self.imgs[index]
			label = self.labels[index]
			img = img.astype(np.float)
			img /= 255
			img[0,:,:] -= 0.51892472
			img[0,:,:] /= 0.37666158
			img[1,:,:] -= 0.4434646
			img[1,:,:] /= 0.33505249
			img[2,:,:] -= 0.40640972
			img[2,:,:] /= 0.32253156
			prob = random.uniform(0.0,1.0)
            if prob > 0.5:
                img, label = self.flip(img, label)
                img, label = self.rotate(img, label)
			return (img, label)

		else:
			img = self.imgs[index]
			label = self.labels[index]
			size = self.val_sizes[index]
			img = img.astype(np.float)
			img /= 255
			img[0,:,:] -= 0.51892472
			img[0,:,:] /= 0.37666158
			img[1,:,:] -= 0.4434646
			img[1,:,:] /= 0.33505249
			img[2,:,:] -= 0.40640972
			img[2,:,:] /= 0.32253156
			return(img, label, size)

	def __len__(self):
		return self.N


if __name__ == '__main__':
    dset_train = Melanoma('/home/ahmed/melanoma_new/training2018_512/', '/home/ahmed/melanoma_data/2016 Test/', sizes_array="/home/ahmed/melanoma_new/sizes_18.npy" , img_dim=512, transforms=None, is_train=True)
    dset_val = Melanoma('/home/ahmed/melanoma_new/training2018_512/', 'home/ahmed/Desktop/', sizes_array="/home/ahmed/melanoma_new/sizes_18.npy", img_dim=512,transforms=None, is_train=False)
    train_dloader = DataLoader(dset_train, batch_size=4, shuffle=True, num_workers=4)
    val_dloader = DataLoader(dset_val, batch_size=1, shuffle=True, num_workers=4)
    for img, label, size in val_dloader:
    	print(img.shape)
    	print(label.shape)
    	print(img.mean())
    	print(img.min())
    	print(img.max())
    	print(np.unique(label))
    	i = np.transpose(img[0], (1,2,0))
    	misc.imshow(i)
    	misc.imshow(label[0]*255)
    	print(size)
    	break
 