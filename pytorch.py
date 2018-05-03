import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import numpy as np
import imageio
import glob
import matplotlib.pyplot as plt
from scipy import misc
from PIL import Image
from torch.utils.data import DataLoader

class melanomaData(Dataset):

	def __init__(self, data_path, channels, rows, cols, transforms=None, is_train=True):
		
		self.transforms = transforms
		self.data_path = data_path
		self.rows = rows
		self.cols = cols
		self.channels = channels

		img_folder = sorted(glob.glob(self.data_path + 'image/*'))
		label_folder = sorted(glob.glob(self.data_path + 'label/*'))

		imgs_all = np.ndarray((len(img_folder), rows, cols, channels), dtype=np.uint8)
		labels_all = np.ndarray((len(label_folder), rows, cols, 1), dtype=np.uint8)
		
		for i, img in enumerate(img_folder):
			img = Image.open(img)
			label = Image.open(label_folder[i])
			imgs_all[i] = img
			labels_all[i, ..., 0] = label

		labels_all[labels_all > 128] = 255
		labels_all[labels_all <= 128] = 0

		np.random.seed(231)

		val_idx = np.random.choice(imgs_all.shape[0], int(0.2 * imgs_all.shape[0]), replace=False)
		train_idx = np.array([idx for idx in range(imgs_all.shape[0]) if not idx in val_idx ])
		
		self.mean = np.mean(imgs_all[train_idx], (0,1,2))
		self.std = np.std(imgs_all[train_idx], (0,1,2))

		if is_train:
			self.imgs = imgs_all[train_idx]
			self.labels = labels_all[train_idx]
		else:
			self.imgs = imgs_all[val_idx]
			self.labels = labels_all[val_idx]

		self.N = self.imgs.shape[0]

		self.labels = self.labels / 255
		self.labels = np.transpose(self.labels, (0,3,1,2))
		# self.imgs = np.transpose(self.imgs, (0,3,1,2))
		print(self.imgs.shape)

	def __getitem__(self, index):
		
		img = self.imgs[index]
		print(img.shape)
		label = self.labels[index, 0, ...]
		if self.transforms is not None:
			img = self.transforms(img)

		return(img, label)

	def __len__(self):
		return self.N


if __name__ == '__main__':
    # Define transforms (1)
    transformations = transforms.Compose([transforms.Scale(32), transforms.ToTensor()])
    # Call the dataset
    dset_train = melanomaData('/home/ahmed/github/melanoma.1.0/dataset/2016data/train/', 3, 256, 256, transformations, is_train=True)
    # dset_val = melanomaData('/home/ahmed/github/melanoma.1.0/dataset/2016data/train/', 3, 256, 256, transformations, is_train=False)
    # train_dloader = DataLoader(dset_train, batch_size=4, shuffle=True, num_workers=4)
    # val_dloader = DataLoader(dset_val, batch_size=1, shuffle=False, num_workers=4)
    # for img, label in train_dloader:
    # 	print(img.size())
    # for img, label in val_dloader:
    # 	print(img.size())
    img, label = dset_train.__getitem__(2)
    print(img.shape)
    print(img.min())
    print(label.max())
    print(label.min())
    # print(np.unique(label))
    # print(dset.tr_mean)
	