from torchvision import transforms
from torch.utils.data.dataset import Dataset
import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy import misc
from PIL import Image
from torch.utils.data import DataLoader
import os

class Melanoma(Dataset):

	def __init__(self, data_path, test_path, transforms=None, is_train=True, is_test=False):
		
		self.transforms = transforms
		self.is_test = is_test

		img_folder = np.array(sorted(glob.glob(data_path + 'image/*')))
		label_folder = np.array(sorted(glob.glob(data_path + 'label/*')))
		test_folder = np.array(sorted(glob.glob(test_path + 'image/*')))
		if is_test:
			# self.img_names = [f for f in os.listdir(test_path) if (os.path.isfile(os.path.join(test_path, f)))]
			self.img_names = test_folder



		n_total = len(img_folder)

		np.random.seed(231)

		val_idx = np.random.choice(n_total, int(0.2 * n_total), replace=False)
		train_idx = np.array([idx for idx in range(n_total) if not idx in val_idx ])

		if is_train:
			self.imgs = img_folder[train_idx]
			self.labels = label_folder[train_idx]
		elif is_test:
			self.imgs = test_folder
		else:
			self.imgs = img_folder[val_idx]
			self.labels = label_folder[val_idx]

		self.N = len(self.imgs)

# mean is [0.51892472 0.4431646  0.40640972]
# mean is [0.37666158 0.33505249 0.32253156]

	def __getitem__(self, index):

		if self.is_test:
			img = Image.open(self.imgs[index])
			t = transforms.Compose([transforms.ToTensor()])
			img = t[img]
			label = None
		else:
			img = Image.open(self.imgs[index])
			label = np.array(Image.open(self.labels[index]))
			
			label[label > 128] = 255 
			label[label <= 128] = 0
			label = label / 255 
			# label = np.transpose(label, (2,0,1))
		
			if self.transforms is not None:
				img = self.transforms(img)

		return(img, label)

	def __len__(self):
		return self.N


if __name__ == '__main__':
    transformations = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.51892472, 0.4431646,  0.40640972], [0.37666158, 0.33505249, 0.32253156])])
    dset_train = melanomaData('/home/ahmed/github/melanoma.1.0/dataset/2016data/train/', transformations, is_train=True)
    dset_val = melanomaData('/home/ahmed/github/melanoma.1.0/dataset/2016data/train/', transformations, is_train=False)
    train_dloader = DataLoader(dset_train, batch_size=4, shuffle=True, num_workers=4)
    val_dloader = DataLoader(dset_val, batch_size=1, shuffle=True, num_workers=4)

 