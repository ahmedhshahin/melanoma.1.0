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

	def __init__(self, data_path, transforms=None, is_train=True):
		
		self.transforms = transforms

		img_folder = np.array(sorted(glob.glob(data_path + 'image/*'))[:1])
		label_folder = np.array(sorted(glob.glob(data_path + 'label/*'))[:1])

		n_total = len(img_folder)

		np.random.seed(231)

		val_idx = np.random.choice(n_total, int(0.2 * n_total), replace=False)
		train_idx = np.array([idx for idx in range(n_total) if not idx in val_idx ])

		if is_train:
			self.imgs = img_folder[train_idx]
			self.labels = label_folder[train_idx]
		else:
			self.imgs = img_folder[val_idx]
			self.labels = label_folder[val_idx]

		self.N = len(self.imgs)

# mean is [0.51892472 0.4431646  0.40640972]
# std is [0.37666158 0.33505249 0.32253156]

	def __getitem__(self, index):
		
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
    # Define transforms (1)
    transformations = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.51892472, 0.4431646,  0.40640972], [0.37666158, 0.33505249, 0.32253156])])
    # Call the dataset
    dset_train = melanomaData('/home/ahmed/github/melanoma.1.0/dataset/2016data/train/', transformations, is_train=True)
    dset_val = melanomaData('/home/ahmed/github/melanoma.1.0/dataset/2016data/train/', transformations, is_train=False)
    train_dloader = DataLoader(dset_train, batch_size=4, shuffle=True, num_workers=4)
    val_dloader = DataLoader(dset_val, batch_size=1, shuffle=True, num_workers=4)
    # for im , l in train_dloader:
    #     print(np.mean(im.numpy()[0], (1,2)))
    #     print(np.std(im.numpy()[0], (1,2)))
    #     break
    # img, label = dset_train.__getitem__(5)
    # # print(img.max())
    # print(label.min())
    # print(label.max())

    