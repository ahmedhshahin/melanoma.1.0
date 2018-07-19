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
import random

class Melanoma(Dataset):

    def __init__(self, data_path, test_path, sizes_array, img_dim, is_train=True, is_test=False):
        
        self.transforms = transforms
        self.is_test = is_test
        self.is_train = is_train

        img_folder = np.array(sorted(glob.glob(data_path + '/train_aug4/*.jpg')))
        label_folder = np.array(sorted(glob.glob(data_path + '/train_aug4/*.png')))
        val_folder = np.array(sorted(glob.glob(data_path + '/val4/*.jpg')))
        val_label_folder = np.array(sorted(glob.glob(data_path + '/val_label4/*.png')))
        test_folder = np.array(sorted(glob.glob(test_path + '/*.jpg')))
        self.val_sizes = np.load(sizes_array)
        print("Training: ", len(img_folder))
        print("Validation: ", len(val_folder))

        # n_total = len(img_folder)

        # np.random.seed(231)
        # val_idx = np.random.choice(n_total, int(0.2 * n_total), replace=False)
        # train_idx = np.array([idx for idx in range(n_total) if not idx in val_idx ])

        train_img_names = img_folder
        train_label_names = label_folder
        val_img_names = val_folder
        val_label_names = val_label_folder
        # self.val_sizes = sizes[val_idx]

        if is_test:
            self.img_names = test_folder
            self.imgs = np.zeros((len(test_folder), 192, img_dim, 3), dtype=np.uint8)
            for i in range(len(test_folder)):
                self.imgs[i] = misc.imread(test_folder[i])

        elif is_train:
            self.imgs = np.zeros((len(train_img_names), 192, img_dim, 3), dtype=np.uint8)
            self.labels = np.zeros((len(train_img_names), 192, img_dim), dtype=np.uint8)
            for i in range(len(train_img_names)):
                self.imgs[i] = misc.imread(train_img_names[i])
                self.labels[i] = misc.imread(train_label_names[i])
            self.labels[self.labels < 128] = 0
            self.labels[self.labels >= 128] = 1

        else:
            self.imgs = np.zeros((len(val_img_names), 192, img_dim, 3), dtype=np.uint8)
            self.labels = np.zeros((len(val_img_names), 192, img_dim), dtype=np.uint8)
            for i in range(len(val_img_names)):
                self.imgs[i] = misc.imread(val_img_names[i])
                self.labels[i] = misc.imread(val_label_names[i])
            self.labels[self.labels < 128] = 0
            self.labels[self.labels >= 128] = 1

        self.N = len(self.imgs)

        # self.is_test = is_test
        # self.is_train = is_train

        # train_img_names = np.array(sorted(glob.glob(train_path + '/*.jpg')))
        # train_label_names = np.array(sorted(glob.glob(train_path + '/*.jpg')))

        # val_img_names = np.array(sorted(glob.glob(val_path + '/*.jpg')))
        # val_label_names = np.array(sorted(glob.glob(val_path + '/*.png')))

        # test_folder = np.array(sorted(glob.glob(test_path + '/*.jpg')))
        # self.val_sizes = np.load(sizes_array)

        # n_total = len(img_folder)

        # np.random.seed(231)
        # val_idx = np.random.choice(n_total, int(0.2 * n_total), replace=False)
        # train_idx = np.array([idx for idx in range(n_total) if not idx in val_idx ])

        # train_img_names = img_folder[train_idx]
        # train_label_names = label_folder[train_idx]
        # val_img_names = img_folder[val_idx]
        # val_label_names = label_folder[val_idx]
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
        ang = random.randint(-180, 180)
        x = self.rotateImage(x, ang)
        y = self.rotateImage(y, ang)
        return (x, y)

    def __getitem__(self, index):

        if self.is_test:
            img = self.imgs[index]
            img = img.astype(np.float)
            img /= 255
            img = np.transpose(img, (2,0,1))
            img = torch.from_numpy(img).type(torch.FloatTensor)
            return img

        elif self.is_train:
            img = self.imgs[index]
            label = self.labels[index]
            img = img.astype(np.float)
            img /= 255
            # prob = random.uniform(0.0,1.0)
            # if prob > 0.5:
            #     img, label = self.flip(img, label)
            #     img, label = self.rotate(img, label)
            img = np.transpose(img, (2,0,1))
            img = torch.from_numpy(img).type(torch.FloatTensor)
            label = torch.from_numpy(label).type(torch.FloatTensor)
            return (img, label)

        else:
            img = self.imgs[index]
            label = self.labels[index]
            size = self.val_sizes[index]
            img = img.astype(np.float)
            img /= 255
            img = np.transpose(img, (2,0,1))
            img = torch.from_numpy(img).type(torch.FloatTensor)
            label = torch.from_numpy(label).type(torch.FloatTensor)
            return(img, label, size)

    def __len__(self):
        return self.N


if __name__ == '__main__':
    dset_train = Melanoma('/home/ahmed/melanoma_new/training2018_512/', '/home/ahmed/melanoma_data/2016 Test/', sizes_array="/home/ahmed/melanoma_new/sizes_18.npy" , img_dim=512, is_train=True)
    dset_val = Melanoma('/home/ahmed/melanoma_new/training2018_512/', 'home/ahmed/Desktop/', sizes_array="/home/ahmed/melanoma_new/sizes_18.npy", img_dim=512,is_train=False)
    # train_dloader = DataLoader(dset_train, batch_size=4, shuffle=True, num_workers=4)
    # val_dloader = DataLoader(dset_val, batch_size=1, shuffle=True, num_workers=4)
    # for img, label in train_dloader:
    #     print(img.shape)
    #     print(label.shape)
    #     print(img.mean())
    #     print(img.min())
    #     print(img.max())
    #     print(np.unique(label))
    #     i = np.transpose(img[0], (1,2,0))
    #     misc.imshow(i)
    #     misc.imshow(label[0]*255)
    #     print(size)
    #     break