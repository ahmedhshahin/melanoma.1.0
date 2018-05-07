import os
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from torch.utils.data import Dataset
from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import pairwise_distances
from numpy.linalg import inv, norm
import cv2

mean = np.zeros((3,))
std = np.zeros((3,))

class Potsdam(Dataset):
    """ Class for manipulating Dataset"""
    def __init__(self, data_dir='../isprs/potsdam/'):
        super(Potsdam, self).__init__()
        pass
    
    def read_image(self):
        pass
        
    def read_dem(self):
        pass
    
    def calc_stats(self):
        pass
    
    def __len__(self):
        pass
        
    def __getitem__(self, index):
        pass

class Vaihingen(Dataset):
    """ Class for manipulating Dataset"""
    def __init__(self, data_dir='../isprs/vaihingen/'):
        super(Vaihingen, self).__init__()
        pass
    
    def read_image(self):
        pass
        
    def read_dem(self):
        pass
    
    def calc_stats(self):
        pass
    def __len__(self):
        pass
        
    def __getitem__(self, index):
        pass

class DGRoad(Dataset):
    """ Class for manipulating Dataset"""
    def __init__(self, is_train, idx, data_dir, img_dim = 512, step = 256, crop_size = 256, augment = False, is_test = False, transform = None):
        super(DGRoad, self).__init__()
        self.img_dim = img_dim
        self.data_dir = data_dir
        self.transform = transform
        self.is_test = is_test
        if self.is_test:
            self.img_names = [f for f in os.listdir(self.data_dir) if (os.path.isfile(os.path.join(self.data_dir, f)) and 'sat' in f)]
        else:
            self.img_names = np.array(sorted([f for f in os.listdir(self.data_dir) if (os.path.isfile(os.path.join(self.data_dir, f)) and 'sat' in f)]))[idx].tolist()
            self.msk_names = np.array(sorted([f for f in os.listdir(self.data_dir) if (os.path.isfile(os.path.join(self.data_dir, f)) and 'mask' in f)]))[idx].tolist()
        self.is_train = is_train
        self.step = step
        self.crop_size = crop_size
        self.augment = augment
        self.read_images()
        if not self.is_test:
            self.read_masks()
        #if is_train:
        #    self.calc_stats()
    
    def read_images(self):
        self.X = np.zeros((len(self.img_names), self.img_dim, self.img_dim, 3), dtype = np.uint8)
        for index in range(len(self.img_names)):
            img_name = self.img_names[index]
            x = cv2.imread(os.path.join(self.data_dir, img_name))#.transpose(2,0,1)
            x = cv2.resize(x, (self.img_dim, self.img_dim))
            #print x.dtype
            self.X[index] = x.copy()
            del x
            #break
        
    def read_masks(self):
        self.Y = np.zeros((len(self.msk_names), self.img_dim, self.img_dim), dtype = np.uint8)
        for index in range(len(self.msk_names)):
            mask_name = self.msk_names[index]
            y = cv2.cvtColor(cv2.imread(os.path.join(self.data_dir, mask_name)), cv2.COLOR_BGR2GRAY)
            y = y.reshape(y.shape[0], y.shape[1])
            y[np.where(y < 128)] = 0
            y[np.where(y >= 128)] = 1
            y = cv2.resize(y, (self.img_dim, self.img_dim))
            self.Y[index] = y.copy()
            del y
            #break

    def read_img_and_mask(self, index):
        img_name = self.img_names[index]
        x = cv2.imread(os.path.join(self.data_dir, img_name))#.transpose(2,0,1)
        x = cv2.resize(x, (self.img_dim, self.img_dim)) 
        mask_name = self.msk_names[index]
        y = cv2.cvtColor(cv2.imread(os.path.join(self.data_dir, mask_name)), cv2.COLOR_BGR2GRAY)
        y[np.where(y < 128)] = 0
        y[np.where(y >= 128)] = 1
        y = cv2.resize(y, (self.img_dim, self.img_dim))
        return (x, y)
        
    def crop(self, index):
        img_index = index % len(self.img_names)
        crop_pos = index / len(self.img_names)
        row = crop_pos / (self.img_dim/self.step)
        col = crop_pos % (self.img_dim/self.step)
        #print index, img_index, crop_pos, row, col
        return (self.X[img_index, row*self.step:row*self.step+self.crop_size, col*self.step:col*self.step+self.crop_size], self.Y[img_index, row*self.step:row*self.step+self.crop_size, col*self.step:col*self.step+self.crop_size])
        
    def random_crop(self, index):
        img_index = index % len(self.img_names)
        row_start = random.randint(0, 1024-self.crop_size)
        col_start = random.randint(0, 1024-self.crop_size)
        return (self.X[img_index, row_start:row_start+self.crop_size, col_start:col_start+self.crop_size], self.Y[img_index, row_start:row_start+self.crop_size, col_start:col_start+self.crop_size])
    
    def calc_stats(self):
        global mean, std
        for c in range(3):
            mean[c] = self.X[:,:,:,c].mean()
            std[c] = self.X[:,:,:,c].std()
            
    def normalize_one(self, x):
        x = x.astype(np.float)
        x /= 255.0
        return x

    def normalize_scaler(self, x):
        x = x.astype(np.float)
        for c in range(3):
            x[:,:,c] = (x[:,:,c] - mean[c]) / std[c]
        return x
    
    def __len__(self):
        return len(self.img_names)*(self.img_dim/self.step)**2
    
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
        if self.is_train:
            x, y = self.random_crop(index)
            prob = random.uniform(0.0,1.0)
            if prob > 0.5:
                x,y = self.flip(x,y)
                x,y = self.rotate(x,y)
        elif self.is_test:
            x = self.X[index]
        else:
            x, y = self.crop(index)
        x = self.normalize_one(x)
        x = x.transpose(2, 0, 1)
        x = torch.from_numpy(x).type(torch.FloatTensor)
        if self.is_test:
            return x
        y = torch.from_numpy(y).type(torch.FloatTensor)
        if not (self.transform is None):
            x, y = self.transform(x, y)
        return (x, y)
