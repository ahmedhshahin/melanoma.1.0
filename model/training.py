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
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import pairwise_distances
from numpy.linalg import inv, norm
import cv2
from time import time
from utils import *

class Training():
    def __init__(self, model, model_params, criterion, val_metric, initial_lr, dataset, dataset_params, batch_size_train, train_steps_before_update, batch_size_val, cuda_device, test_mode = False, overfit_mode = False, data_parallel = False):
        self.net = model(**model_params)
        self.net = nn.DataParallel(self.net)
        self.criterion = criterion
        self.val_metric = val_metric
        self.cuda_device = cuda_device
        self.net.cuda(self.cuda_device)
        self.dataset_params = dataset_params
        self.dataset = dataset
        self.test_mode = test_mode
        self.max_count = train_steps_before_update
        self.overfit_mode = overfit_mode
        self.data_parallel = data_parallel
        self.data_parallel_flag = True
        

        # if not test_mode:
        #     self.optimizer = torch.optim.Adam(self.net.parameters(), lr=initial_lr)
        #     self.batch_size_train = batch_size_train
        #     self.batch_size_val = batch_size_val
        train_params = dataset_params.copy()
        val_params = dataset_params.copy()
        test_params = dataset_params.copy()
        val_params['is_train'] = False
        test_params['is_test'] = True
        test_params['is_train'] = False
        #     train_params['is_train'] = True
        #     self.n = 12452/2
        #     #idx = np.arange(self.n)
        #     #np.random.shuffle(idx)
        #     idx = np.load('idx.npy')
        #     m = int(0.75*self.n)
        #     if self.overfit_mode:
        #         m = 100
        #     train_params['idx'] = idx[:m]
        #     train_dataset = dataset(**train_params)
        #     self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
        #                                            batch_size=batch_size_train,
        #                                            shuffle=True)
        #     if not self.overfit_mode:
        #         val_params = dataset_params
        #         val_params['is_train'] = False
        #         train_params['idx'] = idx[m:]
        #         val_dataset = dataset(**val_params)
        #         self.val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
        #                                                    batch_size=batch_size_val,
        #                                                    shuffle=False)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=initial_lr)
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val

        transformations = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.51892472, 0.4431646,  0.40640972], [0.37666158, 0.33505249, 0.32253156])])


        self.dataset_train = dataset(**train_params)
        self.dataset_val = dataset(**val_params)
        self.dataset_test = dataset(**test_params)

        self.train_loader = torch.utils.data.DataLoader(self.dataset_train, batch_size=batch_size_train, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(self.dataset_val, batch_size=batch_size_val, shuffle=False)
        self.test_loader = torch.utils.data.DataLoader(self.dataset_test, batch_size=1, shuffle=False)


        if self.overfit_mode:
            self.val_loader = self.train_loader
            self.test_loader = self.train_loader

        # for x, y, _ in self.train_loader:
        #     print(x.shape)
        #     print(y.shape)
        #     print(_)

        #     plt.imshow(np.transpose(x[0, ...], (1,2,0)))
        #     plt.imshow(y[0, ...])
        #     break

        self.train_loss_hist = []
        self.val_loss_hist = []
        
        self.best_val = 0.0
        # t = self.val_batches()
    def train_model(self, n_epochs):
        if self.data_parallel and not self.data_parallel_flag:
            self.net = nn.DataParallel(self.net)
            self.data_parallel_flag = True
        for e in range(n_epochs):
            print("Epoch {0} / {1} :".format(e, n_epochs))
            t1 = time()
            t_loss = self.train_batches()
            self.train_loss_hist.append(t_loss)
            # if self.overfit_mode:
            #     return t_loss
            v_loss = self.val_batches()
            self.val_loss_hist.append(v_loss)
            if self.best_val < np.max(v_loss):
                self.best_val = np.max(v_loss)
                self.save_checkpoint(e, self.best_val)
                print('saved')
            t2 = time()
            print(e, (t2-t1)/60.0, t_loss, v_loss) 
    
    def train_batches(self):
        self.net.train()
        epoch_loss = 0.0
        batch_loss = None
        count = 0
        for i, (images, labels, _) in enumerate(self.train_loader):  
            # Convert torch tensor to Variable
            images = Variable(images.cuda(self.cuda_device))
            labels = Variable(labels.cuda(self.cuda_device))

            # Forward + Backward + Optimize
            self.optimizer.zero_grad()  # zero the gradient buffer
            out5 = self.net(images)
            #aux_loss = self.criterion(out1, labels) + self.criterion(out2, labels) + self.criterion(out3, labels) + self.criterion(out4, labels)
            final_layer_loss = self.criterion(out5, labels)
            count += 1
            loss = final_layer_loss / self.max_count
            loss.backward()
            epoch_loss += final_layer_loss.item()
            if count == self.max_count:
                self.optimizer.step()
                count = 0
        return epoch_loss/(i+1)


    
    def val_batches(self):
        self.net.eval()
        # Test the Model
        # m = self.n - int(self.n*0.75)
        # pred = np.zeros((self.dataset_val.__len__(), 512 , 512))
        pred = np.zeros((self.val_loader.dataset.__len__(), 1 , 512 , 512))
        # y = np.zeros((self.dataset_val.__len__(), 512 , 512), dtype = np.uint8)
        y = []
        orgn_size = []
        cnt = 0
        for images, labels, size in self.val_loader:
            images = Variable(images, requires_grad=False).cuda(self.cuda_device)
            pred[cnt:cnt+images.size(0)] = self.net(images).cpu().data.numpy()#.reshape(4, -1)
            # y[cnt:cnt+4] = labels.cpu().numpy().astype(np.uint8)#.reshape(4, -1)
            y.append(labels.cpu().numpy().astype(np.uint8))
            orgn_size.append(size)
            cnt += images.size(0)
        print(pred.mean())
        print(pred.min())
        print(pred.max())
        max_score = 0
        for thresh in [0.5]:
            score = 0.0
            for p in range(pred.shape[0]):
                img = rev_padding(pred[p][0], orgn_size[p]) / 255.0
                print(img.mean())
                print(img.min())
                print(img.max())
                temp = np.zeros(img.shape)
                temp[img >= thresh] = 1
                label = rev_padding(y[p][0], orgn_size[p])
                plt.imsave("/content/l{0}.png".format(p), img)
                score += calc_jaccard(temp, label)
                if max_score < score:
                    max_score = score
        mean_loss = max_score / pred.shape[0]

        # mean_loss = [self.val_metric(y, pred, thresh) for thresh in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]]
        # mean_loss = [self.val_metric(y, pred, thresh) for thresh in [0.5]]
        return mean_loss
        
    def predict_test(self, save_dir, thresh, batch_size = 1):
        self.net.eval()
        # test_params = self.dataset_params
        # test_params['is_train'] = False
        # test_params['is_test'] = True
        # test_params['data_dir'] = data_dir
        # test_params['idx'] = None
        # test_dataset = self.dataset(**test_params)
        test_loader = self.test_loader
        img_names = [name.split('_')[0] for name in test_dataset.img_names]
        for i, img in enumerate(test_loader):
            img = Variable(img, requires_grad=False).cuda(self.cuda_device)
            prob = self.net(img).cpu().data.numpy().reshape((512,512))
            prob *= 255
            prob[np.where(prob >= thresh*255)] = 255
            prob[np.where(prob < thresh*255)] = 0
            prob = prob.astype(np.uint8)
            # msk = np.zeros((prob.shape[0], prob.shape[1], 3), dtype = np.uint8)
            # msk[:,:,0] = msk[:,:,1] = msk[:,:,2] = prob
            msk = prob
            cv2.imwrite(os.path.join(save_dir, img_names[i]+'_mask.png'), msk)
        
    def modify_lr(self, new_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
    
    def save_checkpoint(self, epoch, val_iou):
        state = {
            'epoch': epoch,
            'state_dict': self.net.state_dict(),
            'val_iou': val_iou,
            'optimizer' : self.optimizer.state_dict()
        }
        filename = 'best_model.pth.tar'
        torch.save(state, filename)
    
    def load_checkpoint(self, filename, initial_lr, load_net1 = False, load_optimizer = True):
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch'] + 1
        self.best_val = checkpoint['val_iou']
        if load_net1:
            d = checkpoint['state_dict']
            d = {k.replace('module.', ''):v for k,v in d.items()}
            self.net.net1.load_state_dict(checkpoint['state_dict'])
        else:
            self.net.load_state_dict(checkpoint['state_dict'])
        self.net.cuda(self.cuda_device)
        if not self.test_mode and load_optimizer:
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=initial_lr)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda(self.cuda_device)
