from dataset import Melanoma
from torchvision import transforms
from model import DeepNevus
from training import Training
import torch.nn as nn
import numpy as np
from sklearn.metrics import jaccard_similarity_score
from utils import val_metric, soft_dice_loss, CrossEntropyLoss2d
import torch
# from unet_model import UNet

#torch.backends.cudnn.benchmark = True
# transformations = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.51892472, 0.4431646,  0.40640972], [0.37666158, 0.33505249, 0.32253156])])

train_object = Training(model = DeepNevus, model_params = {'in_channels': 3, 'n_outputs': 1, 'n_filters' : 128}, criterion = soft_dice_loss, val_metric = val_metric, initial_lr = 5e-5, dataset = Melanoma, dataset_params = {'data_path': '/media/ubi-comp/sda2/Hassaan/fold4', 'test_path' : '/media/ubi-comp/sda2/Hassaan/fold4/val4', 'sizes_array' : '/media/ubi-comp/sda2/Hassaan/fold4/sizes_4.npy', 'img_dim' : 256, 'is_train' : True, 'is_test' : False}, batch_size_train = 16, train_steps_before_update = 1, batch_size_val = 1, cuda_device = 1, data_parallel = False, test_mode = False, overfit_mode = False)

# train_object = Training(model = UNet, model_params = {'in_channels': 3, 'n_outputs': 1, 'n_filters' : 128}, criterion = soft_dice_loss, val_metric = val_metric, initial_lr = 5e-6, dataset = Melanoma, dataset_params = {'data_path': '/home/karim/Documents/hassan/HASSAAN/overfit/', 'test_path' : '/home/karim/Documents/hassan/HASSAAN/res', 'sizes_array' : '/home/karim/Documents/hassan/HASSAAN/sizes_18.npy', 'img_dim' : 512, 'is_train' : True, 'is_test' : False}, batch_size_train = 2, train_steps_before_update = 4, batch_size_val = 1, cuda_device = 0, data_parallel = False, test_mode = False, overfit_mode = False)


# train_object.train_model(200)
train_object.load_checkpoint('F4_0.837.pth.tar', 5e-6)
train_object.train_model(100)
# train_object.predict_test(save_dir = '/home/karim/Documents/hassan/HASSAAN/tmp_res/', thresh=0.5)
# train_object.predict_test(save_dir = '/content/melanoma.1.0/dataset/2016data/test/', thresh=0.5)
