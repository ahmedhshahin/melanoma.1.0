from datasets import DGRoad
from dataset import Melanoma
from torchvision import transforms
from models import FullResulotionNet
from training import Training
import torch.nn as nn
import numpy as np
from sklearn.metrics import jaccard_similarity_score
from utils import val_metric, soft_dice_loss
import torch
# from unet import UNet

#torch.backends.cudnn.benchmark = True
transformations = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.51892472, 0.4431646,  0.40640972], [0.37666158, 0.33505249, 0.32253156])])

train_object = Training(model = FullResulotionNet, model_params = {'in_channels': 3, 'n_outputs': 1, 'n_filters' : 128}, criterion = soft_dice_loss, val_metric = val_metric, initial_lr = 5e-6, dataset = Melanoma, dataset_params = {'data_path': '/content/melanoma.1.0/dataset/overfit/', 'test_path' : '/content/melanoma.1.0/dataset/2016data/test/', 'transforms' : transformations, 'is_train' : True, 'is_test' : False}, batch_size_train = 2, train_steps_before_update = 1, batch_size_val = 1, cuda_device = 0, data_parallel = False, test_mode = False, overfit_mode = False)

train_object.train_model(20)
train_object.predict_test(save_dir = '/content/melanoma.1.0/dataset/2016data/test/', thresh=0.5)
