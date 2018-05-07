from datasets import DGRoad
from models import FullResulotionNet
from training import Training
import torch.nn as nn
import numpy as np
from sklearn.metrics import jaccard_similarity_score
from utils import val_metric, soft_dice_loss
import torch
from time import time
#torch.backends.cudnn.benchmark = True

train_object = Training(model = FullResulotionNet, model_params = {'in_channels': 3, 'n_outputs': 1, 'n_filters' : 64}, criterion = soft_dice_loss, val_metric = val_metric, initial_lr = 1e-4, dataset = DGRoad, dataset_params = {'data_dir': '../road_data/train/', 'img_dim':1024, 'crop_size': 512, 'step': 512, 'augment':True}, batch_size_train = 4, train_steps_before_update = 4, batch_size_val = 1, cuda_device = 0, data_parallel = True, test_mode = False)

train_object.load_checkpoint('model_iou_614.pth.tar', 5e-5)
train_object.modify_lr(1e-5)
train_object.train_on_val_batches()
#t1 = time()
#arr = train_object.val_batches_with_aug()
#print (time() - t1)/60, arr
train_object.predict_test(data_dir = '../road_data/valid/', save_dir = '../road_data/sub/', thresh = 0.5)
