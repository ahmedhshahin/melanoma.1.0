import torch
from torch.autograd import Function
from itertools import repeat
import numpy as np
from sklearn.utils import class_weight


def val_metric(y, p, thresh):
    pred = np.zeros(p.shape, dtype = np.uint8)
    pred[np.where(p >= thresh)] = 1
    score = 0.0
    for i in range(y.shape[0]):
        score += calc_jaccard(y[i], pred[i])
    return score/y.shape[0]

def calc_jaccard(x,y):
    eps = 1e-9
    intersection = np.sum(x[y == 1])
    union = np.sum(x) + np.sum(y) - intersection + eps
    return (intersection / union) * 100

def soft_dice_loss(inputs, targets):
    num = targets.size(0)
    m1  = inputs.view(num,-1).float()
    m2  = targets.view(num,-1).float()
    # weights = class_weight.compute_class_weight('balanced', np.unique(targets), targets)
    intersection = (m1 * m2)
    eps = 1e-8
    score = 2. * (intersection.sum(1) + eps) / (m1.sum(1) + m2.sum(1) + eps)
    score = 1 - score.sum()/num
    return score
