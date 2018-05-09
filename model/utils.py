import torch
from torch.autograd import Function
from itertools import repeat
import numpy as np
from sklearn.utils import class_weight
import PIL
import math
from scipy import misc
from PIL import Image


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

def padding(img, new_dim):
    if type(img) is PIL.JpegImagePlugin.JpegImageFile:
        img = np.array(img)
    a, b = img.shape[:2]
    ratio = b*1.0 / a
    if len(img.shape) == 3:
        if ratio < 1.35:
            # pad b
            new_b = int(math.ceil(b * (1.35 / ratio)))
            if (new_b % 2 == 1): new_b += 1
            output = np.zeros((a, new_b, 3))
            output[:, (new_b-b)//2 : (new_b+b)//2 , :] = img
            output_sq = np.zeros((new_b, new_b, 3))
            output_sq[(new_b-a)//2 : (new_b+a)//2, :, :] = output
        else:
            new_a = int(math.ceil(a * (ratio / 1.35)))
            if (new_a % 2 == 1): new_a += 1
            output = np.zeros((new_a, b, 3))
            output[(new_a-a)//2 : (new_a+a)//2, :, :] = img
            if new_a > b:
                output_sq = np.zeros((new_a, new_a, 3))
                output_sq[:, (new_a-b)//2 : (new_a+b)//2, :] = output
            else:
                output_sq = np.zeros((b,b,3))
                output_sq[(b-new_a)//2 : (b+new_a)//2, :, :] = output
        output_sq = misc.imresize(output_sq, (new_dim,new_dim,3))
        output_sq = Image.fromarray(output_sq)
    else:
        if ratio < 1.35:
            # pad b
            new_b = int(math.ceil(b * (1.35 / ratio)))
            if (new_b % 2 == 1): new_b += 1
            output = np.zeros((a, new_b))
            output[:, (new_b-b)//2 : (new_b+b)//2] = img
            output_sq = np.zeros((new_b, new_b))
            output_sq[(new_b-a)//2 : (new_b+a)//2, :] = output
        else:
            new_a = int(math.ceil(a * (ratio / 1.35)))
            if (new_a % 2 == 1): new_a += 1
            output = np.zeros((new_a, b))
            output[(new_a-a)//2 : (new_a+a)//2, :] = img
            if new_a > b:
                output_sq = np.zeros((new_a, new_a))
                output_sq[(new_a-b)//2 : (new_a+b)//2, :] = output
            else:
                output_sq = np.zeros((b,b))
                output_sq[(b-new_a)//2 : (b+new_a)//2, :] = output
        output_sq = misc.imresize(output_sq, (new_dim,new_dim))
    return output_sq


def rev_padding(img, orgn_size):
    a, b = orgn_size[:2]
    i, j = img.shape[:2]
    ratio = b / a
    if len(img.shape) == 3:
        if ratio < 1.35:
            new_b = math.ceil(b * (1.35/ratio))
            if (new_b % 2 == 1): new_b += 1
            img_sq = misc.imresize(img, (new_b, new_b, 3))
            output = img_sq[(new_b-a)//2:(new_b+a)//2, :, :]
            origin = output[:, (new_b-b)//2:(new_b+b)//2, :]
        else:
            new_a = math.ceil(a * (ratio / 1.35))
            if (new_a % 2 == 1): new_a += 1
            if new_a > b:
                img_sq = misc.imresize(img, (new_a, new_a, 3))
                output = img_sq[:, (new_a-b)//2:(new_a+b)//2, :]
                origin = output[(new_a-a)//2:(new_a+a)//2, :, :]
            else:
                img_sq = misc.imresize(img, (b, b, 3))
                output = img_sq[(b-new_a)//2:(b+new_a)//2, :, :]
                origin = output[(new_a-a)//2:(new_a+a)//2, :, :]
    else:
        if ratio < 1.35:
            new_b = math.ceil(b * (1.35/ratio))
            if (new_b % 2 == 1): new_b += 1
            img_sq = misc.imresize(img, (new_b, new_b))
            output = img_sq[(new_b-a)//2:(new_b+a)//2, :]
            origin = output[:, (new_b-b)//2:(new_b+b)//2]
        else:
            new_a = math.ceil(a * (ratio / 1.35))
            if (new_a % 2 == 1): new_a += 1
            if new_a > b:
                img_sq = misc.imresize(img, (new_a, new_a))
                output = img_sq[:, (new_a-b)//2:(new_a+b)//2]
                origin = output[(new_a-a)//2:(new_a+a)//2, :]
            else:
                img_sq = misc.imresize(img, (b, b))
                print(a, b, new_a, img_sq)
                output = img_sq[(b-new_a)//2:(b+new_a)//2, :]
                origin = output[(new_a-a)//2:(new_a+a)//2, :]
    return origin