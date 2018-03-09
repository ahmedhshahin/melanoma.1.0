"""Legacy objectives module.

Only kept for backwards API compatibility.
"""
from __future__ import absolute_import
from .losses import *

def dice(self, y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return -1 * (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_acc(y_true, y_pred, smooth=1):
    return - dice(y_true, y_pred)
