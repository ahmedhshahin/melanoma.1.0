# Model implementation in PyTorch
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import matplotlib

class SoftAttention(nn.Module):
    def __init__(self, in_channels, stride):
        super(SoftAttention, self).__init__()
        if stride == 1:
            self.layers = nn.Sequential(nn.Conv2d(in_channels, in_channels, groups = in_channels, kernel_size=3, padding = 1), nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True))
        else:
            self.layers = nn.Sequential(nn.Conv2d(in_channels, in_channels, groups = in_channels, kernel_size=stride+1, stride = stride, padding = stride/2), nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True), nn.Upsample(scale_factor=stride))
    
    def forward(self, x):
        return self.layers(x)

class SoftAttentionPooling(nn.Module):
    def __init__(self, stride, in_channels):
        super(SoftAttentionPooling, self).__init__()
        self.layers = [nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding = 1), nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True))]
        if stride > 1:
            self.layers = [nn.MaxPool2d(stride, stride)] + self.layers + [nn.Upsample(scale_factor=stride, mode = 'bilinear')]
        self.layers = nn.Sequential(*self.layers)
    
    def forward(self, x):
        return self.layers(x)

class WideFoV(nn.Module):
    def __init__(self, in_channels):
        super(WideFoV, self).__init__()
        self.conv1d_1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True))
        self.soft_attention_s1 = SoftAttention(in_channels, 1)
        self.soft_attention_s2 = SoftAttention(in_channels, 2)
        self.soft_attention_s4 = SoftAttention(in_channels, 4)
        self.soft_attention_s8 = SoftAttention(in_channels, 8)
        self.conv1d_2 = nn.Sequential(nn.Conv2d(in_channels*4, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True))
    
    def forward(self, x):
        x = self.conv1d_1(x)
        x_s1 = self.soft_attention_s1(x)
        x_s2 = self.soft_attention_s2(x)
        x_s4 = self.soft_attention_s4(x)
        x_s8 = self.soft_attention_s8(x)
        x = torch.cat([x_s1, x_s2, x_s4, x_s8], dim = 1)
        x = self.conv1d_2(x)
        return x

class WideFoVPooling(nn.Module):
    def __init__(self, in_channels, residual = True):
        super(WideFoVPooling, self).__init__()
        self.conv2d = nn.Sequential(nn.Conv2d(in_channels, in_channels, dilation=2, kernel_size=3, padding = 2), nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True))
        self.soft_attention_s1 = SoftAttentionPooling(1, in_channels)
        self.soft_attention_s2 = SoftAttentionPooling(2, in_channels)
        self.soft_attention_s4 = SoftAttentionPooling(4, in_channels)
        #self.soft_attention_s8 = SoftAttentionPooling(8, in_channels)
        self.conv1d = nn.Sequential(nn.Conv2d(in_channels*3, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True))
        self.residual = residual
    
    def forward(self, x_input):
        x = self.conv2d(x_input)
        x_s1 = self.soft_attention_s1(x)
        x_s2 = self.soft_attention_s2(x)
        x_s4 = self.soft_attention_s4(x)
        #x_s8 = self.soft_attention_s8(x)
        x = torch.cat([x_s1, x_s2, x_s4], dim = 1)
        x = self.conv1d(x)
        if self.residual:
            x = x + x_input
        return x


class FullResulotionNet(nn.Module):
    def __init__(self, in_channels = 3, n_layers = 3, n_filters = 64, n_outputs = 1):
        super(FullResulotionNet, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, n_filters, dilation=2, kernel_size=3, padding = 2), nn.BatchNorm2d(n_filters), nn.ReLU(inplace=True))
        self.max_pool = nn.MaxPool2d(2, 2)
        self.wfov_1_10 = nn.Sequential(*[WideFoVPooling(n_filters) for i in range(10)])
        self.upsample = nn.Upsample(scale_factor=2, mode = 'bilinear')
        self.classif = nn.Sequential(nn.Conv2d(2*n_filters, n_outputs, kernel_size=1), nn.Sigmoid())
    
    def forward(self, x):
        x = self.conv(x)
        x_pool = self.max_pool(x)
        print(x_pool.size())
        out10 = self.wfov_1_10(x_pool)
        print(out10.size())
        out10 = self.upsample(out10)
        print(out10.size())
        out11 = torch.cat([x, out10], dim = 1)
        print(out11.size())
        pred = self.classif(out11)
        print(pred.size())
        return pred
