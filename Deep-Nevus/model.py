from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=2, dilation=2)


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super(ConvRelu, self).__init__()
        self.conv = nn.Sequential(conv3x3(in_, out), conv3x3(out, out))
        self.activation = nn.Sequential(nn.BatchNorm2d(out), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(DecoderBlock, self).__init__()

        self.block = nn.Sequential(
            ConvRelu(in_channels, middle_channels),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)
class DecoderBlockWithPSP(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(DecoderBlockWithPSP, self).__init__()

        self.block = nn.Sequential(
            ConvRelu(in_channels, middle_channels),
            _PyramidPoolingModule(middle_channels, middle_channels // 4, (1,2,3,6)),
            nn.ConvTranspose2d(middle_channels * 2, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class _PyramidPoolingModule(nn.Module):
    def __init__(self, in_dim, reduction_dim, setting):
        super(_PyramidPoolingModule, self).__init__()
        self.features = []
        for s in setting:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim, momentum=.95),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.upsample(f(x), x_size[2:], mode='bilinear'))
        out = torch.cat(out, 1)
        return out


class DeepNevus(nn.Module):
    def __init__(self, num_filters=32, pretrained=False):
        """
        :param num_filters:
        :param pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with VGG11
        """
        super(DeepNevus, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = models.vgg11(pretrained=pretrained).features

        self.relu = self.encoder[1]
        self.conv1 = nn.Sequential(self.encoder[0], nn.BatchNorm2d(64))
        self.conv2 = nn.Sequential(self.encoder[3], nn.BatchNorm2d(128))
        self.conv3s = nn.Sequential(self.encoder[6], nn.BatchNorm2d(256))
        self.conv3 = nn.Sequential(self.encoder[8], nn.BatchNorm2d(256))
        self.conv4s = nn.Sequential(self.encoder[11], nn.BatchNorm2d(512))
        self.conv4 = nn.Sequential(self.encoder[13], nn.BatchNorm2d(512))
        self.conv5s = nn.Sequential(self.encoder[16], nn.BatchNorm2d(512))
        self.conv5 = nn.Sequential(self.encoder[18], nn.BatchNorm2d(512))

        self.center = DecoderBlockWithPSP(num_filters * 8 * 2, num_filters * 8 * 4, num_filters * 8 * 2)
        self.dec5 = DecoderBlockWithPSP(num_filters * (16 + 16), num_filters * 8 * 2, num_filters * 8)
        self.dec4 = DecoderBlockWithPSP(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 4)
        self.dec3 = DecoderBlock(num_filters * (8 + 4), num_filters * 4 * 2, num_filters * 2)
        self.dec2 = DecoderBlock(num_filters * (4 + 2), num_filters * 2 * 2, num_filters)
        self.dec1 = ConvRelu(num_filters * (2 + 1), num_filters)

        # self.wide = nn.Sequential(*[WideFoVPooling(64) for i in range(1)])
        self.ppm1 = _PyramidPoolingModule(64, 16, (1, 2, 3, 6))
        self.ppm2 = _PyramidPoolingModule(128, 32, (1, 2, 3, 6))
        self.ppm3 = _PyramidPoolingModule(256, 64, (1, 2, 3, 6))
        self.ppm4 = _PyramidPoolingModule(512, 128, (1, 2, 3, 6))
        self.ppm5 = _PyramidPoolingModule(512, 512, (1, 2, 3, 6))

        self.final = nn.Sequential(nn.Conv2d(num_filters, 1, kernel_size=1))

    def forward(self, x):
        conv1 = self.relu(self.conv1(x))    # 64 * 192 * 256
        # conv1_sk = self.ppm1(conv1)

        conv2 = self.relu(self.conv2(self.pool(conv1)))     # 128 * 96 * 128
        # conv2_sk = self.ppm2(conv2)

        conv3s = self.relu(self.conv3s(self.pool(conv2))) 
        conv3 = self.relu(self.conv3(conv3s))         # 256 * 48 * 64
        # conv3_sk = self.ppm3(conv3)

        conv4s = self.relu(self.conv4s(self.pool(conv3)))
        conv4 = self.relu(self.conv4(conv4s)) # 512 * 24 * 32
        # conv4_sk = self.ppm4(conv4)

        conv5s = self.relu(self.conv5s(self.pool(conv4)))
        conv5 = self.relu(self.conv5(conv5s))   # 512 * 12 * 16
        # conv5_sk = self.ppm5(conv5)

        center = self.center(self.pool(conv5))
        # print(center.size())

        dec5 = self.dec5(torch.cat([center, conv5], 1))         # 1024 * 32 * 32
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))        # 64 * 64
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))       # 128 * 128
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))       # 256 * 256
        # print(dec2.size())
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        return F.sigmoid(self.final(dec1))
