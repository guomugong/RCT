import torch
import torch.nn as nn
from model.unet_parts import ConvBNReLU


class VGG16BN(nn.Module):
    def __init__(self, in_channel):
        super(VGG16BN, self).__init__()
        self.conv1_1 = ConvBNReLU(in_channel, 64)
        self.conv1_2 = ConvBNReLU(64, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = ConvBNReLU(64, 128)
        self.conv2_2 = ConvBNReLU(128, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = ConvBNReLU(128, 256)
        self.conv3_2 = ConvBNReLU(256, 256)
        self.conv3_3 = ConvBNReLU(256, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = ConvBNReLU(256, 512)
        self.conv4_2 = ConvBNReLU(512, 512)
        self.conv4_3 = ConvBNReLU(512, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = ConvBNReLU(512, 512)
        self.conv5_2 = ConvBNReLU(512, 512)
        self.conv5_3 = ConvBNReLU(512, 512)

    def forward(self, input):
        conv1_1 = self.conv1_1(input)
        conv1_2 = self.conv1_2(conv1_1)
        pool1 = self.pool1(conv1_2)

        conv2_1 = self.conv2_1(pool1)
        conv2_2 = self.conv2_2(conv2_1)
        pool2 = self.pool2(conv2_2)

        conv3_1 = self.conv3_1(pool2)
        conv3_2 = self.conv3_2(conv3_1)
        conv3_3 = self.conv3_3(conv3_2)
        pool3 = self.pool3(conv3_3)

        conv4_1 = self.conv4_1(pool3)
        conv4_2 = self.conv4_2(conv4_1)
        conv4_3 = self.conv4_3(conv4_2)
        pool4 = self.pool4(conv4_3)

        conv5_1 = self.conv5_1(pool4)
        conv5_2 = self.conv5_2(conv5_1)
        conv5_3 = self.conv5_3(conv5_2)

        return conv1_2, conv2_2, conv3_3, conv4_3, conv5_3


def vgg16(in_ch):
    model = VGG16BN(in_ch)
    #print("loading pretrained/5stages_vgg16_bn-6c64b313.pth")
    #model.load_state_dict(torch.load("/home/guo/torch_project/lesion/old/v2/pretrained/5stages_vgg16_bn-6c64b313.pth"), strict=False)
    return model
