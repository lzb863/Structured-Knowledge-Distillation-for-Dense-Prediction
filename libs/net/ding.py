# encoding: utf-8
from pytorchcv.model_provider import get_model as ptcv_get_model
import torch.nn as nn
from libs.net.sync_batchnorm import SynchronizedBatchNorm2d

bn_mom = 0.0003
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, atrous=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = SynchronizedBatchNorm2d(planes, momentum=bn_mom)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1 * atrous, dilation=atrous, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = SynchronizedBatchNorm2d(planes, momentum=bn_mom)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        # self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.bn3 = SynchronizedBatchNorm2d(planes * self.expansion, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class get_resnet18_05_backbone(nn.Module):
    def __init__(self, name, mode, pretrained=False):
        super(get_resnet18_05_backbone, self).__init__()
        self.name = name
        self.pretranined = pretrained
        self.layers = []
        self.conv1_layer = None
        self.layers = [2,2,2,2]
        self.stride_list = [2, 2, 1]
        self.atrous = [1, 2, 1]
        self.os = 16
        self.block = Bottleneck
        self.mode = mode
        self.outdim_16 = 256
        self.outdim_4 = 32
        self.make_net()

    def make_net(self):
        net_temp = ptcv_get_model(self.name, pretrained=self.pretranined)
        self.conv = net_temp.features[0].conv # 32 32 128 128
        self.Max_pool = net_temp.features[0].pool
        self.stage1 = net_temp.features[1] #32 32 64 64
        self.stage2 = net_temp.features[2] #32 64 32 32
        self.stage3 = net_temp.features[3] #32 128 16 16
        if self.mode == 'atrous':
            self.stage4 = self._make_layer(self.block, 128, 64, self.layers[3], stride=self.stride_list[2],
                                       atrous=[item * 16 // self.os for item in self.atrous])
        elif self.mode == 'deconv':
            self.stage4 = self._make_layer_deconv(self.block, 128, 64)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1, atrous=None):
        downsample = None
        if atrous == None:
            atrous = [1] * blocks
        elif isinstance(atrous, int):
            atrous_list = [atrous] * blocks
            atrous = atrous_list
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                SynchronizedBatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )

        layers = []
        layers.append(block(inplanes, planes, stride=stride, atrous=atrous[0], downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(planes * block.expansion, planes, stride=1, atrous=atrous[i]))

        return nn.Sequential(*layers)

    def _make_layer_deconv(self, block, inplanes, planes):
        downsample = None
        atrous = [1, 1]
        blocks = 2
        stride=1
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, padding=1,
                          kernel_size=3, stride=2, bias=False),
                SynchronizedBatchNorm2d(planes, momentum=bn_mom),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(planes, planes* block.expansion, padding=0,
                          kernel_size=2, stride=2, bias=False),
            )

        layers = []
        layers.append(downsample)
        for i in range(0, blocks):
            layers.append(block(planes* block.expansion, planes, stride=1, atrous=atrous[i]))

        return nn.Sequential(*layers)

    def get_layers(self):
        return self.layers

    def get_conv1(self):
        return self.conv1_layer

    def forward(self, x):
        self.layers = []
        x = self.conv(x)
        # print('conv1:',x.shape)
        self.conv1_layer = x
        x = self.Max_pool(x)
        x = self.stage1(x)
        # print('stage1:', x.shape)
        self.layers.append(x)
        x = self.stage2(x)
        # print('stage2:', x.shape)
        self.layers.append(x)
        x = self.stage3(x)
        # print('stage3:', x.shape)
        self.layers.append(x)
        x = self.stage4(x)
        # print('stage4:', x.shape)
        self.layers.append(x)

        return x


class get_Darktiny_backbone(nn.Module):
    def __init__(self, name, pretrained=False):
        super(get_Darktiny_backbone, self).__init__()
        self.name = name
        self.pretranined = pretrained
        self.layers = []
        self.conv1_layer = None
        self.layers = [2, 2, 2, 2]
        self.stride_list = [2, 2, 1]
        self.atrous = [1, 2, 1]
        self.os = 16
        self.block = Bottleneck
        self.make_net()

    def make_net(self):
        net_temp = ptcv_get_model(self.name, pretrained=self.pretranined)
        self.stage1 = net_temp.features.stage1  # 32, 16, 128, 128
        self.stage2 = net_temp.features.stage2  # 32, 32, 64, 64
        self.stage3 = net_temp.features.stage3  # 32, 128, 32, 32
        self.stage4 = net_temp.features.stage4  # 32, 256, 16, 16
        self.stage5 = net_temp.features.stage5  # 32, 128, 16, 16
        self.stage6 = self._make_layer(self.block, 128, 64, self.layers[-1], stride=self.stride_list[2],
                                       atrous=[item * 16 // self.os for item in self.atrous]) # 32, 256, 16, 16

    def _make_layer(self, block, inplanes, planes, blocks, stride=1, atrous=None):
        downsample = None
        if atrous == None:
            atrous = [1] * blocks
        elif isinstance(atrous, int):
            atrous_list = [atrous] * blocks
            atrous = atrous_list
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                SynchronizedBatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )

        layers = []
        layers.append(block(inplanes, planes, stride=stride, atrous=atrous[0], downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(planes * block.expansion, planes, stride=1, atrous=atrous[i]))

        return nn.Sequential(*layers)

    def get_layers(self):
        return self.layers

    def get_conv1(self):
        return self.conv1_layer

    def forward(self, x):
        self.layers = []
        x = self.stage1(x)
        # print('stage1:',x.shape)
        self.conv1_layer = x
        x = self.stage2(x)
        # print('stage2:', x.shape)
        self.layers.append(x)
        x = self.stage3(x)
        # print('stage3:', x.shape)
        self.layers.append(x)
        x = self.stage4(x)
        # print('stage4:', x.shape)
        self.layers.append(x)
        x = self.stage5(x)
        # print('stage5:', x.shape)
        self.layers.append(x)
        x = self.stage6(x)
        # print('stage6:', x.shape)
        self.layers.append(x)