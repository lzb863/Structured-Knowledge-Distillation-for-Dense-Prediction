# encoding: utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision import models
import libs.net.resnet_atrous as atrousnet
import libs.net.xception as xception
import libs.net.resnet_c as resnet_c
import libs.net.pytorchcvNets.bninception as bninception
import libs.net.pytorchcvNets.resnet as resnet
import libs.net.ding as ding

def build_backbone(backbone_name, pretrained=False, os=16):
    if backbone_name == 'res18_0.5_16_dilation':
        net = ding.get_resnet18_05_backbone(pretrained=pretrained, name="resnet18_wd2", mode="atrous")
        return net
    elif backbone_name == 'res18_0.5_16_deconv':
        net = ding.get_resnet18_05_backbone(pretrained=pretrained, name="resnet18_wd2", mode="deconv")
        return net
    elif backbone_name == 'bninception_16':
        net = bninception.bninception(pretrained=pretrained)
        # net = nn.Sequential(*list(net.children())[:-3])
        return net
    elif backbone_name == 'std_res50_16':
        net = resnet_c.resnet50(pretrained=pretrained)
        # net = nn.Sequential(*list(net.children())[:-3])
        return net
    elif backbone_name == 'res18_atrous':
        net = atrousnet.resnet18_atrous(pretrained=pretrained, os=os)
        return net
    elif backbone_name == 'res50_atrous':
        net = atrousnet.resnet50_atrous(pretrained=pretrained, os=os)
        return net
    elif backbone_name == 'res101_atrous':
        net = atrousnet.resnet101_atrous(pretrained=pretrained, os=os)
        return net
    elif backbone_name == 'res152_atrous':
        net = atrousnet.resnet152_atrous(pretrained=pretrained, os=os)
        return net
    elif backbone_name == 'xception' or backbone_name == 'Xception':
        net = xception.xception(pretrained=pretrained, os=os)
        return net
    else:
        raise ValueError('backbone.py: The backbone named %s is not supported yet.' % backbone_name)
