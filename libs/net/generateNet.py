# encoding: utf-8
import torch
import torch.nn as nn

from libs.net.deeplabv3plus import deeplabv3plus
from libs.net.deeplabv3plus_nodilation import deeplabv3plus_nodilation
from libs.net.deeplabv3plus_res18 import deeplabv3plus_res18

'''
self.INIT_model = edict({
    "type": "deeplabv3plus",
    "args":{
        'num_classes': self.num_classes,
        "MODEL_BACKBONE": '',
        'MODEL_OUTPUT_STRIDE': 16,
        'MODEL_ASPP_OUTDIM': 256,
        'MODEL_SHORTCUT_DIM': 48,
        'MODEL_SHORTCUT_KERNEL': 1,
    }
})
'''
from libs.net.remo.deeplabv3plusFpn import deeplabv3plus_fpn
from libs.net.remo.deeplabv3plusFpn_multiDecod import  deeplabv3plus_fpn_multidecode
from libs.net.remo.deeplabv3plus_mutilASPP import deeplabv3plus_mutilAspp
from libs.net.remo.deeplabv3plus_baseOc import deeplabv3plus_baseOc

from libs.net.unet_se.Unet34_scSE_hyper import Unet_scSE_hyper as unet34_scSE_hyper

from libs.net.unet_se.Unet32_scSE_hyper import Unet_scSE_hyper as unet32_scSE_hyper

from libs.net.unetBase.unet_base import UnetBase
'''
self.INIT_model = edict({
    "type": "UnetBase",
    "args":{
        'num_classes': self.MODEL_NUM_CLASSES,
    }
})
'''

from libs.net.Unet.unet import remo_UNet_oc
'''
self.INIT_model = edict({
    "type": "remo_UNet_oc",
    "args":{
        "basenet": "se_resnet50",
        "num_filters": 16,
        "pretrained": 'imagenet',
        "num_classes": self.MODEL_NUM_CLASSES
    }
})

basenet = ('vgg11', 'vgg13', 'vgg16', 'vgg19',
                   'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn',
                   'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                   'resnext101_32x4d', 'resnext101_64x4d',
                   'se_resnet50', 'se_resnet101', 'se_resnet152',
                   'se_resnext50_32x4d', 'se_resnext101_32x4d', 'senet154',
                   'darknet')
'''
from libs.net.Unet.unet import  remo_UNet

from libs.net.UnetPlusPlus.unet_plus import SE_Res50UNet

from libs.net.TSmodule import TSmodule
'''
self.INIT_model = edict({
    "type": "SE_Res50UNet",
    "args":{
        'num_classes': self.MODEL_NUM_CLASSES,
        'num_filters': 32,
        'cls_only': False,
        'is_deconv': False,
        'pretrained': 'None'
        
    }
})
'''