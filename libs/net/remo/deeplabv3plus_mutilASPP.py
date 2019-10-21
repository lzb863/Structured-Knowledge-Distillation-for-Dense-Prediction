# encoding: utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from libs.net.sync_batchnorm import SynchronizedBatchNorm2d
from torch.nn import init
from libs.net.backbone import build_backbone
from libs.net.ASPP import *


class deeplabv3plus_mutilAspp(nn.Module):
    def __init__(self, cfg):
        super(deeplabv3plus_mutilAspp, self).__init__()
        self.backbone = None
        self.backbone_layers = None
        input_channel = 2048

        cfg.TRAIN_BN_MOM = 0.1
        self.aspp = ASPP(dim_in=input_channel,
                         dim_out=cfg.MODEL_ASPP_OUTDIM, # 256
                         rate=16 // cfg.MODEL_OUTPUT_STRIDE,
                         bn_mom=cfg.TRAIN_BN_MOM
                         )

        input_channel_layer1 = 512
        # 用于1/8特征图做aspp
        self.aspp_layer1 = ASPP_size4(dim_in=input_channel_layer1,
                         dim_out=cfg.MODEL_ASPP_OUTDIM, # 256
                         rate=16 // cfg.MODEL_OUTPUT_STRIDE,
                         bn_mom=cfg.TRAIN_BN_MOM
                        )
        input_channel_layer0 = 256
        # 用于1/4特征图做aspp
        self.aspp_layer0 = ASPP_size3(dim_in=input_channel_layer0 ,
                         dim_out=cfg.MODEL_ASPP_OUTDIM, # 256
                         rate=16 // cfg.MODEL_OUTPUT_STRIDE,
                         bn_mom=cfg.TRAIN_BN_MOM
                        )

        self.dropout1 = nn.Dropout(0.5)
        self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.upsample_sub = nn.UpsamplingBilinear2d(scale_factor=cfg.MODEL_OUTPUT_STRIDE // 4)
        self.upsample_sub_layer1 = nn.UpsamplingBilinear2d(scale_factor=cfg.MODEL_OUTPUT_STRIDE // 8)

        indim = 256
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(indim, cfg.MODEL_SHORTCUT_DIM, cfg.MODEL_SHORTCUT_KERNEL, 1,
                      padding=cfg.MODEL_SHORTCUT_KERNEL // 2, bias=True),
            SynchronizedBatchNorm2d(cfg.MODEL_SHORTCUT_DIM, momentum=cfg.TRAIN_BN_MOM),
            nn.ReLU(inplace=True),
        )
        self.cat_conv = nn.Sequential(
            nn.Conv2d(cfg.MODEL_ASPP_OUTDIM * 2 + cfg.MODEL_SHORTCUT_DIM, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1,
                      bias=True),
            SynchronizedBatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1, bias=True),
            SynchronizedBatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, SynchronizedBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.backbone = build_backbone(cfg.MODEL_BACKBONE, os=cfg.MODEL_OUTPUT_STRIDE)
        self.backbone_layers = self.backbone.get_layers()

    def forward(self, x):
        x_bottom = self.backbone(x)
        layers = self.backbone.get_layers() # 1/16
        '''
            0    torch.Size([4, 256, 128, 128]) # 1/4
            1    torch.Size([4, 512, 64, 64])   # 1/8 
            2    torch.Size([4, 1024, 32, 32])  # 1/16
            3    torch.Size([4, 2048, 32, 32])  # 1/16 
        '''
        feature_aspp = self.aspp(layers[-1])
        feature_aspp = self.dropout1(feature_aspp) # 1/16

        feature_aspp_layer1 = self.aspp_layer1(layers[1])
        feature_aspp_layer1 = self.dropout1(feature_aspp_layer1) # 1/8

        feature_aspp_layer0 = self.aspp_layer0(layers[0])
        feature_aspp_layer0 = self.dropout1(feature_aspp_layer0)  # 1/4

        feature_aspp = self.upsample_sub(feature_aspp) # 1/4
        feature_aspp_layer1 = self.upsample_sub_layer1(feature_aspp_layer1) # 1/4
        feature_shallow = self.shortcut_conv(feature_aspp_layer0)


        feature_cat = torch.cat([feature_aspp, feature_aspp_layer1, feature_shallow], 1)
        result = self.cat_conv(feature_cat)
        result = self.cls_conv(result)
        result = self.upsample4(result)
        return result
