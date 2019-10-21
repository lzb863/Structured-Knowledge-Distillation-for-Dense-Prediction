# encoding: utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from libs.net.sync_batchnorm import SynchronizedBatchNorm2d
from torch.nn import init
from libs.net.backbone import build_backbone
from libs.net.ASPP import ASPP


class deeplabv3plus_nodilation(nn.Module):
    def __init__(self, args):
        super(deeplabv3plus_nodilation, self).__init__()
        self.backbone = None
        self.backbone_layers = None
        input_channel = 2048

        args.TRAIN_BN_MOM = 0.1
        self.dropout1 = nn.Dropout(0.5)
        self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.upsample_sub = nn.UpsamplingBilinear2d(scale_factor=args.MODEL_OUTPUT_STRIDE // 4)

        self.cat_conv = nn.Sequential(
            nn.Conv2d(args.MODEL_ASPP_OUTDIM + args.MODEL_SHORTCUT_DIM, args.MODEL_ASPP_OUTDIM, 3, 1, padding=1,
                      bias=True),
            SynchronizedBatchNorm2d(args.MODEL_ASPP_OUTDIM, momentum=args.TRAIN_BN_MOM),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(args.MODEL_ASPP_OUTDIM, args.MODEL_ASPP_OUTDIM, 3, 1, padding=1, bias=True),
            SynchronizedBatchNorm2d(args.MODEL_ASPP_OUTDIM, momentum=args.TRAIN_BN_MOM),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(args.MODEL_ASPP_OUTDIM, args.num_classes, 1, 1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, SynchronizedBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.backbone = build_backbone(args.MODEL_BACKBONE, os=args.MODEL_OUTPUT_STRIDE)
        indim4 = self.backbone.outdim_4
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(indim4, args.MODEL_SHORTCUT_DIM, args.MODEL_SHORTCUT_KERNEL, 1,
                      padding=args.MODEL_SHORTCUT_KERNEL // 2, bias=True),
            SynchronizedBatchNorm2d(args.MODEL_SHORTCUT_DIM, momentum=args.TRAIN_BN_MOM),
            nn.ReLU(inplace=True),
        )
        indim16 = self.backbone.outdim_16
        self.aspp_conv = nn.Sequential(
            nn.Conv2d(indim16, args.MODEL_ASPP_OUTDIM, 1, 1, padding=0, bias=True),
            SynchronizedBatchNorm2d(args.MODEL_ASPP_OUTDIM, momentum=args.TRAIN_BN_MOM),
            nn.ReLU(inplace=True),
        )
        # self.backbone_layers = self.backbone.get_layers()

    def forward(self, x):
        x_bottom = self.backbone(x)
        feature_aspp = self.aspp_conv(x_bottom)
        feature_aspp = self.dropout1(feature_aspp) # 1/8
        feature_aspp = self.upsample_sub(feature_aspp) # 1/4
        feature_shallow = self.shortcut_conv(self.backbone.layers[0])
        feature_cat = torch.cat([feature_aspp, feature_shallow], 1)
        result = self.cat_conv(feature_cat)
        result = self.cls_conv(result)
        result = self.upsample4(result)
        return result
    
    # for ts
    def forward_till_aspp(self, x):
        x_bottom = self.backbone(x)
        feature_aspp = self.aspp_conv(x_bottom)
        # layers = self.backbone.get_layers() # 1/16
        return feature_aspp

    def aspp_to_catFeat(self, feature_aspp):
        feature_aspp = self.dropout1(feature_aspp) # 1/8
        feature_aspp = self.upsample_sub(feature_aspp) # 1/4
        # layers = self.backbone.get_layers() # 1/16
        feature_shallow = self.shortcut_conv(self.backbone.layers[0])
        feature_cat = torch.cat([feature_aspp, feature_shallow], 1)
        return feature_cat

    def catFeat_to_predict(self, feature_cat):
        result = self.cat_conv(feature_cat)
        result = self.cls_conv(result)
        result = self.upsample4(result)
        return result
